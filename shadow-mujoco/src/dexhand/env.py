"""In-hand cube reorientation: the task, as a Gymnasium environment.

The task is the community-standard dexterity benchmark (OpenAI 2018, DeXtreme
2022): a cube rests in an upturned palm; the policy must rotate it to a target
orientation using only finger motion, without dropping it. Success is orientation
within `success_thresh` radians, at which point a new target is sampled — so one
episode chains as many reorientations as the policy can manage before timeout
or a drop.

Three observation views are produced every step, because the training recipe
needs all three at different times:

    actor    what the teacher policy sees during RL: full privileged state
    critic   what the value function sees: same privileged state (the
             asymmetric seam — when a student replaces the actor, the critic
             keeps its privileged inputs and its trained weights)
    student  what could exist on hardware: noisy joint positions, previous
             action, fingertip tactile, goal. No cube pose — that is the
             point; a student closes the gap with history and tactile.

Action processing applies an exponential moving average before the position
targets reach the actuators. This is not cosmetic: raw PPO exploration noise
at 20 Hz commands the fingers to thrash, which both breaks grasps in sim and
is exactly what "smooth, precise, safe hardware actions" forbids on a real
hand. The EMA is part of the environment, so the policy learns THROUGH the
smoothing rather than being smoothed after the fact (which changes the policy's
effective dynamics and invalidates what it learned).

Timing: physics at model.opt.timestep (2 ms for the vendored hands), control at
`ctrl_dt` via frame-skipping. Reset settles the cube onto the palm for
`settle_steps` control periods before the episode starts — without this, every
episode opens with a falling cube and the return signal is dominated by
ballistics no policy controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np

from dexhand import rotations as rot
from dexhand.hands import HandSpec, get_hand
from dexhand.randomize import DomainRandomizer, DRConfig
from dexhand.scene import Scene, build_scene, settle_check
from dexhand.tactile import fingertip_forces


@dataclass
class EnvConfig:
    ctrl_dt: float = 0.05           # 20 Hz control, the rate the ad's hands run near
    episode_len: int = 200          # control steps (10 s)
    settle_steps: int = 10          # un-rewarded control periods after reset

    # -- goals ------------------------------------------------------------
    goal_mode: str = "z"            # "z": rotations about world z | "full": uniform SO(3)
    max_goal_angle: float = np.pi / 2   # z-mode only; curriculum widens this
    success_thresh: float = 0.4     # radians (the OpenAI/DeXtreme convention)
    max_goals: int = 50             # safety valve on goal cycling

    # -- termination ------------------------------------------------------
    drop_below: float = 0.06        # cube centre this far under the palm = dropped
    drop_lateral: float = 0.15      # or this far sideways from the palm

    # -- action -----------------------------------------------------------
    action_ema: float = 0.7         # target = ema*prev + (1-ema)*new

    # -- reward -----------------------------------------------------------
    # dense term 1/(dist+eps) follows the widely replicated IsaacGymEnvs
    # shadow-hand baseline; bonuses/penalties are rescaled to our dense range
    dist_eps: float = 0.1
    rot_reward_scale: float = 1.0
    success_bonus: float = 50.0
    drop_penalty: float = 20.0
    action_rate_penalty: float = 0.01
    ctrl_penalty: float = 0.001

    dr: DRConfig = field(default_factory=DRConfig)


class ReorientEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, hand: str | HandSpec = "shadow", cfg: Optional[EnvConfig] = None,
                 seed: Optional[int] = None, render_mode: Optional[str] = None,
                 check_palm: bool = True):
        self.cfg = cfg or EnvConfig()
        self.hand = get_hand(hand) if isinstance(hand, str) else hand
        self.scene: Scene = build_scene(self.hand)
        if check_palm:
            held, pos = settle_check(self.scene)
            if not held:
                raise RuntimeError(
                    f"hand '{self.hand.name}' does not hold the cube: after settling it is at "
                    f"{np.round(pos, 3)}. The attach_quat/cube_spawn in hands.py are wrong for "
                    "this model — every episode would be a one-step drop."
                )
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.frame_skip = max(1, int(round(self.cfg.ctrl_dt / self.model.opt.timestep)))
        self.dr = DomainRandomizer(self.scene, self.cfg.dr)
        self.rng = np.random.default_rng(seed)

        # hand joints = every dof that is not the cube's free joint
        cube_dofs = set(range(self.scene.cube_qvel_adr, self.scene.cube_qvel_adr + 6))
        self.hand_dof_ids = np.array([i for i in range(self.model.nv) if i not in cube_dofs])
        cube_qs = set(range(self.scene.cube_qpos_adr, self.scene.cube_qpos_adr + 7))
        self.hand_qpos_ids = np.array([i for i in range(self.model.nq) if i not in cube_qs])

        self.nu = self.model.nu
        self.n_tips = len(self.scene.fingertip_body_ids)
        self._ctrl_mid = self.scene.ctrl_range.mean(axis=1)
        self._ctrl_half = 0.5 * (self.scene.ctrl_range[:, 1] - self.scene.ctrl_range[:, 0])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.nu,), dtype=np.float32)
        dims = self._obs_dims()
        self.observation_space = gym.spaces.Dict(
            {k: gym.spaces.Box(-np.inf, np.inf, (d,), dtype=np.float32) for k, d in dims.items()}
        )

        self._smoothed = np.zeros(self.nu)
        self._prev_action = np.zeros(self.nu)
        self._delay_buf: list = []
        self._goal = np.array([1.0, 0.0, 0.0, 0.0])
        self._steps = 0
        self._successes = 0

    # -- observation layout ----------------------------------------------

    def _obs_dims(self) -> Dict[str, int]:
        nj = len(self.hand_qpos_ids)
        state = nj + len(self.hand_dof_ids) + 3 + 4 + 3 + 3 + 4 + 4 + self.n_tips + self.nu
        student = nj + self.nu + self.n_tips + 4
        return {"actor": state, "critic": state, "student": student}

    def _cube_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        a = self.scene.cube_qpos_adr
        return self.data.qpos[a:a + 3].copy(), rot.quat_normalize(self.data.qpos[a + 3:a + 7].copy())

    def _obs(self) -> Dict[str, np.ndarray]:
        qpos = self.data.qpos[self.hand_qpos_ids].copy()
        qvel = self.data.qvel[self.hand_dof_ids].copy()
        cube_pos, cube_quat = self._cube_pose()
        palm = self.data.xpos[self.scene.palm_body_id]
        v = self.scene.cube_qvel_adr
        linvel = self.data.qvel[v:v + 3].copy()
        angvel = self.data.qvel[v + 3:v + 6].copy()
        diff = rot.quat_mul(self._goal, rot.quat_conj(cube_quat))
        tactile = fingertip_forces(self.model, self.data, self.scene.fingertip_geom_ids,
                                   against_geom=self.scene.cube_geom_id)

        state = np.concatenate([
            qpos, 0.2 * qvel, cube_pos - palm, cube_quat, linvel, 0.2 * angvel,
            self._goal, diff, tactile, self._prev_action,
        ]).astype(np.float32)

        n = self.cfg.dr
        noisy_qpos = qpos + self.rng.normal(0, n.obs_noise_joint, qpos.shape) if n.enabled else qpos
        student = np.concatenate([
            noisy_qpos, self._prev_action, tactile, self._goal,
        ]).astype(np.float32)

        return {"actor": state, "critic": state.copy(), "student": student}

    # -- goals ------------------------------------------------------------

    def _sample_goal(self, cube_quat: np.ndarray) -> np.ndarray:
        """A goal is only a goal if it starts OUTSIDE the success threshold.

        Without the floor, easy curricula (max_goal_angle near the threshold)
        sample goals the cube already satisfies, and the policy collects the
        success bonus for doing nothing — a training run that looks like rapid
        mastery and is actually a broken reward. Caught live: the first smoke
        run reported the max_goals cap of successes at iteration 1.
        """
        cfg = self.cfg
        floor = cfg.success_thresh + 0.1
        if cfg.goal_mode == "z":
            hi = max(cfg.max_goal_angle, floor + 1e-6)
            mag = self.rng.uniform(floor, hi)
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            delta = rot.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), sign * mag)
            return rot.quat_normalize(rot.quat_mul(delta, cube_quat))
        if cfg.goal_mode == "full":
            for _ in range(20):
                q = rot.random_quat(self.rng)
                if rot.rot_dist(q, cube_quat) > floor:
                    return q
            return q  # 20 misses at floor ~0.5 rad is astronomically unlikely
        raise ValueError(f"unknown goal_mode {self.cfg.goal_mode!r}")

    def _set_goal(self, q: np.ndarray) -> None:
        self._goal = q
        self.data.mocap_quat[self.scene.goal_mocap_id] = q

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg

        self.dr.randomize(self.rng)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)  # derived quantities follow the DR draw

        noise = cfg.dr.init_qpos_noise if cfg.dr.enabled else 0.0
        self.data.qpos[self.hand_qpos_ids] += self.rng.normal(0, noise, len(self.hand_qpos_ids))
        a = self.scene.cube_qpos_adr
        self.data.qpos[a:a + 3] = np.asarray(self.hand.cube_spawn) + \
            np.concatenate([self.rng.uniform(-0.01, 0.01, 2), [0.0]])
        self.data.qpos[a + 3:a + 7] = rot.random_z_quat(self.rng, np.pi)

        self._smoothed[:] = 0.0
        self._prev_action[:] = 0.0
        self._delay_buf = [np.zeros(self.nu)] * self.dr.action_delay

        # settle: let the cube land before anything is scored
        self.data.ctrl[:] = self._ctrl_mid
        for _ in range(cfg.settle_steps * self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        _, cube_quat = self._cube_pose()
        self._set_goal(self._sample_goal(cube_quat))
        self._steps = 0
        self._successes = 0

        return self._obs(), {"successes": 0}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        if self.dr.action_delay > 0:
            self._delay_buf.append(action.copy())
            applied = self._delay_buf.pop(0)
        else:
            applied = action

        self._smoothed = cfg.action_ema * self._smoothed + (1 - cfg.action_ema) * applied
        self.data.ctrl[:] = self._ctrl_mid + self._smoothed * self._ctrl_half
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        cube_pos, cube_quat = self._cube_pose()
        palm = self.data.xpos[self.scene.palm_body_id]
        dist = rot.rot_dist(cube_quat, self._goal)

        dropped = bool(
            cube_pos[2] < palm[2] - cfg.drop_below
            or np.linalg.norm(cube_pos[:2] - palm[:2]) > cfg.drop_lateral
        )
        success = bool(dist < cfg.success_thresh) and not dropped

        reward = cfg.rot_reward_scale / (dist + cfg.dist_eps)
        reward -= cfg.action_rate_penalty * float(np.sum((action - self._prev_action) ** 2))
        reward -= cfg.ctrl_penalty * float(np.sum(action ** 2))
        if success:
            reward += cfg.success_bonus
            self._successes += 1
            self._set_goal(self._sample_goal(cube_quat))
            dist = rot.rot_dist(cube_quat, self._goal)
        if dropped:
            reward -= cfg.drop_penalty

        self._prev_action = action
        self._steps += 1

        terminated = dropped or self._successes >= cfg.max_goals
        truncated = self._steps >= cfg.episode_len and not terminated
        info = {
            "rot_dist": float(dist),
            "is_success": success,
            "successes": self._successes,
            "dropped": dropped,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="task_view")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
