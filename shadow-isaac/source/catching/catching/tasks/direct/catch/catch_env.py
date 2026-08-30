"""The catch environment: wiring, and as little logic as possible.

NOTE: unvalidated. This file imports isaaclab and therefore cannot run -- or
even import -- on the laptop it was written on. That is exactly why it holds no
task logic. Phases, reward and termination live in `rewards.py`, which imports
no isaaclab and is pinned against the NumPy reference in `harness/catch.py` by
`tests/test_parity.py`. What is left here is buffer bookkeeping and asset
plumbing, which is the part a simulator has to check anyway.

The one judgement call that cannot be tested anywhere: the hand must rest PALM
UP. Gravity is what holds a caught ball. Check it in the viewer before reading
anything into a training curve.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg  # noqa: F401  (re-exported for cfg typing)
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from . import grasp as gp
from . import joints as jt
from . import rewards as rw
from .catch_env_cfg import CatchEnvCfg


class CatchEnv(DirectRLEnv):
    cfg: CatchEnvCfg

    def __init__(self, cfg: CatchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._task_cfg = rw.CatchCfg(
            is_drop_stage=cfg.stage == "drop",
            is_pop_stage=cfg.stage == "pop",
            settle_steps=cfg.settle_steps,
            episode_len=self.max_episode_length,
            catch_radius=cfg.catch_radius, secure_speed=cfg.secure_speed,
            min_contacts=cfg.min_contacts, hold_steps=cfg.hold_steps,
            drop_below=cfg.drop_below, drop_lateral=cfg.drop_lateral,
            max_force=cfg.max_force, launch_deadline=cfg.launch_deadline,
            track_scale=cfg.track_scale, secure_bonus=cfg.secure_bonus,
            hold_scale=cfg.hold_scale, apex_scale=cfg.apex_scale,
            drop_penalty=cfg.drop_penalty, force_penalty=cfg.force_penalty,
            action_rate_penalty=cfg.action_rate_penalty, ctrl_penalty=cfg.ctrl_penalty,
            ctrl_dt=cfg.sim.dt * cfg.decimation, ball_radius=cfg.ball_radius,
            # The fall the ball actually makes: clearance above the fingertips
            # PLUS the fingertips' height above the palm.
            spawn_height=cfg.spawn_height + cfg.hand_reach,
        )
        # Checked here rather than trusted. The first run on the box shipped a
        # configuration this rejects -- 12 control steps of fall -- and nothing
        # noticed, because the validator existed only on the laptop side.
        self._task_cfg.validate()

        # 24 joints, 20 actuators: the four fingers' distal joints are driven
        # by fixed tendons, not actuators, and they sit at indices 17, 18, 19
        # and 22 -- so range(20) is wrong in both directions. See joints.py.
        self._act_idx = jt.actuated_indices(self.hand.data.joint_names,
                                            expected=cfg.action_space)

        bn = list(self.hand.data.body_names)
        self._knuckle_idx = [bn.index(n) for n in
                             ("robot0_ffknuckle", "robot0_mfknuckle",
                              "robot0_rfknuckle", "robot0_lfmetacarpal")]
        self._tipmeas_idx = [bn.index(n) for n in
                             ("robot0_ffdistal", "robot0_mfdistal",
                              "robot0_rfdistal", "robot0_lfdistal")]
        self._palm_idx = self.hand.find_bodies(cfg.palm_body_expr)[0]
        self._tip_idx = self.hand.find_bodies(cfg.contact_body_expr)[0]
        if len(self._palm_idx) != 1:
            raise RuntimeError(
                f"palm_body_expr {cfg.palm_body_expr!r} matched {len(self._palm_idx)} bodies; "
                f"expected exactly one. Run `make bodies` to see the real names.")
        if len(self._tip_idx) != cfg.num_contact_bodies:
            raise RuntimeError(
                f"contact_body_expr {cfg.contact_body_expr!r} matched "
                f"{len(self._tip_idx)} bodies, expected {cfg.num_contact_bodies}. A regex "
                f"that matches nothing reports no contacts instead of failing, so this is "
                f"checked.")

        z = lambda d=torch.float32: torch.zeros(self.num_envs, device=self.device, dtype=d)  # noqa: E731
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._flights = z(torch.long)
        self._held_steps = z(torch.long)
        self._peak_height = z()
        self._phase = z(torch.long)
        self._reason = z(torch.long)
        self._checked_obs_width = False
        self._reported_reach = False
        self._checked_palm = False
        # Lateral offset from the palm origin to the middle of the cup. Zero
        # until the geometry can be read: body positions are only valid once
        # the reset pose has been stepped, so the first episode drops on the
        # palm origin and every one after that is corrected.
        self._spawn_offset = torch.zeros(3, device=self.device)

        # Reset into a CUPPED hand, not the asset's default pose. The default
        # holds the fingers straight, which put the fingertips 16.9 cm above the
        # palm -- so a ball spawned clear of them fell 21 cm onto the tips and
        # rolled off. Resetting cupped lowers the fingers and gives the ball a
        # pocket to settle into. It is a property of the task, not of the
        # scripted controller, so it lives here.
        lim = self.hand.data.soft_joint_pos_limits[0].cpu().numpy()
        self._reset_qpos = torch.as_tensor(
            gp.pose_to_qpos(gp.OPEN_POSE, list(self.hand.data.joint_names),
                            lim[:, 0], lim[:, 1]),
            device=self.device).unsqueeze(0)

    # -- scene ------------------------------------------------------------

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        self.contacts = ContactSensor(self.cfg.contact_cfg)
        self.ball_contacts = ContactSensor(self.cfg.ball_contact_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.sensors["contacts"] = self.contacts
        self.scene.sensors["ball_contacts"] = self.ball_contacts

        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)

    # -- control ----------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        limits = self.hand.data.soft_joint_pos_limits[:, self._act_idx, :]
        lo, hi = limits[..., 0], limits[..., 1]
        target = lo + 0.5 * (self._actions * self.cfg.action_scale + 1.0) * (hi - lo)
        self.hand.set_joint_position_target(target, joint_ids=self._act_idx)

    # -- task state -------------------------------------------------------

    @property
    def _palm_pos(self) -> torch.Tensor:
        return self.hand.data.body_pos_w[:, self._palm_idx[0], :]

    def _contact_forces(self) -> torch.Tensor:
        """Per-body net contact force magnitude, over every surface that can
        carry the ball -- palm, proximal, middle and distal.

        Reported in log1p, as in the MuJoCo sibling: contact forces span three
        orders of magnitude between a graze and a full press, and a network fed
        raw newtons spends its input range on the rare hard ones.
        """
        f = self.contacts.data.net_forces_w  # (envs, bodies, 3)
        return torch.linalg.norm(f, dim=-1)

    def _ball_force(self) -> torch.Tensor:
        """Net contact force on the ball: the hand-ball force, by elimination.

        The ball touches nothing but the hand while an episode is live -- ground
        contact means it has already terminated as `dropped` -- so this needs no
        filtering, which is what makes it usable at all: Isaac Lab can only
        filter a sensor that matches one prim per environment.
        """
        f = self.ball_contacts.data.net_forces_w  # (envs, 1, 3)
        return torch.linalg.norm(f, dim=-1).squeeze(-1)

    def _check_palm_is_up(self) -> None:
        """Refuse to run a hand that is not palm up.

        This is the check whose absence cost the most. The asset was left lying
        on its side with the palm facing sideways, and nothing complained: the
        scripted catcher reported 100% caught, because a ball wedged in a
        sideways crook is slow, touching, above the palm origin and laterally
        centred -- every is_secured condition, and not a catch. A green metric
        is not evidence the scene is right.

        The palm's sign is anchored on the finger curl: the reset pose is
        cupped, so the fingertips lie on the palmar side of the knuckle row.
        """
        bp = self.hand.data.body_pos_w[0]
        palm = bp[self._palm_idx[0]].cpu().numpy()
        knuck = bp[self._knuckle_idx].cpu().numpy()
        tips = bp[self._tipmeas_idx].cpu().numpy()

        across = knuck[-1] - knuck[0]
        along = knuck.mean(axis=0) - palm
        n = np.cross(across / np.linalg.norm(across), along / np.linalg.norm(along))
        n /= np.linalg.norm(n)
        if np.dot(tips.mean(axis=0) - knuck.mean(axis=0), n) < 0:
            n = -n

        # Toward the fingers, lateral only -- the height is hand_reach's job.
        toward = knuck.mean(axis=0) - palm
        toward[2] = 0.0
        self._spawn_offset = torch.as_tensor(
            self.cfg.spawn_toward_knuckles * toward, dtype=torch.float32,
            device=self.device)
        print(f"[catch] dropping {np.linalg.norm(self._spawn_offset.cpu().numpy())*100:.1f} cm "
              f"toward the fingers from the palm origin")

        tilt = float(np.degrees(np.arccos(float(np.clip(n[2], -1.0, 1.0)))))
        print(f"[catch] palm normal {np.round(n, 3)}, {tilt:.1f} deg off vertical")
        if tilt > self.cfg.max_palm_tilt_deg:
            raise RuntimeError(
                f"the palm is {tilt:.1f} deg off vertical (normal {np.round(n, 3)}); "
                f"gravity is what holds a caught ball, so a hand on its side cannot "
                f"catch. Run `make palmup` to find an orientation that works and put "
                f"it in CatchEnvCfg.hand_rot.")

    def _read_state(self):
        """Read the simulator. Pure -- mutates nothing, so it is safe to call
        from any hook in any order.

        There is deliberately no cache here. `reset()` calls
        `_get_observations()` directly, without `_get_dones()` ever having run,
        so a cache filled in the dones hook is an AttributeError on the very
        first reset. It is also not merely absent on reset: the counters below
        are incremented, so a second call per step would double-count held
        steps. Reading is cheap (norms over five bodies); the mutation is what
        must happen exactly once, and it lives in `_update_counters`.
        """
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        palm = self._palm_pos
        forces = self._contact_forces()          # tactile observation, 16 bodies
        ball_force = self._ball_force()           # the hand-ball force
        # A count of 0 or 1: "is the hand touching the ball at all". Kept as a
        # count so the reward code is unchanged, and `min_contacts` still reads
        # as a threshold on it. Two fingertips was never a definition of held --
        # the kinematics carry the criterion, and a ball in free fall gains
        # 2.45 m/s over the hold window, so it cannot fake them.
        contacts = (ball_force > self.cfg.contact_threshold).long()
        return ball_pos, ball_vel, contacts, ball_force, palm, forces

    def _update_counters(self, state) -> None:
        """The mutating half. Called exactly once per step, from `_get_dones`."""
        ball_pos, ball_vel, contacts, max_force, palm, _ = state
        self._peak_height = torch.maximum(self._peak_height, ball_pos[:, 2])
        self._phase = rw.phase_of(self.episode_length_buf, ball_pos, ball_vel,
                                  contacts, palm, self._flights, self._task_cfg)
        self._flights = torch.where(self._phase == rw.FLIGHT,
                                    torch.ones_like(self._flights), self._flights)
        self._held_steps = torch.where(self._phase == rw.SECURED,
                                       self._held_steps + 1,
                                       torch.zeros_like(self._held_steps))
        self._reason = rw.terminate(ball_pos, palm, max_force, self.episode_length_buf,
                                    self._held_steps, self._flights, self._task_cfg)
        # What the termination check actually saw, kept for diagnosis: the
        # simulator state mid-step is not what a script can read after it,
        # because terminated environments have already been reset by then.
        self._dbg_force = max_force.detach().clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Checked here, not at reset: `write_root_pose_to_sim` has not been
        # stepped at reset time, so body_pos_w still describes the old pose and
        # the check would read a hand that no longer exists.
        if not self._checked_palm and int(self.episode_length_buf.max()) > 3:
            self._check_palm_is_up()
            self._checked_palm = True
        self._update_counters(self._read_state())
        timeout = self._reason == rw.TIMEOUT
        terminated = (self._reason != rw.NONE) & ~timeout
        return terminated, timeout

    def _get_rewards(self) -> torch.Tensor:
        ball_pos, ball_vel, contacts, max_force, palm, _ = self._read_state()
        terms = rw.reward_terms(ball_pos, ball_vel, contacts, max_force, palm,
                                self._actions, self._prev_actions, self._phase,
                                self._held_steps, self._peak_height, self._task_cfg)
        reward = rw.total(terms)
        reward = torch.where(self._reason == rw.DROPPED,
                             reward - self._task_cfg.drop_penalty, reward)
        # Named terms on the extras, so an ablation or a diagnosis costs nothing.
        self.extras["log"] = {f"reward/{k}": v.mean() for k, v in terms.items()}
        self.extras["log"]["task/caught"] = (self._reason == rw.CAUGHT).float().mean()
        self.extras["log"]["task/dropped"] = (self._reason == rw.DROPPED).float().mean()
        self.extras["log"]["task/never_launched"] = (
            self._reason == rw.NEVER_LAUNCHED).float().mean()
        return reward

    # -- observations -----------------------------------------------------

    def _get_observations(self) -> dict:
        ball_pos, ball_vel, _, _, palm, forces = self._read_state()
        xy, valid = rw.predict_intercept_xy(ball_pos, ball_vel, palm[:, 2])
        obs = torch.cat([
            self.hand.data.joint_pos,
            0.2 * self.hand.data.joint_vel,
            ball_pos - palm,
            ball_vel,
            (xy - palm[:, :2]) * valid.unsqueeze(-1),
            valid.float().unsqueeze(-1),
            torch.log1p(forces),
            self._actions,
        ], dim=-1)
        if not self._checked_obs_width:
            if obs.shape[-1] != self.cfg.observation_space:
                raise RuntimeError(
                    f"observation is {obs.shape[-1]} wide but observation_space is "
                    f"{self.cfg.observation_space}. Most likely num_hand_dofs "
                    f"({self.cfg.num_hand_dofs}) does not match this asset's "
                    f"{self.hand.data.joint_pos.shape[-1]}.")
            self._checked_obs_width = True
        return {"policy": obs}

    # -- reset ------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Place the hand explicitly every reset. See CatchEnvCfg.hand_rot: the
        # spawn-time path does not give the same world orientation.
        root = torch.zeros(n, 7, device=self.device)
        root[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(
            self.cfg.hand_pos, device=self.device)
        root[:, 3:] = torch.tensor(self.cfg.hand_rot, device=self.device)
        self.hand.write_root_pose_to_sim(root, env_ids=env_ids)

        joint_pos = self._reset_qpos.expand(n, -1).clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        palm = self._palm_pos[env_ids]
        jitter = self.cfg.spawn_jitter * (2.0 * torch.rand(n, 2, device=self.device) - 1.0)
        pos = palm.clone()
        pos[:, :2] += jitter + self._spawn_offset[:2]
        # Clear of the FINGERS, not of the palm origin. Spawning at
        # palm_z + spawn_height + radius put a 3 cm ball at 8-14 cm above the
        # palm origin, and this hand's fingers reach ~9-10 cm: the ball was
        # spawned overlapping the fingertips, and PhysX resolved the
        # interpenetration by flinging it out at 253 N -- every episode, every
        # reset, tripping the force limit before the task began. The top of the
        # hand is measured rather than assumed, because it moves with the pose.
        # Spawn from the palm plus the CONFIGURED reach, not from the body
        # positions read here: `write_joint_state_to_sim` above has not been
        # stepped yet, so those still describe the previous pose. The measured
        # value is reported below so a drift between the two is visible.
        pos[:, 2] = (palm[:, 2] + self.cfg.hand_reach
                     + self.cfg.ball_radius + self.cfg.spawn_height)
        top = self.hand.data.body_pos_w[env_ids][:, self._tip_idx, 2].max(dim=-1).values
        if not self._reported_reach:
            reach = float((top - palm[:, 2]).mean())
            print(f"[catch] measured hand reach {reach*100:.1f} cm above the palm "
                  f"(cfg.hand_reach is {self.cfg.hand_reach*100:.1f} cm); "
                  f"total fall {(reach + self.cfg.spawn_height)*100:.1f} cm")
            self._reported_reach = True

        vel = torch.zeros(n, 6, device=self.device)
        if self.cfg.stage == "toss":
            # Launched to reach exactly spawn_height above the release point.
            vel[:, 2] = torch.sqrt(torch.tensor(2.0 * 9.81 * self.cfg.spawn_height,
                                                device=self.device))
        elif self.cfg.stage == "pop":
            # Nothing is launched: the fingers have to do it. The ball starts
            # resting on the hand, just clear of it rather than inside it.
            pos[:, 2] = palm[:, 2] + self.cfg.hand_reach + self.cfg.ball_radius + 0.005

        root = torch.zeros(n, 7, device=self.device)
        root[:, :3] = pos
        root[:, 3] = 1.0
        self.ball.write_root_pose_to_sim(root, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(vel, env_ids=env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._flights[env_ids] = 0
        self._held_steps[env_ids] = 0
        self._peak_height[env_ids] = palm[:, 2]
