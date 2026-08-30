"""Run the scripted catcher and report what happens.

The gate before PPO. This controller sees ground-truth ball state and closes
the hand on a closed-form prediction of when the ball arrives at palm height.
If it cannot catch, the fault is the environment -- unreachable `is_secured`
thresholds, a ball with too much restitution, a palm facing the wrong way -- and
no amount of learning will fix it. Same role `stack_sm.py` plays next door.

Read the printed table, not just the catch rate:

* mostly `dropped`      the hand closes too late, too early, or too weakly.
                        Sweep --lead-time first; it is the one real knob.
* mostly `out_of_bounds` the ball is not landing on the palm at all, which is
                        a spawn or hand-orientation problem, not a timing one.
* catches but no holds  `secure_speed` or `hold_steps` is unreachable; the ball
                        is being touched and bounced rather than cradled.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Catch-Shadow-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--spawn-toward", type=float, default=None,
                    help="override CatchEnvCfg.spawn_toward_knuckles (0=palm origin, "
                         "1=knuckle row)")
parser.add_argument("--debug-steps", type=int, default=0,
                    help="print raw sensor and termination state for N steps")
parser.add_argument("--lead-time", type=float, default=0.08,
                    help="seconds before arrival at which the hand starts closing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import catching  # noqa: E402, F401
from catching.tasks.direct.catch import grasp, rewards as rw  # noqa: E402

REASONS = ["running", "dropped", "out_of_bounds", "excess_force",
           "caught", "never_launched", "timeout"]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    if args_cli.spawn_toward is not None:
        env_cfg.spawn_toward_knuckles = args_cli.spawn_toward
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    # Poses are written in radians and converted through the limits the
    # simulator actually reports -- the hand's ranges do not share a sign
    # convention, so a grasp written in action space gets the thumb backwards.
    names = [u.hand.data.joint_names[i] for i in u._act_idx]
    limits = u.hand.data.soft_joint_pos_limits[0, u._act_idx, :].cpu().numpy()
    open_a = grasp.pose_to_action(grasp.OPEN_POSE, names, limits[:, 0], limits[:, 1])
    closed_a = grasp.pose_to_action(grasp.GRASP_POSE, names, limits[:, 0], limits[:, 1])
    print(f"[scripted] {len(names)} actuated joints, lead time {args_cli.lead_time*1000:.0f} ms")

    # Where the hand ACTUALLY points, in the running environment. `make orient`
    # measures the stock SHADOW_HAND_CFG and so cannot verify the rotation this
    # env applies -- which is how a hand rotated onto its side passed as
    # "palm up" while the metric read 100% caught. A ball wedged in a sideways
    # crook satisfies every is_secured condition without being a catch.
    env_names = list(u.hand.data.body_names)
    bp = u.hand.data.body_pos_w[0].cpu().numpy()

    def body(nm):
        return bp[env_names.index(nm)]

    palm_p = body("robot0_palm")
    knuck = np.array([body(n) for n in ["robot0_ffknuckle", "robot0_mfknuckle",
                                        "robot0_rfknuckle", "robot0_lfmetacarpal"]])
    tips_p = np.array([body(n) for n in ["robot0_ffdistal", "robot0_mfdistal",
                                         "robot0_rfdistal", "robot0_lfdistal"]])
    across = knuck[-1] - knuck[0]
    across /= np.linalg.norm(across)
    along = knuck.mean(axis=0) - palm_p
    along /= np.linalg.norm(along)
    normal = np.cross(across, along)
    normal /= np.linalg.norm(normal)
    # Sign anchored on the CURL, not on the thumb: the reset pose is cupped, so
    # the fingertips lie on the palmar side of the knuckle row. That is a fact
    # about this pose rather than an assumption about thumb anatomy.
    if np.dot(tips_p.mean(axis=0) - knuck.mean(axis=0), normal) < 0:
        normal = -normal
    thumb = body("robot0_thdistal") - palm_p
    thumb /= np.linalg.norm(thumb)
    print("[orient] BEFORE the first reset -- the hand is placed at reset, not at")
    print("[orient] spawn, so this is the asset's spawn pose. The env prints the")
    print("[orient] real one as '[catch] palm normal ...' a few steps in.")
    print(f"[orient] spawn-pose palm normal {np.round(normal, 3)}")
    print(f"[orient] along palm    {np.round(along, 3)}")
    print(f"[orient] thumb dir     {np.round(thumb, 3)}")
    tilt = np.degrees(np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1, 1)))
    print(f"[orient] spawn-pose tilt {tilt:.1f} deg (not the pose that runs)")

    open_t = torch.as_tensor(open_a, device=u.device).unsqueeze(0)
    closed_t = torch.as_tensor(closed_a, device=u.device).unsqueeze(0)

    env.reset()
    tally: dict[str, int] = {}
    closest = []
    # Which of is_secured's four conditions actually blocks a catch. Counted
    # only over steps where the ball is genuinely resting on the hand, because
    # a condition that fails while the ball is still in the air says nothing.
    cond = {"contacts": 0, "lateral": 0, "above": 0, "slow": 0}
    resting = 0
    touched = 0
    steps_seen = 0
    peak_force = 0.0
    tips_hist = np.zeros(17, dtype=int)
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        ball_pos = u.ball.data.root_pos_w
        ball_vel = u.ball.data.root_lin_vel_w
        palm = u.hand.data.body_pos_w[:, u._palm_idx[0], :]

        t_arrive, valid = rw.time_to_plane(ball_pos, ball_vel, palm[:, 2])
        # A ball that is not coming holds the hand open; read as 0 it would mean
        # "arriving now" and the hand would clench at nothing.
        c = torch.where(valid,
                        torch.clamp((args_cli.lead_time - t_arrive) / args_cli.lead_time,
                                    0.0, 1.0),
                        torch.zeros_like(t_arrive)).unsqueeze(-1)
        action = (1.0 - c) * open_t + c * closed_t

        # Read the sensor BEFORE stepping. After the step, every environment
        # that terminated has already been reset, so the readings describe a
        # pristine spawn rather than the moment that ended the episode -- which
        # made a 100% excess_force run look like a 0% contact run.
        forces = torch.linalg.norm(u.contacts.data.net_forces_w, dim=-1)
        tips = (forces > 0.1).sum(dim=-1)
        ball_f = torch.linalg.norm(u.ball_contacts.data.net_forces_w, dim=-1).squeeze(-1)

        _, _, terminated, truncated, _ = env.step(action)
        closest.append(torch.linalg.norm(ball_pos - palm, dim=-1).min().item())

        if i == 0 and args_cli.debug_steps:
            print(f"[cfg] {u._task_cfg}")
            print(f"[cfg] max_episode_length={u.max_episode_length} "
                  f"step_dt={u.step_dt} episode_length_buf={u.episode_length_buf.tolist()}")
            r2 = rw.terminate(ball_pos, palm, ball_f, u.episode_length_buf,
                              u._held_steps, u._flights, u._task_cfg)
            print(f"[cfg] replicated terminate -> "
                  f"{torch.bincount(r2, minlength=7).tolist()}")
        if i < args_cli.debug_steps:
            bf = u.ball_contacts.data.net_forces_w
            hf = u.contacts.data.net_forces_w
            print(f"[dbg {i}] ball_force_w{tuple(bf.shape)} "
                  f"min={bf.min().item():.3g} max={bf.max().item():.3g} "
                  f"nan={int(torch.isnan(bf).sum())} | "
                  f"hand{tuple(hf.shape)} max={hf.max().item():.3g} "
                  f"nan={int(torch.isnan(hf).sum())}")
            print(f"        env saw max_force={u._dbg_force.max().item():.4g} "
                  f"(threshold {u._task_cfg.max_force}) "
                  f"shape={tuple(u._dbg_force.shape)}")
            print(f"        _reason{tuple(u._reason.shape)}="
                  f"{torch.bincount(u._reason.flatten(), minlength=7).tolist()} "
                  f"threshold={u.cfg.max_force} "
                  f"ballz-palmz={(ball_pos[:,2]-palm[:,2]).min().item():.3f}")
        lateral = torch.linalg.norm(ball_pos[:, :2] - palm[:, :2], dim=-1)
        speed = torch.linalg.norm(ball_vel, dim=-1)
        # "Resting on the hand": slow, above the palm, and close enough that it
        # is clearly not still falling past.
        # "Resting ON the hand": slow, close, and TOUCHED. Without the contact
        # term this also counts the ball sitting still at spawn before it
        # starts falling -- from rest it takes 6 steps to reach 0.25 m/s, and
        # those steps swamped the statistic.
        rest = ((speed < 0.25) & (ball_pos[:, 2] > palm[:, 2]) & (lateral < 0.10)
                & (u.episode_length_buf > 20))
        steps_seen += ball_f.numel()
        touched += int((ball_f > u.cfg.contact_threshold).sum())
        peak_force = max(peak_force, float(ball_f.max()))
        n = int(rest.sum())
        if n:
            resting += n
            cond["contacts"] += int((ball_f[rest] <= u.cfg.contact_threshold).sum())
            cond["lateral"] += int((lateral[rest] > u.cfg.catch_radius).sum())
            cond["above"] += int((ball_pos[rest, 2] <= palm[rest, 2]).sum())
            cond["slow"] += int((speed[rest] > u.cfg.secure_speed).sum())
            for k in range(17):
                tips_hist[k] += int((tips[rest] == k).sum())

        done = terminated | truncated
        if done.any():
            for code in u._reason[done].tolist():
                tally[REASONS[code]] = tally.get(REASONS[code], 0) + 1

    total = sum(tally.values()) or 1
    print(f"\n[scripted] {sum(tally.values())} episodes over {args_cli.steps} steps")
    for name in REASONS[1:]:
        n = tally.get(name, 0)
        print(f"  {name:>15s}  {n:5d}  {100.0*n/total:5.1f}%")
    # 3D distance to the palm ORIGIN -- not comparable to catch_radius, which is
    # lateral only. It says how far down into the hand the ball ever got.
    print(f"[scripted] closest ball-to-palm-origin distance (3D): {np.min(closest)*100:.1f} cm")

    print(f"\n[diag] ball touched the hand on {touched} of {steps_seen} env-steps "
          f"({100.0*touched/max(steps_seen,1):.1f}%), peak force {peak_force:.1f} N")
    print(f"[diag] {resting} env-steps with the ball resting on the hand")
    if resting:
        print("  is_secured conditions FAILING while the ball rests there:")
        for k, v in cond.items():
            print(f"    {k:>10s}  {v:7d}  {100.0*v/resting:5.1f}%")
        print("  hand bodies in contact while resting (includes self-contact):")
        for k in range(17):
            if tips_hist[k]:
                print(f"    {k:2d} bodies  {tips_hist[k]:7d}  {100.0*tips_hist[k]/resting:5.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
