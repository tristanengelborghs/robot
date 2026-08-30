"""The catch task's phase, reward and termination logic, vectorised in torch.

This is a deliberate second implementation of `harness/catch.py`. The reference
there is NumPy, scalar, and tested; this one runs on thousands of environments
at once on the GPU and imports no isaaclab, so it can be tested on a laptop
too. `tests/test_parity.py` drives both with the same random states and asserts
they agree -- which is what makes the duplication safe rather than a second
place for the reward to be quietly wrong.

Keeping them separate rather than sharing code is the point. The reference is
written for clarity and is the thing a reader should trust; this one is written
for the accelerator, and any divergence between them is a test failure instead
of a training run that silently optimises something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

G = 9.81

# Phase and reason codes. These MUST match harness.catch.PHASE_CODES and
# harness.catch.REASON_CODES; test_parity.py asserts it.
SETTLE, READY, FLIGHT, SECURED = 0, 1, 2, 3
NONE, DROPPED, OUT_OF_BOUNDS, EXCESS_FORCE, CAUGHT, NEVER_LAUNCHED, TIMEOUT = range(7)


@dataclass
class CatchCfg:
    """Reward-relevant scalars. Field names mirror harness.catch.CatchConfig,
    and test_parity builds one from the other field by field, so a field added
    on one side and forgotten on the other is a test failure."""

    is_drop_stage: bool = True
    is_pop_stage: bool = False

    settle_steps: int = 20
    episode_len: int = 400
    spawn_height: float = 0.05

    catch_radius: float = 0.08
    secure_speed: float = 0.25
    min_contacts: int = 1
    hold_steps: int = 30

    drop_below: float = 0.10
    drop_lateral: float = 0.25
    max_force: float = 40.0
    launch_deadline: int = 150

    track_scale: float = 2.0
    secure_bonus: float = 50.0
    hold_scale: float = 1.0
    apex_scale: float = 10.0
    drop_penalty: float = 20.0
    force_penalty: float = 0.5
    action_rate_penalty: float = 0.01
    ctrl_penalty: float = 0.001

    # -- validation -------------------------------------------------------
    ctrl_dt: float = 1.0 / 240.0
    ball_radius: float = 0.03
    min_flight_steps: int = 20
    max_travel_fraction: float = 0.25

    def validate(self) -> None:
        """Refuse a configuration the task cannot be learned in.

        Mirrors harness.catch.CatchConfig.validate, and test_parity asserts the
        two agree. It exists in both places because only this one is reachable
        from the environment -- the first smoke run shipped a config the
        reference validator would have rejected, and nothing checked.
        """
        import math

        if self.is_drop_stage:
            flight = math.sqrt(2.0 * self.spawn_height / G)
            v = math.sqrt(2.0 * G * self.spawn_height)
        else:
            v = math.sqrt(2.0 * G * self.spawn_height)
            flight = 2.0 * v / G
        steps = flight / self.ctrl_dt
        if steps < self.min_flight_steps:
            raise ValueError(
                f"{steps:.1f} control steps of flight at {1 / self.ctrl_dt:.0f} Hz for a "
                f"{self.spawn_height * 100:.0f} cm drop; need >= {self.min_flight_steps}. "
                f"Raise the control rate (lower decimation) or raise spawn_height.")
        travel = v * self.ctrl_dt
        limit = self.max_travel_fraction * 2.0 * self.ball_radius
        if travel > limit:
            raise ValueError(
                f"ball travels {travel * 100:.1f} cm per control step, more than "
                f"{self.max_travel_fraction:.0%} of its {2 * self.ball_radius * 100:.0f} cm "
                f"diameter ({limit * 100:.1f} cm). Raise the control rate.")


def time_to_plane(pos: torch.Tensor, vel: torch.Tensor,
                  z_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Seconds until the ball crosses `z_plane` descending, and whether it does.

    The descending root is the larger one; the ascending root for a ball just
    thrown from the hand is ~0 seconds away and is never the crossing worth
    catching at. Invalid entries come back as 0, so callers must consult the
    mask -- a 0 read as a time means "arriving now".
    """
    disc = vel[:, 2] ** 2 + 2.0 * G * (pos[:, 2] - z_plane)
    valid = disc >= 0.0
    t = (vel[:, 2] + torch.sqrt(torch.clamp(disc, min=0.0))) / G
    valid = valid & (t > 0.0)
    return torch.where(valid, t, torch.zeros_like(t)), valid


def predict_intercept_xy(pos: torch.Tensor, vel: torch.Tensor,
                         z_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Where the ball crosses `z_plane` descending, and whether it ever does."""
    t, valid = time_to_plane(pos, vel, z_plane)
    return pos[:, :2] + vel[:, :2] * t.unsqueeze(-1), valid


def is_secured(ball_pos: torch.Tensor, ball_vel: torch.Tensor, contacts: torch.Tensor,
               palm_pos: torch.Tensor, cfg: CatchCfg) -> torch.Tensor:
    """Held, not merely touched: a ball grazing a fingertip on its way past
    reports contact too."""
    lateral = torch.linalg.norm(ball_pos[:, :2] - palm_pos[:, :2], dim=-1)
    speed = torch.linalg.norm(ball_vel, dim=-1)
    return ((contacts >= cfg.min_contacts)
            & (lateral <= cfg.catch_radius)
            & (ball_pos[:, 2] > palm_pos[:, 2])
            & (speed <= cfg.secure_speed))


def phase_of(steps: torch.Tensor, ball_pos: torch.Tensor, ball_vel: torch.Tensor,
             contacts: torch.Tensor, palm_pos: torch.Tensor,
             flights: torch.Tensor, cfg: CatchCfg) -> torch.Tensor:
    """Phase codes. Applied lowest priority first, so the order matches the
    reference's sequence of early returns."""
    phase = torch.full_like(steps, READY)
    airborne = (contacts == 0) & (ball_pos[:, 2] > palm_pos[:, 2])
    phase = torch.where(airborne, torch.full_like(phase, FLIGHT), phase)
    held = is_secured(ball_pos, ball_vel, contacts, palm_pos, cfg)
    phase = torch.where(held & (flights > 0), torch.full_like(phase, SECURED), phase)
    phase = torch.where(held & (flights == 0), torch.full_like(phase, READY), phase)
    phase = torch.where(steps < cfg.settle_steps, torch.full_like(phase, SETTLE), phase)
    return phase


def reward_terms(ball_pos: torch.Tensor, ball_vel: torch.Tensor, contacts: torch.Tensor,
                 max_force: torch.Tensor, palm_pos: torch.Tensor, action: torch.Tensor,
                 prev_action: torch.Tensor, phase: torch.Tensor, held_steps: torch.Tensor,
                 peak_height: torch.Tensor, cfg: CatchCfg) -> Dict[str, torch.Tensor]:
    """Named terms, so ablating or logging one costs nothing."""
    zeros = torch.zeros_like(ball_pos[:, 0])
    terms: Dict[str, torch.Tensor] = {}

    in_flight = phase == FLIGHT
    point_xy, valid = predict_intercept_xy(ball_pos, ball_vel, palm_pos[:, 2])
    err = torch.linalg.norm(point_xy - palm_pos[:, :2], dim=-1)
    terms["track"] = torch.where(in_flight & valid, cfg.track_scale / (err + 0.05), zeros)

    if cfg.is_pop_stage:
        gained = torch.clamp(peak_height - palm_pos[:, 2], min=0.0)
        apex = cfg.apex_scale * torch.clamp(gained / cfg.spawn_height, max=1.0)
        terms["apex"] = torch.where(in_flight, apex, zeros)

    secured = phase == SECURED
    terms["hold"] = torch.where(secured, torch.full_like(zeros, cfg.hold_scale), zeros)
    terms["secure"] = torch.where(secured & (held_steps == 1),
                                  torch.full_like(zeros, cfg.secure_bonus), zeros)

    over = max_force > cfg.max_force
    terms["force"] = torch.where(over, -cfg.force_penalty * (max_force - cfg.max_force), zeros)

    terms["action_rate"] = -cfg.action_rate_penalty * torch.sum((action - prev_action) ** 2, dim=-1)
    terms["effort"] = -cfg.ctrl_penalty * torch.sum(action ** 2, dim=-1)

    # A settling environment earns nothing at all, including penalties: it has
    # not been asked to do anything yet.
    settling = phase == SETTLE
    for k, v in terms.items():
        terms[k] = torch.where(settling, zeros, v)
    return terms


def total(terms: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(terms.values()), dim=0).sum(dim=0)


def terminate(ball_pos: torch.Tensor, palm_pos: torch.Tensor, max_force: torch.Tensor,
              steps: torch.Tensor, held_steps: torch.Tensor, flights: torch.Tensor,
              cfg: CatchCfg) -> torch.Tensor:
    """Reason codes, 0 for still running.

    Applied lowest priority last-wins-first: written in reverse so the earliest
    check in the reference wins, because a ball that is both dropped and out of
    time is dropped.
    """
    reason = torch.full_like(steps, NONE)
    lateral = torch.linalg.norm(ball_pos[:, :2] - palm_pos[:, :2], dim=-1)

    def apply(mask, code):
        return torch.where(mask & (reason == NONE), torch.full_like(reason, code), reason)

    reason = apply(ball_pos[:, 2] < palm_pos[:, 2] - cfg.drop_below, DROPPED)
    reason = apply(lateral > cfg.drop_lateral, OUT_OF_BOUNDS)
    reason = apply(max_force > cfg.max_force, EXCESS_FORCE)
    reason = apply((held_steps >= cfg.hold_steps) & (flights > 0), CAUGHT)
    if not cfg.is_drop_stage:
        reason = apply((flights == 0) & (steps >= cfg.launch_deadline), NEVER_LAUNCHED)
    reason = apply(steps >= cfg.episode_len, TIMEOUT)
    return reason
