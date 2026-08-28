"""Domain randomization: train on a distribution of worlds, not a point.

A policy trained in one perfectly-known simulator learns that simulator's
quirks as if they were physics. The standard cure (OpenAI's dexterity work
established it for exactly this task) is to randomize every parameter the real
world refuses to pin down — friction, mass, actuator strength, damping, gravity
direction — per episode, so the only strategies that survive training are the
ones that do not depend on knowing them.

Implementation detail that matters: randomization PERTURBS A PRISTINE SNAPSHOT
taken at construction, never the current model. Perturbing in place compounds —
after a thousand resets the friction has random-walked far outside the intended
range, training silently becomes non-stationary, and nothing errors. Snapshot
and restore makes every reset an independent draw.

Each field is scaled log-uniformly (uniform in log space) where the parameter
is a positive scale factor: a x0.7..x1.3 friction range should make x0.7 and
x1/0.7 equally likely, which uniform sampling does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dexhand.scene import Scene


@dataclass
class DRConfig:
    enabled: bool = True
    #: multiplicative ranges, applied log-uniformly: value in [x lo, x hi]
    friction: tuple = (0.7, 1.3)      # sliding friction, cube + all hand geoms
    cube_mass: tuple = (0.7, 1.3)
    actuator_gain: tuple = (0.85, 1.15)
    damping: tuple = (0.8, 1.25)
    #: additive: gravity tilted by up to this many radians from vertical
    gravity_tilt: float = 0.05
    #: additive noise on the student's observed joint positions (radians).
    #: The teacher's cube pose is deliberately left exact: it is the privileged
    #: signal, and the student never sees it at all.
    obs_noise_joint: float = 0.01
    #: initial hand pose jitter (radians)
    init_qpos_noise: float = 0.05
    #: actions applied with this many control steps of delay (0 = none, sampled per episode up to this)
    max_action_delay: int = 1


class DomainRandomizer:
    """Owns the pristine snapshot of one env's model and re-rolls it at reset."""

    def __init__(self, scene: Scene, cfg: DRConfig):
        self.cfg = cfg
        m = scene.model
        self._scene = scene
        self._pristine: Dict[str, np.ndarray] = {
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "dof_damping": m.dof_damping.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "gravity": m.opt.gravity.copy(),
        }
        #: per-episode observation-noise state, read by the env
        self.action_delay = 0

    def restore(self) -> None:
        m = self._scene.model
        m.geom_friction[:] = self._pristine["geom_friction"]
        m.body_mass[:] = self._pristine["body_mass"]
        m.body_inertia[:] = self._pristine["body_inertia"]
        m.dof_damping[:] = self._pristine["dof_damping"]
        m.actuator_gainprm[:] = self._pristine["actuator_gainprm"]
        m.actuator_biasprm[:] = self._pristine["actuator_biasprm"]
        m.opt.gravity[:] = self._pristine["gravity"]

    def randomize(self, rng: np.random.Generator) -> None:
        """Restore pristine values, then apply one fresh independent draw."""
        self.restore()
        if not self.cfg.enabled:
            self.action_delay = 0
            return
        m, cfg = self._scene.model, self.cfg

        m.geom_friction[:, 0] *= _logu(rng, *cfg.friction)
        # mass and inertia scale together (same material, same shape), and the
        # env calls mj_setConst afterwards — body_subtreemass and friends are
        # derived quantities that a bare write to body_mass leaves stale, so
        # the "heavier" cube would still accelerate like the nominal one
        scale = _logu(rng, *cfg.cube_mass)
        m.body_mass[self._scene.cube_body_id] *= scale
        m.body_inertia[self._scene.cube_body_id] *= scale
        m.dof_damping[:] *= _logu(rng, *cfg.damping, size=m.nv)

        # position actuators: gainprm[0] = kp and biasprm[1] = -kp must move
        # together or the actuator's equilibrium point shifts instead of its
        # strength — a subtly different randomization than intended
        gain = _logu(rng, *cfg.actuator_gain, size=m.nu)
        m.actuator_gainprm[:, 0] *= gain
        m.actuator_biasprm[:, 1] *= gain

        tilt = cfg.gravity_tilt
        if tilt > 0:
            ax, ay = rng.uniform(-tilt, tilt, 2)
            g = np.linalg.norm(self._pristine["gravity"])
            m.opt.gravity[:] = [g * np.sin(ax), g * np.sin(ay), -g * np.cos(ax) * np.cos(ay)]

        self.action_delay = int(rng.integers(0, cfg.max_action_delay + 1))


def _logu(rng: np.random.Generator, lo: float, hi: float, size=None) -> np.ndarray | float:
    """Log-uniform multiplicative factor(s) in [lo, hi]."""
    draw = rng.uniform(np.log(lo), np.log(hi), size)
    return np.exp(draw)
