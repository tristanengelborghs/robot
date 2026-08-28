"""Compose the task scene: hand + cube + goal ghost, one compiled MjModel.

Composition is programmatic (MjSpec attachment) rather than a hand-written XML
per hand, because the hand files are vendored third-party assets: editing them
to add a cube means re-editing on every Menagerie update, and XML <include>
cannot reposition an included body. MjSpec attaches the hand's root body under
a frame we control, so the palm-up orientation is our decision, applied at
build time, and the vendored files stay pristine.

Two decisions here are physics-relevant, and both are the kind that fail
silently:

Solver options are copied from the HandSpec, not inherited. MjSpec attachment
keeps the PARENT's <option> on conflict and only warns. The Shadow model was
tuned with elliptic friction cones and impratio 10; compiled without them the
grasp contacts are visibly softer and a policy trained on that physics is a
policy for a different hand.

The goal cube is a mocap body with contact disabled (contype=0 conaffinity=0).
It shows the target orientation to a viewer and to a future vision student,
floating beside the hand. Giving it contact would let the solver "help" the
task by parking the real cube against the ghost — a reward exploit that looks
like success in every scalar metric.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from dexhand.hands import HandSpec

#: half-extent of the cube, metres (OpenAI/DeXtreme use a 5.5 cm cube; ours
#: matches at 0.055/2 rounded — small enough to reorient, big enough to grip)
CUBE_HALF = 0.03
CUBE_MASS = 0.06
GOAL_OFFSET = np.array([0.0, 0.25, 0.1])  # ghost floats beside the hand


@dataclass
class Scene:
    model: mujoco.MjModel
    hand: HandSpec
    #: qpos/qvel addresses of the cube free joint
    cube_qpos_adr: int
    cube_qvel_adr: int
    cube_body_id: int
    cube_geom_id: int
    palm_body_id: int
    goal_mocap_id: int
    fingertip_body_ids: Tuple[int, ...]
    #: actuator ctrlrange (nu, 2) — actions in [-1, 1] lerp into this
    ctrl_range: np.ndarray
    #: ids of every geom belonging to each fingertip body, for tactile
    fingertip_geom_ids: Tuple[Tuple[int, ...], ...]


def build_scene(hand: HandSpec) -> Scene:
    """Hand file + cube + goal ghost -> compiled model and the ids the env needs."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.cone = (
        mujoco.mjtCone.mjCONE_ELLIPTIC if hand.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    spec.option.impratio = hand.impratio

    child = mujoco.MjSpec.from_file(str(hand.xml_path))
    frame = spec.worldbody.add_frame(pos=list(hand.attach_pos), quat=list(hand.attach_quat))
    with warnings.catch_warnings():
        # the option-conflict warning is the exact thing the lines above resolve
        warnings.simplefilter("ignore")
        frame.attach_body(child.worldbody.first_body(), "", "")

    cube = spec.worldbody.add_body(name="cube", pos=list(hand.cube_spawn))
    cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="cube_free")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        mass=CUBE_MASS,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.85, 0.3, 0.15, 1.0],
    )

    goal = spec.worldbody.add_body(
        name="goal", mocap=True, pos=list(np.asarray(hand.cube_spawn) + GOAL_OFFSET)
    )
    goal.add_geom(
        name="goal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        rgba=[0.2, 0.6, 0.9, 0.4],
        contype=0,
        conaffinity=0,
        mass=0.001,  # mocap bodies take no dynamics, but the compiler wants inertia
    )

    spec.worldbody.add_light(pos=[0.3, -0.3, 1.2], dir=[-0.2, 0.2, -1.0])
    spec.worldbody.add_camera(
        name="task_view", pos=[0.7, -0.5, 0.75], xyaxes=[0.6, 0.8, 0.0, -0.35, 0.26, 0.9]
    )

    model = spec.compile()

    def bid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"body '{name}' not found in composed scene for hand '{hand.name}'")
        return i

    tip_ids = tuple(bid(b) for b in hand.fingertip_bodies)
    tip_geoms = tuple(
        tuple(g for g in range(model.ngeom) if model.geom_bodyid[g] == t) for t in tip_ids
    )
    for b, geoms in zip(hand.fingertip_bodies, tip_geoms):
        if not geoms:
            raise ValueError(f"fingertip body '{b}' has no geoms — tactile would be silently zero")

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    goal_body = bid("goal")

    return Scene(
        model=model,
        hand=hand,
        cube_qpos_adr=int(model.jnt_qposadr[joint_id]),
        cube_qvel_adr=int(model.jnt_dofadr[joint_id]),
        cube_body_id=bid("cube"),
        cube_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        palm_body_id=bid(hand.palm_body),
        goal_mocap_id=int(model.body_mocapid[goal_body]),
        fingertip_body_ids=tip_ids,
        ctrl_range=model.actuator_ctrlrange.copy(),
        fingertip_geom_ids=tip_geoms,
    )


def settle_check(scene: Scene, seconds: float = 2.0) -> Tuple[bool, np.ndarray]:
    """Drop the cube from its spawn point onto a relaxed hand; does it stay?

    This is the palm-up test, and it is a physics experiment rather than a
    geometry check on purpose: no hand file says where its palm surface is, and
    a wrong attach_quat produces a scene that compiles, steps, and looks fine in
    every scalar until you notice that every episode ends with a 1-step drop.
    The Allegro preset shipped exactly that way before this check existed.

    Returns (held, cube_position). Held means the cube is still within a palm's
    reach of the palm body after `seconds` of free physics with actuators at
    their mid-range.
    """
    m = scene.model
    d = mujoco.MjData(m)
    d.ctrl[:] = scene.ctrl_range.mean(axis=1)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    a = scene.cube_qpos_adr
    cube = d.qpos[a:a + 3].copy()
    palm = d.xpos[scene.palm_body_id]
    held = bool(cube[2] > palm[2] - 0.03 and np.linalg.norm(cube[:2] - palm[:2]) < 0.12)
    return held, cube
