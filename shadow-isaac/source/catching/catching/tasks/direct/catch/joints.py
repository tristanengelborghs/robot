"""Which of the hand's joints an action can actually command.

The Shadow Hand has 24 joints and 20 actuators. The gap is not a rounding
error, it is the tendons: on each of the four fingers the distal joint (FFJ0,
MFJ0, RFJ0, LFJ0) is driven through a tendon shared with the joint above it,
not by an actuator of its own. That is the real underactuation of the hardware,
and Isaac Lab's asset models it -- those four joints come back from the
simulator with stiffness 0, damping 0 and an effort limit of 0, alongside four
fixed tendons.

The thumb is the exception: THJ0 is independently actuated (effort 0.810), as
it is on the real hand.

None of which is visible from the joint *ordering*. The four coupled joints sit
at indices 17, 18, 19 and 22 of 24, so the obvious `range(20)` commands three
joints that cannot move and silently drops LFJ1 and THJ1, which can. The
symptom is a little finger and a thumb that never respond and a policy that
cannot close a grip -- and nothing anywhere raises. Hence this module, and
hence `tests/test_joints.py`, which pins the selection against the joint names
actually read off the asset on the box.

No isaaclab import, so it is testable on a laptop.
"""

from __future__ import annotations

from typing import Sequence

#: Distal finger joints driven by a fixed tendon rather than an actuator.
#: Matched on the suffix so the `robot0_` prefix is not load-bearing.
COUPLED_SUFFIXES = ("FFJ0", "MFJ0", "RFJ0", "LFJ0")


def is_coupled(name: str) -> bool:
    return any(name.endswith(s) for s in COUPLED_SUFFIXES)


def actuated_indices(joint_names: Sequence[str], expected: int = 20) -> list[int]:
    """Indices of the joints an action commands, in the asset's own order.

    Raises:
        RuntimeError: if the count is not ``expected``. A silent miscount here
            is a hand that never fully closes, which looks like a training
            problem rather than an indexing one.
    """
    idx = [i for i, n in enumerate(joint_names) if not is_coupled(n)]
    if len(idx) != expected:
        missing = [n for n in joint_names if is_coupled(n)]
        raise RuntimeError(
            f"{len(idx)} actuated joints among {len(joint_names)}, expected {expected}. "
            f"Treated as tendon-coupled: {missing}. Run `make bodies` and check the "
            f"effort limits -- a coupled joint reports stiffness, damping and effort 0."
        )
    return idx
