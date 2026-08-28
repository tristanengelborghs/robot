"""Every hand preset must physically hold the cube, or the task is a drop."""

from __future__ import annotations

import pytest

from dexhand.hands import HANDS
from dexhand.scene import build_scene, settle_check


@pytest.mark.parametrize("name", sorted(HANDS))
def test_preset_holds_the_cube(name):
    """The Allegro preset shipped with the Shadow's cube_spawn — 30 cm down a
    forearm the Allegro does not have — and every episode was a one-step drop
    that no metric flagged. Settle the cube on a relaxed hand; it must stay."""
    spec = HANDS[name]
    if not spec.exists():
        pytest.skip(f"{name}: asset not fetched (make assets)")
    held, pos = settle_check(build_scene(spec))
    assert held, f"{name}: cube settled at {pos}, not on the palm"


def test_env_refuses_a_hand_that_drops_the_cube():
    import dataclasses

    from dexhand.env import ReorientEnv
    from dexhand.hands import TOY

    # the exact shape of the Allegro bug: a spawn point in empty space. (An
    # upside-down or vertical toy hand is not a usable negative — the cube
    # balances on the palm's back or on the slab's edge and the check is right
    # to call that held.)
    spawn_in_space = dataclasses.replace(TOY, cube_spawn=(0.3, 0.0, 0.27))
    with pytest.raises(RuntimeError, match="does not hold the cube"):
        ReorientEnv(spawn_in_space)
