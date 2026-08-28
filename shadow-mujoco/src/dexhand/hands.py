"""Hand descriptions: which MJCF to load and which parts of it the task touches.

The environment never hardcodes a hand. Everything it needs to know — the model
file, the actuators it may command, the fingertip bodies that count as tactile
sensors, where the palm is — lives in a HandSpec, so swapping the Shadow Hand
for the Allegro (or for the three-finger toy hand the tests use) is a config
change, not an edit to the environment.

Two of these fields are less obvious than they look:

fingertip_bodies, not fingertip geoms. The Menagerie hands leave their collision
geoms unnamed, so tactile extraction keys on bodies and aggregates every contact
whose geom belongs to that body. Naming geoms ourselves would mean patching
vendored files, which is a maintenance treadmill.

attach_quat exists because a hand model bakes its own base orientation into its
root body (the Shadow model's forearm carries quat="0 1 0 1") and an in-hand
task needs the palm facing up so gravity holds the object against it. Both
Menagerie hands happen to leave the palm up under an identity attachment; what
differs is where the palm centre is, hence the per-hand cube_spawn. Neither
was read off the XML (nothing in it says "palm"): `scene.settle_check` drops
the cube and sees whether it stays, the env runs that check at construction,
and tests/test_hands.py runs it for every preset whose asset is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

MENAGERIE = Path(__file__).resolve().parent.parent.parent / "third_party" / "mujoco_menagerie"
TEST_ASSETS = Path(__file__).resolve().parent.parent.parent / "tests" / "assets"


@dataclass(frozen=True)
class HandSpec:
    """Everything the task needs to know about a hand, and nothing else."""

    name: str
    xml_path: Path
    fingertip_bodies: Tuple[str, ...]
    palm_body: str
    #: worldbody position of the hand's root body
    attach_pos: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    #: wxyz orientation of the attachment frame; palm must end up facing +z
    attach_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: cube spawn point, chosen above the palm centre (world frame)
    cube_spawn: Tuple[float, float, float] = (0.30, -0.01, 0.36)
    #: solver settings the model was tuned with. MjSpec attachment does NOT
    #: inherit <option> from the child file — the parent's defaults silently win,
    #: and the Shadow model without elliptic cones + impratio 10 has visibly
    #: mushier grasp contacts. They must be re-applied to the composed scene.
    cone: str = "elliptic"
    impratio: float = 10.0

    def exists(self) -> bool:
        return self.xml_path.is_file()


SHADOW = HandSpec(
    name="shadow",
    xml_path=MENAGERIE / "shadow_hand" / "right_hand.xml",
    # order matters: tactile obs dimensions follow it (FF, MF, RF, LF, thumb)
    fingertip_bodies=("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"),
    palm_body="rh_palm",
)

ALLEGRO = HandSpec(
    name="allegro",
    xml_path=MENAGERIE / "wonik_allegro" / "right_hand.xml",
    fingertip_bodies=("ff_tip", "mf_tip", "rf_tip", "th_tip"),
    palm_body="palm",
    # the Allegro's root IS the palm, so the centre sits at the attach point;
    # the Shadow's spawn (30 cm down the forearm) put the cube in empty space
    # and every Allegro episode was a one-step drop until settle_check existed
    cube_spawn=(0.0, 0.0, 0.36),
)

TOY = HandSpec(
    name="toy",
    xml_path=TEST_ASSETS / "toy_hand.xml",
    fingertip_bodies=("finger_a_tip", "finger_b_tip", "finger_c_tip"),
    palm_body="palm",
    attach_pos=(0.0, 0.0, 0.2),
    cube_spawn=(0.0, 0.0, 0.27),
)

HANDS = {h.name: h for h in (SHADOW, ALLEGRO, TOY)}


def get_hand(name: str) -> HandSpec:
    if name not in HANDS:
        raise KeyError(f"unknown hand '{name}'; available: {sorted(HANDS)}")
    spec = HANDS[name]
    if not spec.exists():
        raise FileNotFoundError(
            f"hand model not found: {spec.xml_path}\n"
            "Menagerie assets are a separate download (the repo stays clone-and-test):\n"
            "  make assets\n"
            "The 'toy' hand ships with the tests and needs no download."
        )
    return spec
