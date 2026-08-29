"""Fix three things about the box's Isaac Lab that stop a headless run working.

Runs *inside* the container, as a plain script -- no Isaac Lab, no simulator, no
dependencies -- and is applied by ``make sync`` every time, because the
container's filesystem is not persistent. Nothing here is a lasting edit to
somebody else's code: it is a rented machine that gets rebuilt, and every patch
is reapplied from this file, which is in our repository and reviewable.

All three are in Isaac Lab 3.0.0 itself rather than in anything we wrote, and
none can be worked around from our side -- Isaac Lab's own scripts hit them:

1. **The Franka asset moved.** ``isaaclab_assets/robots/franka.py`` asks for
   ``.../Robots/FrankaEmika/panda_instanceable.usd``; NVIDIA reorganised the 6.0
   asset bucket and the file now lives under ``FrankaEmika/Legacy/``. The
   shipped path returns 404, so every Franka task dies in ``spawn_from_usd``
   before it can step once. Checked against the bucket on 2026-08-29: the old
   path 404s, the new one is the same asset.

2. **replay_demos.py cannot run headless.** It builds an ``Se3Keyboard`` for its
   pause/resume keys unconditionally, and that reaches for
   ``omni.appwindow.get_default_app_window()``, which does not exist without a
   window. Replaying a dataset has nothing to do with a keyboard, so the device
   is stubbed out rather than the replay given up on.

3. **``--validate_states`` raises on every dataset.** ``EpisodeData.get_state``
   returns ``states[index, None]``, keeping a leading singleton dimension, while
   ``compare_states`` has already indexed the runtime state down to a single
   environment. It then compares a length of 1 against a length of 7 and raises
   "State shape of root_pose for asset robot don't match" at the first
   comparison -- for Isaac Lab's own recorded datasets as much as for ours.

Every patch is idempotent: a second run finds it already applied. Each is also
pinned by a test that checks its search pattern still matches the copy of the
upstream source vendored in reference/isaaclab_source/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ISAACLAB = Path("/workspace/isaaclab")

FRANKA_CFG = ISAACLAB / "source/isaaclab_assets/isaaclab_assets/robots/franka.py"
STALE_USD = "/Robots/FrankaEmika/panda_instanceable.usd"
MOVED_USD = "/Robots/FrankaEmika/Legacy/panda_instanceable.usd"

REPLAY = ISAACLAB / "scripts/tools/replay_demos.py"
KEYBOARD_IMPORT = "from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg"
KEYBOARD_STUB = '''from isaaclab.devices import Se3KeyboardCfg  # patched: headless keyboard stub


class Se3Keyboard:  # patched: headless keyboard stub
    """No-op stand-in. The real one needs an app window; replay does not need it."""

    def __init__(self, *args, **kwargs):
        pass

    def add_callback(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
'''


VALIDATE_OLD = """                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):"""
VALIDATE_NEW = """                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                # patched: squeeze batch dim -- EpisodeData.get_state returns
                # states[index, None], which keeps a leading singleton
                # dimension, while the runtime state above has already been
                # indexed down to one environment. Comparing their lengths
                # therefore compares 1 against 7 and raises for every dataset.
                if len(dataset_asset_state) == 1 and len(runtime_asset_state) != 1:
                    dataset_asset_state = dataset_asset_state[0]
                if len(dataset_asset_state) != len(runtime_asset_state):"""


def patch(path: Path, old: str, new: str, *, marker: str, what: str) -> bool:
    """Replace ``old`` with ``new`` in ``path``, unless ``marker`` is already there."""
    if not path.exists():
        print(f"[patch] {what}: {path} does not exist -- skipped", file=sys.stderr)
        return False

    text = path.read_text()
    if marker in text:
        print(f"[patch] {what}: already applied")
        return False
    if old not in text:
        # Isaac Lab moved on and this patch is now aimed at nothing. Say so
        # loudly: silently skipping is how a fix rots into a superstition.
        print(f"[patch] {what}: pattern not found in {path} -- CHECK WHETHER IT IS STILL NEEDED", file=sys.stderr)
        return False

    path.write_text(text.replace(old, new))
    print(f"[patch] {what}: applied to {path}")
    return True


def main() -> int:
    patch(FRANKA_CFG, STALE_USD, MOVED_USD, marker=MOVED_USD, what="franka usd moved upstream")
    patch(REPLAY, KEYBOARD_IMPORT, KEYBOARD_STUB, marker="headless keyboard stub", what="replay keyboard headless")
    patch(REPLAY, VALIDATE_OLD, VALIDATE_NEW, marker="squeeze batch dim", what="replay state validation shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
