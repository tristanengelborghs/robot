"""Fix three things in the box's Isaac Lab that stop a headless run working.

Runs *inside* the container, as a plain script -- it imports nothing but the
standard library, because the container has no simulator-free Python and this
must work before anything else does. ``make sync`` applies it every time, since
the container's filesystem does not survive a rebuild.

Nothing here is a lasting edit to somebody else's code. It is a rented machine,
every patch is reapplied from this file, and this file is in our repository
where it can be read and argued with.

All three problems are in Isaac Lab 3.0.0 itself rather than in anything we
wrote, and none can be worked around from an environment config, because Isaac
Lab's own scripts hit them too:

1. **The Franka asset moved.** ``isaaclab_assets/robots/franka.py`` asks for
   ``.../Robots/FrankaEmika/panda_instanceable.usd``; NVIDIA reorganised the 6.0
   asset bucket and the file now lives under ``FrankaEmika/Legacy/``. The
   shipped path returns 404, so every Franka task dies in ``spawn_from_usd``
   before it can step once. Checked against the bucket on 2026-08-29: the old
   path 404s, the new one is the same asset. The table in the same scene still
   resolves, which is what makes this look like a network fault at first.

2. **replay_demos.py cannot start headless.** It builds an ``Se3Keyboard`` for
   its pause/resume keys unconditionally, and that reaches for
   ``omni.appwindow.get_default_app_window()``, which does not exist without a
   window. Replaying a recorded dataset does not need a keyboard, so the device
   becomes optional rather than mandatory -- with a window, as in a
   ``--livestream`` run, the real device is still built and the keys still work.

3. **``--validate_states`` raises on every dataset.** ``EpisodeData.get_state``
   returns ``states[index, None]``, keeping a leading singleton dimension, while
   ``compare_states`` has already indexed the runtime state down to a single
   environment. It compares a length of 1 against a length of 7 and raises
   "State shape of root_pose for asset robot don't match" at the first
   comparison -- for Isaac Lab's own recorded datasets as much as for ours.

Every patch is idempotent: a second run finds it already applied. Each is also
pinned by a test in tests/test_patches.py that checks its search text still
matches the copy of the upstream source vendored under reference/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ISAACLAB = Path("/workspace/isaaclab")

FRANKA_CFG = ISAACLAB / "source/isaaclab_assets/isaaclab_assets/robots/franka.py"
REPLAY = ISAACLAB / "scripts/tools/replay_demos.py"


@dataclass(frozen=True)
class Patch:
    """One literal search-and-replace against a file of Isaac Lab's.

    Attributes:
        name: What this patch is for, used in the log line.
        path: The file to edit.
        old: The exact text to replace. If it is missing *and* the patch is not
            already applied, Isaac Lab has changed and we want to hear about it.
        new: What to put there instead.
        marker: Text that is present once the patch has been applied. This is
            what makes reapplying it a no-op.
    """

    name: str
    path: Path
    old: str
    new: str
    marker: str


# ---------------------------------------------------------------------------
# 1. The Franka USD that moved upstream.
# ---------------------------------------------------------------------------

STALE_USD = "/Robots/FrankaEmika/panda_instanceable.usd"
MOVED_USD = "/Robots/FrankaEmika/Legacy/panda_instanceable.usd"

FRANKA_USD_PATCH = Patch(
    name="franka usd moved upstream",
    path=FRANKA_CFG,
    old=STALE_USD,
    new=MOVED_USD,
    marker=MOVED_USD,
)


# ---------------------------------------------------------------------------
# 2. The keyboard replay_demos.py insists on, even with no window to read.
# ---------------------------------------------------------------------------

KEYBOARD_OLD = """    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')"""

KEYBOARD_NEW = '''    # patched: the keyboard is optional. Se3Keyboard reaches for
    # omni.appwindow.get_default_app_window(), which does not exist on a
    # headless run, and replaying a recorded dataset does not need a keyboard.
    # With a window -- a --livestream run -- the real device is still built and
    # the pause and resume keys still work.
    try:
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        teleop_interface.add_callback("N", play_cb)
        teleop_interface.add_callback("B", pause_cb)
        print('Press "B" to pause and "N" to resume the replayed actions.')
    except Exception as keyboard_error:
        print(f"[patched] no keyboard available ({keyboard_error}); replaying without pause and resume keys")

        class _NoKeyboard:
            """Stand-in carrying the only method the replay loop calls."""

            def reset(self):
                pass

        teleop_interface = _NoKeyboard()'''

KEYBOARD_PATCH = Patch(
    name="replay keyboard optional",
    path=REPLAY,
    old=KEYBOARD_OLD,
    new=KEYBOARD_NEW,
    marker="patched: the keyboard is optional",
)


# ---------------------------------------------------------------------------
# 3. The state comparison that compares a batch of one against a vector of 7.
# ---------------------------------------------------------------------------

VALIDATE_OLD = """                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):"""

VALIDATE_NEW = """                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                # patched: squeeze batch dim. EpisodeData.get_state returns
                # states[index, None], which keeps a leading singleton
                # dimension, while the runtime state above has already been
                # indexed down to one environment. Comparing their lengths
                # therefore compares 1 against 7 and raises for every dataset.
                if len(dataset_asset_state) == 1 and len(runtime_asset_state) != 1:
                    dataset_asset_state = dataset_asset_state[0]
                if len(dataset_asset_state) != len(runtime_asset_state):"""

VALIDATE_PATCH = Patch(
    name="replay state validation shapes",
    path=REPLAY,
    old=VALIDATE_OLD,
    new=VALIDATE_NEW,
    marker="patched: squeeze batch dim",
)


PATCHES = (FRANKA_USD_PATCH, KEYBOARD_PATCH, VALIDATE_PATCH)


def apply(patch: Patch) -> bool:
    """Apply one patch. Returns True if this call changed the file.

    A patch that finds neither its search text nor its marker is reported
    loudly and skipped. Skipping quietly is how a fix rots into a superstition:
    the run fails minutes later on a rented GPU, and nobody connects the error
    to a search string that stopped matching.
    """
    if not patch.path.exists():
        print(f"[patch] {patch.name}: {patch.path} does not exist -- skipped", file=sys.stderr)
        return False

    text = patch.path.read_text()

    if patch.marker in text:
        print(f"[patch] {patch.name}: already applied")
        return False

    if patch.old not in text:
        print(
            f"[patch] {patch.name}: search text not found in {patch.path} -- CHECK WHETHER THIS PATCH IS STILL NEEDED",
            file=sys.stderr,
        )
        return False

    patch.path.write_text(text.replace(patch.old, patch.new))
    print(f"[patch] {patch.name}: applied to {patch.path}")
    return True


def main() -> int:
    for patch in PATCHES:
        apply(patch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
