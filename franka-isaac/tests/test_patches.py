"""The container patches must still match the code they are aimed at.

Every patch in scripts/patch_container.py works by literal string replacement in
Isaac Lab's source. A patch whose pattern no longer matches is worse than no
patch: it silently does nothing, and the failure surfaces minutes later on a
rented GPU as an error nobody connects to a stale search string.

These tests check each pattern against reference/isaaclab_source/, the vendored
copy of the exact Isaac Lab 3.0.0 files taken from the container. When Isaac Lab
is upgraded, `make teleop-source` refreshes that copy and these tests are what
say whether the patches survived -- and, just as usefully, whether they are
still needed at all.

One wrinkle decides the shape of the assertions below. The container's Isaac Lab
is not a git checkout, so `make teleop-source` copies whatever is on the box --
which, after any `make sync`, is already patched. A vendored file may therefore
be in either state, and what each test can honestly require is that it is in
*one* of them: the search pattern is still there, or the patch is already
applied. A file in neither state is the case that matters, and the one these
tests exist to catch: Isaac Lab has changed underneath the patch.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "reference/isaaclab_source"


def load_patch_module():
    path = ROOT / "scripts/patch_container.py"
    spec = importlib.util.spec_from_file_location("patch_container", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patcher():
    return load_patch_module()


def vendored(path_in_isaaclab: str) -> str:
    path = VENDORED / path_in_isaaclab
    assert path.exists(), f"{path} is not vendored; run `make teleop-source`"
    return path.read_text()


def assert_patchable(source: str, *, pattern: str, marker: str, what: str) -> None:
    """The vendored file must be patchable, or already patched."""
    assert pattern in source or marker in source, (
        f"{what}: neither the search pattern nor the applied marker is in the vendored source. "
        "Isaac Lab has changed underneath this patch -- re-read it before the next GPU run."
    )


def test_franka_asset_pattern_still_matches(patcher):
    assert_patchable(
        vendored("source/isaaclab_assets/isaaclab_assets/robots/franka.py"),
        pattern=patcher.STALE_USD,
        marker=patcher.MOVED_USD,
        what="franka usd moved upstream",
    )


def test_keyboard_pattern_still_matches(patcher):
    assert_patchable(
        vendored("scripts/tools/replay_demos.py"),
        pattern=patcher.KEYBOARD_IMPORT,
        marker="headless keyboard stub",
        what="replay keyboard headless",
    )


def test_state_validation_pattern_still_matches(patcher):
    assert_patchable(
        vendored("scripts/tools/replay_demos.py"),
        pattern=patcher.VALIDATE_OLD,
        marker="squeeze batch dim",
        what="replay state validation shapes",
    )


def test_a_file_matching_neither_is_caught():
    with pytest.raises(AssertionError, match="changed underneath this patch"):
        assert_patchable("unrelated source", pattern="gone", marker="also gone", what="t")


def test_the_stub_the_patch_injects_is_valid_python(patcher):
    # It is spliced into a module at import time; a syntax error here would
    # break replay_demos.py rather than fix it.
    compile(patcher.KEYBOARD_STUB, "<stub>", "exec")


def test_patches_are_idempotent(tmp_path, patcher):
    source = vendored("scripts/tools/replay_demos.py")
    if "squeeze batch dim" in source:
        pytest.skip("vendored copy was taken from an already-patched container")
    target = tmp_path / "replay_demos.py"
    target.write_text(source)

    first = patcher.patch(target, patcher.VALIDATE_OLD, patcher.VALIDATE_NEW, marker="squeeze batch dim", what="t")
    second = patcher.patch(target, patcher.VALIDATE_OLD, patcher.VALIDATE_NEW, marker="squeeze batch dim", what="t")

    assert first is True
    assert second is False, "a second sync must not apply the patch twice"
    assert target.read_text().count("squeeze batch dim") == 1


def test_a_patch_that_no_longer_applies_is_reported_not_silently_skipped(tmp_path, patcher, capsys):
    target = tmp_path / "moved_on.py"
    target.write_text("nothing here resembles the upstream source\n")

    applied = patcher.patch(target, "pattern that is gone", "replacement", marker="never", what="stale patch")

    assert applied is False
    assert "CHECK WHETHER IT IS STILL NEEDED" in capsys.readouterr().err
