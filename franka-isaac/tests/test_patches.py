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

import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "reference/isaaclab_source"


def load_patch_module():
    """Import scripts/patch_container.py by path.

    It has to be registered in ``sys.modules`` before it is executed: it defines
    a dataclass, and ``@dataclass`` resolves the class's annotations through
    ``sys.modules[cls.__module__]``, which does not exist yet for a module
    loaded straight from a path.
    """
    path = ROOT / "scripts/patch_container.py"
    spec = importlib.util.spec_from_file_location("patch_container", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def upstream_path_for(patch) -> str:
    """Where this patch's target lives inside the vendored Isaac Lab copy."""
    return str(patch.path).removeprefix("/workspace/isaaclab/")


@pytest.mark.parametrize("index", range(3), ids=lambda i: f"patch{i}")
def test_every_patch_still_matches_its_target(patcher, index):
    patch = patcher.PATCHES[index]
    assert_patchable(
        vendored(upstream_path_for(patch)),
        pattern=patch.old,
        marker=patch.marker,
        what=patch.name,
    )


def test_all_patches_are_covered_by_the_parametrised_test(patcher):
    # The parametrisation above is a fixed range; a fourth patch must not slip
    # in unchecked.
    assert len(patcher.PATCHES) == 3


def test_patches_have_distinct_markers(patcher):
    # Two patches sharing a marker would make the second one a permanent no-op.
    markers = [patch.marker for patch in patcher.PATCHES]
    assert len(set(markers)) == len(markers)


def test_a_file_matching_neither_is_caught():
    with pytest.raises(AssertionError, match="changed underneath this patch"):
        assert_patchable("unrelated source", pattern="gone", marker="also gone", what="t")


def test_the_patched_files_still_parse(patcher):
    """Every file, with all of its patches applied, is still valid Python.

    This is the check that matters, and it cannot be done on a replacement in
    isolation: the fragments are indented pieces of a function body, and one of
    them deliberately ends on a dangling ``if`` whose body follows in the
    original file. Only the finished file means anything.
    """
    by_file: dict[str, list] = {}
    for patch in patcher.PATCHES:
        by_file.setdefault(upstream_path_for(patch), []).append(patch)

    for relative_path, patches in by_file.items():
        source = vendored(relative_path)
        for patch in patches:
            if patch.marker not in source:
                source = source.replace(patch.old, patch.new)
        ast.parse(source, filename=relative_path)


def test_patches_are_idempotent(tmp_path, patcher):
    source = vendored("scripts/tools/replay_demos.py")
    if patcher.VALIDATE_PATCH.marker in source:
        pytest.skip("vendored copy was taken from an already-patched container")

    target = tmp_path / "replay_demos.py"
    target.write_text(source)
    patch = dataclasses.replace(patcher.VALIDATE_PATCH, path=target)

    assert patcher.apply(patch) is True
    assert patcher.apply(patch) is False, "a second sync must not apply the patch twice"
    assert target.read_text().count(patch.marker) == 1


def test_a_patch_that_no_longer_applies_is_reported_not_silently_skipped(tmp_path, patcher, capsys):
    target = tmp_path / "moved_on.py"
    target.write_text("nothing here resembles the upstream source\n")
    stale = patcher.Patch(
        name="stale patch",
        path=target,
        old="search text that is gone",
        new="replacement",
        marker="never present",
    )

    assert patcher.apply(stale) is False
    assert "CHECK WHETHER THIS PATCH IS STILL NEEDED" in capsys.readouterr().err
