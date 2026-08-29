"""The laptop-side scripts must import on the laptop.

``scripts/`` is deliberately not a package -- Isaac Lab's own scripts directory
isn't one either -- so nothing type-checks the module names inside it. Renaming
the harness package once already left ``smoke.py`` importing a module that no
longer existed, and the failure only showed up as a ModuleNotFoundError after
``make smoke`` had already been typed.

Only scripts that avoid ``isaaclab`` can be checked here; the rest import a
simulator this machine cannot install.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# The rest of scripts/ imports isaaclab and can only be loaded on the GPU box.
LAPTOP_SCRIPTS = ["smoke.py", "run_remote.py", "patch_container.py", "inspect_dataset.py"]


@pytest.mark.parametrize("name", LAPTOP_SCRIPTS)
def test_laptop_script_imports(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Safe to execute: every script guards its work behind __main__.
    spec.loader.exec_module(module)


def test_isaaclab_scripts_are_not_claimed_to_be_laptop_scripts():
    # A script that imports isaaclab must not be added to LAPTOP_SCRIPTS: the
    # test above would then fail on the laptop for the wrong reason.
    for name in LAPTOP_SCRIPTS:
        assert "import isaaclab" not in (SCRIPTS / name).read_text()
