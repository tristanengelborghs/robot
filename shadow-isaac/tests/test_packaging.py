"""The extension's metadata must load in a bare build environment.

pip builds with isolation: the backend runs in a fresh environment containing
setuptools and nothing else. `setup.py` therefore may not import anything
third-party -- Isaac Lab's generated version imports `toml` and dies with
ModuleNotFoundError before setup() is reached. This test reads the same file
the same way and checks the fields setup.py depends on are all present.
"""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = ROOT / "source/catching/setup.py"
EXT_TOML = ROOT / "source/catching/config/extension.toml"


def test_extension_metadata_parses_with_the_stdlib():
    with open(EXT_TOML, "rb") as fh:
        data = tomllib.load(fh)
    pkg = data["package"]
    for field in ("version", "description", "repository", "keywords",
                  "author", "maintainer"):
        assert pkg[field], f"extension.toml is missing {field}, which setup.py reads"


def test_setup_py_imports_nothing_third_party():
    """The whole point: a build dependency here is a failure at install time."""
    source = SETUP.read_text()
    assert "import toml\n" not in source, "third-party toml does not survive build isolation"
    assert "import tomllib" in source


def test_the_build_backend_is_declared():
    with open(ROOT / "source/catching/pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["build-system"]["requires"] == ["setuptools>=61"]


def test_the_declared_module_matches_the_package():
    with open(EXT_TOML, "rb") as fh:
        data = tomllib.load(fh)
    assert data["python"]["module"][0]["name"] == "catching"
    assert (ROOT / "source/catching/catching/__init__.py").exists()
