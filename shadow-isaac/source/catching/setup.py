"""Installation script for the 'catching' python package.

Metadata is read from config/extension.toml with `tomllib`, which is in the
standard library from Python 3.11. Isaac Lab's template generator uses the
third-party `toml` package here, and that does not survive pip's build
isolation: the build backend runs in a fresh environment holding setuptools and
nothing else, so `import toml` raises ModuleNotFoundError before setup() is
ever reached. A stdlib parser has no such problem and needs no build
dependency at all.
"""

import os
import tomllib

from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as fh:
    EXTENSION_TOML_DATA = tomllib.load(fh)

PKG = EXTENSION_TOML_DATA["package"]

setup(
    name="catching",
    packages=["catching"],
    author=PKG["author"],
    maintainer=PKG["maintainer"],
    url=PKG["repository"],
    version=PKG["version"],
    description=PKG["description"],
    keywords=PKG["keywords"],
    install_requires=["psutil"],
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.11"],
    zip_safe=False,
)
