"""Sphinx configuration for pygeofetch."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pygeofetch"
copyright = "2026, pygeofetch contributors"
author = "pygeofetch contributors"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("pygeofetch")
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# autodoc imports the real package, which pulls in heavy optional deps
# (rasterio, boto3, etc.). Mock anything not needed just to build docs
# so a missing system lib (e.g. GDAL) never breaks the RTD build.
autodoc_mock_imports = [
    "rasterio",
    "numpy",
    "scipy",
    "matplotlib",
    "boto3",
    "keyring",
    "cryptography",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static")) else []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
