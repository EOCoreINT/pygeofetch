"""Sphinx configuration for pygeofetch."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pygeofetch"
copyright = "2026, Samuel Appiah Kubi"
author = "Samuel Appiah Kubi"

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
    "sphinx_copybutton",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# autodoc imports the real package, which pulls in heavy optional deps
# (rasterio, boto3, numpy, etc.). Mock anything not required just to
# build docs, so a missing system lib (e.g. GDAL) never breaks the
# Read the Docs build.
autodoc_mock_imports = [
    "rasterio",
    "numpy",
    "scipy",
    "matplotlib",
    "boto3",
    "keyring",
    "cryptography",
    "leafmap",
    "rioxarray",
    "localtileserver",
    "pyvista",
    "snaphu",
    "geopandas",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}
html_static_path = (
    ["_static"]
    if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static"))
    else []
)
html_title = "pygeofetch"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
