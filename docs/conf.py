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
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
    "sphinx_reredirects",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# Auto-generate slugged anchors for headings up to h4, so in-page and
# cross-page links like `page.md#some-heading` resolve.
myst_heading_anchors = 4

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

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/EOCoreINT/pygeofetch",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "light_css_variables": {"color-brand-primary": "#0b6e4f", "color-brand-content": "#0b6e4f"},
    "dark_css_variables": {"color-brand-primary": "#4cc38a", "color-brand-content": "#4cc38a"},
}
html_static_path = (
    ["_static"]
    if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static"))
    else []
)
html_css_files = ["custom.css"] if html_static_path else []

html_title = "pygeofetch"


# Pages that moved when the documentation was reorganised. Old URLs keep
# working: each old document name redirects to its new location.
redirects = {
    "core-features/authentication": "../data-access/authentication.html",
    "core-features/search": "../data-access/search.html",
    "core-features/download": "../data-access/download.html",
    "core-features/providers": "../data-access/providers.html",
    "core-features/optical-validation": "../data-access/optical-validation.html",
    "core-features/data-organizer": "../data-access/data-organizer.html",
    "core-features/geocoding": "../data-access/geocoding.html",
    "processing/spectral-indices": "../processing/optical/spectral-indices.html",
    "processing/preprocessing": "../processing/optical/preprocessing.html",
    "processing/classification": "../processing/optical/classification.html",
    "processing/terrain": "../processing/optical/terrain.html",
    "processing/landsat": "../processing/optical/landsat.html",
    "processing/timeseries": "../processing/optical/timeseries.html",
    "processing/postprocessing": "../processing/optical/postprocessing.html",
    "processing/sar": "../processing/sar-insar/sar.html",
    "processing/insar": "../processing/sar-insar/insar.html",
    "processing/insar-mexico-city-tutorial": "../processing/sar-insar/insar-mexico-city-tutorial.html",
    "processing/optical-offset-tracking": "../processing/sar-insar/optical-offset-tracking.html",
    "processing/multi-sensor-pipelines": "../processing/multisensor.html",
    "processing/upgrades": "../processing/sar-insar/advanced-modules.html",
    "reference/pipelines": "../workflows/pipelines.html",
    "reference/roadmap": "../project/roadmap.html",
    "contributing": "project/contributing.html",
}
