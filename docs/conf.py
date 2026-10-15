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
    release = "2.6.3"
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
    "announcement": "x",
    }
html_static_path = (
    ["_static"]
    if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static"))
    else []
)
html_css_files = ["custom.css"] if html_static_path else []
html_js_files = ["gf.js"] if html_static_path else []

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


# ---- Internationalisation -------------------------------------------------
# Translations live in locale/<lang>/LC_MESSAGES/*.po. Untranslated text falls
# back to English, so every language build is complete. Build all languages
# with tools/build_all.sh.
import os as _os
language = _os.environ.get("DOCS_LANG", "en")
locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = False
LANGUAGES = [
    ("en", "English"), ("fr", "Français"), ("es", "Español"), ("pt", "Português"),
    ("de", "Deutsch"), ("zh_CN", "简体中文"), ("ar", "العربية"),
]
html_context = globals().get("html_context", {})
html_context.update({"languages": LANGUAGES, "current_language": language})

# --- UI string translations (navbar, breadcrumb) read from locale/<lang>/LC_MESSAGES/chrome.po
def _load_chrome(lang):
    try:
        import polib
        po = polib.pofile(_os.path.join(_os.path.dirname(__file__), "locale", lang, "LC_MESSAGES", "chrome.po"))
        return {e.msgid: e.msgstr for e in po if e.msgstr}
    except Exception:
        return {}
_chrome = _load_chrome(language)
def setup(app):
    def _ctx(app, pagename, templatename, context, doctree):
        context["gf_t"] = lambda s: _chrome.get(s, s)
    app.connect("html-page-context", _ctx)

    # A plain `sphinx-build` (e.g. the default Read the Docs build) produces English at the
    # site root and then builds every other language into <outdir>/<lang>/ so the language
    # switcher works without any custom CI. Set DOCS_SUBBUILD=1 to skip (used internally).
    def _all_langs(app, exc):
        import subprocess, sys, tempfile
        if exc or app.builder.name != "html" or language != "en":
            return
        if _os.environ.get("DOCS_SUBBUILD") or _os.environ.get("DOCS_SKIP_OTHER_LANGS"):
            return
        for code, _label in LANGUAGES:
            if code == "en":
                continue
            env = dict(_os.environ, DOCS_LANG=code, DOCS_SUBBUILD="1")
            with tempfile.TemporaryDirectory() as dt:
                r = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", "-q", "-d", dt,
                                    "-D", "language=" + code, str(app.srcdir),
                                    _os.path.join(str(app.outdir), code)], env=env)
            if r.returncode:
                print("WARNING: language build failed:", code)
    app.connect("build-finished", _all_langs)
html_context["gf_partial"] = False
html_context["gf_rtl"] = language in ("ar",)

# --- i18n fix: MyST overwrites a document's heading-slug table every time Sphinx re-parses a
# translated fragment, which breaks cross-page anchors in translated builds. Merge instead.
try:
    from myst_parser.mdit_to_docutils.base import DocutilsRenderer as _DR
    _orig_finalise = _DR._render_finalise
    def _render_finalise(self):
        env = getattr(self, "sphinx_env", None)
        old = dict(env.metadata[env.docname].get("myst_slugs", {})) if env else {}
        _orig_finalise(self)
        if env and old:
            new = env.metadata[env.docname].get("myst_slugs", {})
            env.metadata[env.docname]["myst_slugs"] = {**new, **old}
    _DR._render_finalise = _render_finalise
except Exception:
    pass
