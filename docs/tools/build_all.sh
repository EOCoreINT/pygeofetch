#!/usr/bin/env bash
# Build the site: English at build/ (site root) and every other language in build/<lang>/.
# The English build triggers the other languages itself (see conf.py), so a plain
# `sphinx-build -b html docs docs/_build/html` (as Read the Docs runs it) gives the same result.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build
python3 -m sphinx -b html -W --keep-going -q . build
