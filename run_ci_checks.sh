#!/usr/bin/env bash
# Runs exactly what .github/workflows/tests.yml runs, using the venv's own tools.
# (apt-installed ruff/black/mypy are other versions, and a system mypy cannot
#  import the pydantic plugin because pydantic lives in the venv.)
cd "$(dirname "$0")" && cd "$(git rev-parse --show-toplevel 2>/dev/null || echo ..)" || exit 1
source venv/bin/activate 2>/dev/null
python -m pip install -q -U ruff black mypy types-requests types-PyYAML types-python-dateutil
rc=0
echo "== ruff";   python -m ruff check pygeofetch/                    || rc=1
echo "== black";  python -m black --check pygeofetch/ tests/          || rc=1
echo "== mypy";   python -m mypy pygeofetch/ --ignore-missing-imports || rc=1
echo "== pytest"; python -m pytest tests/ -q                          || rc=1
exit $rc
