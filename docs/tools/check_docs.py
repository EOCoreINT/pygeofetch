#!/usr/bin/env python3
"""CI check: the generated CLI reference must match the installed CLI.

    python docs/tools/check_docs.py

Exits non-zero, with a diff, if ``reference/cli-commands.md`` is stale.
Regenerate it with:

    python docs/tools/gen_cli_reference.py > docs/reference/cli-commands.md
"""
from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "reference" / "cli-commands.md"


def main() -> int:
    fresh = subprocess.run(
        [sys.executable, str(HERE / "gen_cli_reference.py")],
        capture_output=True, text=True, check=True,
    ).stdout
    current = PAGE.read_text(encoding="utf-8")
    if fresh == current:
        print("cli-commands.md is up to date")
        return 0
    diff = difflib.unified_diff(
        current.splitlines(), fresh.splitlines(),
        "committed", "generated", lineterm="", n=2,
    )
    print("\n".join(list(diff)[:80]))
    print("\ncli-commands.md is stale; regenerate it (see this file's docstring).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
