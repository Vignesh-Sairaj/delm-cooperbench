#!/usr/bin/env python3
"""Run the 4 Stage-4 Midpoint Reference DeLM arms sharded across KEY_SLOT=1..4."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (str(PACKAGE_ROOT / "src"), str(PACKAGE_ROOT / "vendor" / "delm")):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from delm_cb.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["suite", *sys.argv[1:]]))
