#!/usr/bin/env python3
"""Run the 4 Stage-4 Midpoint Reference DeLM arms sharded across KEY_SLOT=1..4."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    for extra_path in (str(package_root / "src"), str(package_root / "vendor" / "delm")):
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)
    cli_main = importlib.import_module("delm_cb.cli").main
    return int(cli_main(["suite", *sys.argv[1:]]))


if __name__ == "__main__":
    sys.exit(_main())
