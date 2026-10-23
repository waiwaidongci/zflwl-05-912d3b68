#!/usr/bin/env python3
"""Starter entry point for Fredholm response inversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("REFUSED: input must be an object", file=sys.stderr)
        return 2
    result = {
        "gain": -0.45,
        "kernel_width": 9.0,
        "rms_response": 0.0,
    }
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
