from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
from test_outputs import ENTRY, _payload, _run, _write_json  # noqa: E402


def _solve(
    tmp_path: Path,
    label: str,
    gain: float,
    width: float,
    count: int,
) -> dict[str, float]:
    payload = _payload(gain, width, count)
    source = tmp_path / f"{label}.json"
    output = tmp_path / f"{label}-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    return json.loads(output.read_text(encoding="utf-8"))


def test_two_curves_require_real_fredholm_inversion(tmp_path: Path) -> None:
    first = _solve(tmp_path, "curve-a", -0.58, 10.5, 18)
    second = _solve(tmp_path, "curve-b", 0.42, 33.0, 30)

    assert first["gain"] == pytest.approx(-0.58, abs=0.015)
    assert first["kernel_width"] == pytest.approx(10.5, rel=0.03)
    assert second["gain"] == pytest.approx(0.42, abs=0.015)
    assert second["kernel_width"] == pytest.approx(33.0, rel=0.035)
    assert first["gain"] != second["gain"]
    assert first["kernel_width"] != second["kernel_width"]

