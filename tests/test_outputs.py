from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("FREDHOLM_ENTRY", "/app/predict.py")


def _run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _weights(times: list[float]) -> list[float]:
    values = [0.0 for _ in times]
    for index in range(len(times) - 1):
        span = (times[index + 1] - times[index]) / 2.0
        values[index] += span
        values[index + 1] += span
    return values


def _payload(
    gain: float = -0.65,
    width: float = 12.0,
    count: int = 24,
) -> dict[str, object]:
    times = [index / (count - 1) for index in range(count)]
    response = [
        1.3
        + 0.45 * math.cos(2.0 * math.pi * time)
        + 0.18 * math.exp(-3.0 * time)
        + 0.08 * math.sin(4.0 * math.pi * time)
        for time in times
    ]
    weights = _weights(times)
    source = []
    for row, time in enumerate(times):
        integral = math.fsum(
            weights[column]
            * math.exp(-width * (time - times[column]) ** 2)
            * response[column]
            for column in range(count)
        )
        source.append(response[row] - gain * integral)
    return {
        "schema": "fredholm-scan-v1",
        "samples": [
            {
                "time": times[index],
                "source": round(source[index], 10),
                "response": round(response[index], 10),
            }
            for index in range(count)
        ],
        "fit_tolerance": 0.000001,
    }


def test_negative_feedback_curve_recovers_gain_and_width(tmp_path: Path) -> None:
    """题面：
    「`samples` 是 12 到 40 个三元记录，每条记录只含 `time`、`source`、`response`。」
    """

    payload = _payload()
    source = tmp_path / "curve.json"
    output = tmp_path / "parameters.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["gain"] == pytest.approx(-0.65, abs=0.01)
    assert result["kernel_width"] == pytest.approx(12.0, rel=0.02)


def test_broad_kernel_recovery_distinguishes_width_scale(tmp_path: Path) -> None:
    """题面：
    「误差合格后，报告只需 `gain`、`kernel_width`、`rms_response`；均保留六位小数。」
    """

    payload = _payload(-0.35, 28.0, 28)
    source = tmp_path / "broad.json"
    output = tmp_path / "broad-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["kernel_width"] == pytest.approx(28.0, rel=0.025)
    assert result["gain"] == pytest.approx(-0.35, abs=0.012)


def test_positive_feedback_curve_recovers_continuous_gain(tmp_path: Path) -> None:
    """题面：
    「`time` 在 0 到 1 之间并严格递增，第一条必须为 0，最后一条必须为 1。」
    """

    payload = _payload(0.45, 18.0, 26)
    source = tmp_path / "positive.json"
    output = tmp_path / "positive-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["gain"] == pytest.approx(0.45, abs=0.012)
    assert result["kernel_width"] == pytest.approx(18.0, rel=0.025)


def test_fitted_fredholm_system_closes_all_response_points(tmp_path: Path) -> None:
    """题面：
    「输入对象由 `schema`、`samples`、`fit_tolerance` 三个成员构成，不允许其他成员。」
    """

    # FRED-ACCEPT-12
    payload = _payload(-0.8, 5.0, 12)
    source = tmp_path / "closure.json"
    output = tmp_path / "closure-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["rms_response"] <= payload["fit_tolerance"]
    assert result["rms_response"] == round(result["rms_response"], 6)


def test_extra_top_level_key_stops_before_linear_solve(tmp_path: Path) -> None:
    """题面：
    「标准错误仅限一行带 `REFUSED: ` 前缀的原因，退出状态固定为 2。」
    """

    # FRED-TOP FRED-FIELDS FRED-SCHEMA FRED-SAMPLEKEYS FRED-SAMPLEFIELDS FRED-REJECTCHANNEL
    payload = _payload()
    payload["detector"] = "unexpected"
    source = tmp_path / "extra.json"
    output = tmp_path / "extra-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_missing_right_endpoint_is_refused(tmp_path: Path) -> None:
    """题面：
    「现有脚本 `/app/predict.py` 接收两个路径参数。求解器必须基于」
    """

    # FRED-TIME FRED-ENDPOINTS
    payload = _payload()
    payload["samples"][-1]["time"] = 0.98
    source = tmp_path / "endpoint.json"
    output = tmp_path / "endpoint-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_response_is_refused_as_non_numeric(tmp_path: Path) -> None:
    """题面：
    「磁盘上采用无空白 JSON，键字母升序，行末补换行。重复处理同一输入时，」
    """

    payload = _payload()
    payload["samples"][5]["response"] = False
    source = tmp_path / "boolean.json"
    output = tmp_path / "boolean-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_eleven_samples_are_refused_before_linear_solve(tmp_path: Path) -> None:
    """题面：
    「输出文件字节应完全一致。」
    """

    # FRED-SAMPLES-11
    payload = _payload(count=11)
    source = tmp_path / "short-curve.json"
    output = tmp_path / "parameters.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists(), "FRED-SAMPLES-11"


def test_output_is_compact_sorted_and_repeatable(tmp_path: Path) -> None:
    """题面：
    「若没有参数满足误差要求，或输入结构、类型、顺序、端点不合规，就不创建输出文件。」
    """

    # FRED-CLI FRED-FORMAT FRED-REPEAT FRED-NOTESTS
    payload = _payload()
    source = tmp_path / "format.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(source, payload)
    first_run = _run(source, first)
    second_run = _run(source, second)
    raw = first.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert first_run.returncode == 0
    assert second_run.returncode == 0
    assert raw == canonical
    assert raw == second.read_bytes()


def test_gate_requires_solved_response_equation(tmp_path: Path) -> None:
    """题面：
    「`samples` 是 12 到 40 个三元记录，每条记录只含 `time`、`source`、`response`。」
    """

    # FRED-OUTPUT FRED-OUTKEYS FRED-SYSTEM
    payload = _payload(-0.35, 28.0, 28)
    source = tmp_path / "gate.json"
    output = tmp_path / "gate-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["rms_response"] <= payload["fit_tolerance"]
    assert result["gain"] == pytest.approx(-0.35, abs=0.02)
