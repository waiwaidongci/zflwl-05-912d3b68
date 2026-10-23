#!/usr/bin/env python3
"""Reference solver for a second-kind Fredholm response inverse problem."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {"schema", "samples", "fit_tolerance"}
SAMPLE_KEYS = {"time", "source", "response"}


class Refused(ValueError):
    pass


def _number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise Refused(f"{label} is outside its allowed range")
    return number


def parse_input(path: Path) -> tuple[list[dict[str, float]], float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level keys do not match fredholm-scan-v1")
    if payload.get("schema") != "fredholm-scan-v1":
        raise Refused("unsupported schema")
    tolerance = _number(payload.get("fit_tolerance"), "fit_tolerance", 1e-6, 0.1)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not 12 <= len(raw_samples) <= 40:
        raise Refused("samples must contain between 12 and 40 entries")
    samples: list[dict[str, float]] = []
    previous_time = -1.0
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict) or set(raw) != SAMPLE_KEYS:
            raise Refused(f"sample {index} has an invalid key set")
        time = _number(raw.get("time"), "sample time", 0.0, 1.0)
        if time <= previous_time:
            raise Refused("sample times must strictly increase")
        previous_time = time
        samples.append(
            {
                "time": time,
                "source": _number(raw.get("source"), "source", -20.0, 20.0),
                "response": _number(raw.get("response"), "response", -20.0, 20.0),
            }
        )
    if samples[0]["time"] != 0.0 or samples[-1]["time"] != 1.0:
        raise Refused("sample times must span the interval from zero to one")
    return samples, tolerance


def _weights(times: list[float]) -> list[float]:
    weights = [0.0 for _ in times]
    for index in range(len(times) - 1):
        span = (times[index + 1] - times[index]) / 2.0
        weights[index] += span
        weights[index + 1] += span
    return weights


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    work = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-14:
            raise Refused("discrete response system is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for item in range(column, size + 1):
            work[column][item] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for item in range(column, size + 1):
                work[row][item] -= factor * work[column][item]
    return [work[row][size] for row in range(size)]


def predict(
    samples: list[dict[str, float]],
    gain: float,
    width: float,
) -> list[float]:
    times = [sample["time"] for sample in samples]
    sources = [sample["source"] for sample in samples]
    weights = _weights(times)
    matrix: list[list[float]] = []
    for row, time in enumerate(times):
        current = []
        for column, other_time in enumerate(times):
            kernel = math.exp(-width * (time - other_time) ** 2)
            current.append(
                (1.0 if row == column else 0.0)
                - gain * weights[column] * kernel
            )
        matrix.append(current)
    return _solve(matrix, sources)


def _parameter(z: list[float]) -> tuple[float, float]:
    gain = -0.95 + 1.9 * z[0]
    width = math.exp(math.log(0.1) + math.log(800.0) * z[1])
    return gain, width


def _residuals(
    z: list[float],
    samples: list[dict[str, float]],
) -> list[float]:
    gain, width = _parameter(z)
    try:
        predictions = predict(samples, gain, width)
    except Refused:
        return [1e6 for _ in samples]
    return [
        prediction - sample["response"]
        for prediction, sample in zip(predictions, samples)
    ]


def _fit_start(
    start: list[float],
    samples: list[dict[str, float]],
) -> tuple[float, list[float]]:
    z = [min(1.0, max(0.0, value)) for value in start]
    residuals = _residuals(z, samples)
    sse = math.fsum(value * value for value in residuals)
    damping = 1e-3
    for _ in range(60):
        jacobian = [[0.0, 0.0] for _ in residuals]
        for column in range(2):
            step = 1e-5
            plus = z[:]
            minus = z[:]
            plus[column] = min(1.0, plus[column] + step)
            minus[column] = max(0.0, minus[column] - step)
            span = plus[column] - minus[column]
            if span <= 1e-12:
                continue
            r_plus = _residuals(plus, samples)
            r_minus = _residuals(minus, samples)
            for row in range(len(residuals)):
                jacobian[row][column] = (r_plus[row] - r_minus[row]) / span
        normal = [[0.0, 0.0], [0.0, 0.0]]
        gradient = [0.0, 0.0]
        for row, residual in enumerate(residuals):
            for left in range(2):
                gradient[left] += jacobian[row][left] * residual
                for right in range(2):
                    normal[left][right] += jacobian[row][left] * jacobian[row][right]
        for index in range(2):
            normal[index][index] += damping * max(1.0, normal[index][index])
        determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
        if abs(determinant) < 1e-16:
            break
        delta_0 = (
            -gradient[0] * normal[1][1] + gradient[1] * normal[0][1]
        ) / determinant
        delta_1 = (
            -gradient[1] * normal[0][0] + gradient[0] * normal[1][0]
        ) / determinant
        candidate = [
            min(1.0, max(0.0, z[0] + delta_0)),
            min(1.0, max(0.0, z[1] + delta_1)),
        ]
        candidate_residuals = _residuals(candidate, samples)
        candidate_sse = math.fsum(value * value for value in candidate_residuals)
        movement = math.hypot(candidate[0] - z[0], candidate[1] - z[1])
        if candidate_sse < sse:
            z = candidate
            residuals = candidate_residuals
            sse = candidate_sse
            damping = max(1e-10, damping / 3.0)
            if movement < 1e-10:
                break
        else:
            damping *= 8.0
            if damping > 1e9:
                break
    return sse, z


def fit(
    samples: list[dict[str, float]],
    tolerance: float,
) -> dict[str, float]:
    starts = [
        [0.30, 0.35],
        [0.55, 0.55],
        [0.72, 0.72],
        [0.42, 0.18],
        [0.62, 0.88],
    ]
    candidates = [_fit_start(start, samples) for start in starts]
    sse, z = min(candidates, key=lambda item: item[0])
    rms = math.sqrt(sse / len(samples))
    if rms > tolerance:
        raise Refused("Fredholm model cannot reach the requested response tolerance")
    gain, width = _parameter(z)
    return {
        "gain": round(gain, 6),
        "kernel_width": round(width, 6),
        "rms_response": round(rms, 6),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        samples, tolerance = parse_input(source)
        result = fit(samples, tolerance)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    # 输出统一采用排序键编码，使重复数值搜索也能逐字节复核。
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
