#!/usr/bin/env python3
"""Predict validation Loss with the frozen A+B joint scaling law."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping


DEFAULT_MODEL = Path(__file__).resolve().with_name("model.json")
P_COLUMNS = [f"p_{index}" for index in range(1, 18)]
REQUIRED_COLUMNS = ["N_params_B", "D_tokens_B", "Q_score", *P_COLUMNS]


def _dot(coefficients: list[float], features: list[float]) -> float:
    if len(coefficients) != len(features):
        raise ValueError("model coefficient and feature lengths differ")
    return float(sum(float(left) * float(right) for left, right in zip(coefficients, features)))


def _float_value(value: object, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return result


def _base_features(parameter_count: float, token_count: float) -> list[float]:
    log_n = math.log(parameter_count)
    log_d = math.log(token_count)
    return [
        1.0,
        parameter_count ** -0.25,
        token_count ** -0.25,
        parameter_count ** -0.50,
        token_count ** -0.50,
        log_n,
        log_d,
        log_n * log_d,
    ]


def _quality_features(parameter_count: float, token_count: float, quality_score: float) -> list[float]:
    log_n = math.log(parameter_count)
    log_d = math.log(token_count)
    gap = 1.0 - quality_score
    features = [1.0, log_n, log_d, log_n * log_d]
    for power in range(1, 6):
        features.append(gap**power)
    for power in range(1, 4):
        features.append((gap**power) * log_n)
        features.append((gap**power) * log_d)
    return features


def _normalise_proportions(values: list[float], row_number: int) -> list[float]:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"row {row_number}: p_1...p_17 must be finite and nonnegative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError(f"row {row_number}: p_1...p_17 must have positive sum")
    return [value / total for value in values]


def _proportions_from_row(row: Mapping[str, object], row_number: int) -> list[float]:
    values = [_float_value(row.get(column), column, row_number) for column in P_COLUMNS]
    return _normalise_proportions(values, row_number)


def _p_raw_prediction(payload: Mapping[str, object], proportions: list[float]) -> float:
    adapter = payload["p_adapter"]
    offsets = [float(value) for value in adapter["offsets"]]
    beta = [[float(value) for value in row] for row in adapter["beta"]]
    values = [
        offset + math.exp(max(-30.0, min(15.0, _dot(row, proportions))))
        for offset, row in zip(offsets, beta)
    ]
    return float(sum(values) / len(values))


def _p_reference(payload: Mapping[str, object]) -> float:
    return float(payload["p_adapter"]["reference_raw_prediction"])


def _gamma(payload: Mapping[str, object], parameter_count: float) -> float:
    adapter = payload["p_adapter"]["gamma_model"]
    reference_n = float(adapter["reference_n"])
    value = float(adapter["intercept"]) + float(adapter["slope"]) * math.log(parameter_count / reference_n)
    return max(float(adapter["clip_low"]), min(float(adapter["clip_high"]), value))


def predict_joint(
    payload: Mapping[str, object],
    parameter_count: float,
    token_count: float,
    quality_score: float,
    proportions: list[float],
) -> float:
    base_payload = payload["b_only_model"]["base_model"]
    quality_payload = payload["b_only_model"]["quality_effect_model"]
    base = _dot(base_payload["coefficients"], _base_features(parameter_count, token_count))
    quality = _dot(
        quality_payload["coefficients"],
        _quality_features(parameter_count, token_count, quality_score),
    )
    quality_reference = _dot(
        quality_payload["coefficients"],
        _quality_features(parameter_count, token_count, 1.0),
    )
    p_correction = _gamma(payload, parameter_count) * (
        _p_raw_prediction(payload, proportions) - _p_reference(payload)
    )
    return float(base + quality - quality_reference + p_correction)


def predict_csv(input_path: Path, output_path: Path) -> None:
    payload = json.loads(DEFAULT_MODEL.read_text(encoding="utf-8"))
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError("input CSV must contain a header row")
        missing = sorted(set(REQUIRED_COLUMNS).difference(fields))
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(missing)}")
        if "prediction" in fields:
            raise ValueError("input CSV must not contain a prediction column")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            parameter_count = _float_value(row.get("N_params_B"), "N_params_B", row_number)
            token_count = _float_value(row.get("D_tokens_B"), "D_tokens_B", row_number)
            quality_score = _float_value(row.get("Q_score"), "Q_score", row_number)
            if parameter_count <= 0.0:
                raise ValueError(f"row {row_number}: N_params_B must be greater than zero")
            if token_count <= 0.0:
                raise ValueError(f"row {row_number}: D_tokens_B must be greater than zero")
            if not 0.0 <= quality_score <= 1.0:
                raise ValueError(f"row {row_number}: Q_score must be in [0, 1]")

            proportions = _proportions_from_row(row, row_number)
            output = dict(row)
            output["prediction"] = predict_joint(
                payload,
                parameter_count,
                token_count,
                quality_score,
                proportions,
            )
            rows.append(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["prediction"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        predict_csv(args.input, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
