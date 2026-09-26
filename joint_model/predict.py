#!/usr/bin/env python3
"""Predict with the internal A+B joint scaling-law model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "joint_model" / "results" / "joint_model.json"
P_COLUMNS = [f"p_{index}" for index in range(1, 18)]
DEFAULT_INTERVAL_Z = 1.6448536269514722


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


def _p_raw_prediction(payload: Mapping[str, object], proportions: list[float]) -> float:
    adapter = payload["p_adapter"]
    offsets = [float(value) for value in adapter["offsets"]]
    beta = [[float(value) for value in row] for row in adapter["beta"]]
    values = [offset + math.exp(max(-30.0, min(15.0, _dot(row, proportions))))
              for offset, row in zip(offsets, beta)]
    return float(sum(values) / len(values))


def _p_reference(payload: Mapping[str, object]) -> float:
    return float(payload["p_adapter"]["reference_raw_prediction"])


def _gamma_from_config(adapter: Mapping[str, object], parameter_count: float) -> float:
    maximum_interpolation_n = adapter.get("maximum_interpolation_n")
    if maximum_interpolation_n is not None and parameter_count > float(maximum_interpolation_n):
        return float(adapter.get("high_n_gamma", 0.0))
    reference_n = float(adapter["reference_n"])
    value = float(adapter["intercept"]) + float(adapter["slope"]) * math.log(parameter_count / reference_n)
    return max(float(adapter["clip_low"]), min(float(adapter["clip_high"]), value))


def _gamma(payload: Mapping[str, object], parameter_count: float) -> float:
    return _gamma_from_config(payload["p_adapter"]["gamma_model"], parameter_count)


def predict_b(payload: Mapping[str, object], parameter_count: float, token_count: float,
              quality_score: float, proportions: list[float] | None = None) -> tuple[float, float]:
    base_payload = payload["b_only_model"]["base_model"]
    quality_payload = payload["b_only_model"]["quality_effect_model"]
    base = _dot(base_payload["coefficients"], _base_features(parameter_count, token_count))
    quality = _dot(quality_payload["coefficients"], _quality_features(parameter_count, token_count, quality_score))
    reference = _dot(quality_payload["coefficients"], _quality_features(parameter_count, token_count, 1.0))
    base_prediction = base + quality - reference
    if proportions is None:
        return base_prediction, 0.0
    correction = _gamma(payload, parameter_count) * (_p_raw_prediction(payload, proportions) - _p_reference(payload))
    return base_prediction + correction, correction


def _scale_offset(payload: Mapping[str, object], parameter_count: float) -> float:
    adapter = payload["a_adapter"]
    x = math.log(parameter_count / float(adapter["reference_n"]))
    coefficients = [float(value) for value in adapter["offset_polynomial"]]
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def predict_a(payload: Mapping[str, object], parameter_count: float, proportions: list[float]) -> float:
    adapter = payload.get("a_adapter", {})
    p_gamma_config = adapter.get("p_gamma_model", {"reference_n": 0.001, "intercept": 1.0,
                                                       "slope": 0.0, "clip_low": 0.0, "clip_high": 1.0})
    p_gamma = _gamma_from_config(p_gamma_config, parameter_count)
    p_prediction = _p_reference(payload) + p_gamma * (_p_raw_prediction(payload, proportions) - _p_reference(payload))
    return p_prediction + _scale_offset(payload, parameter_count)


def _proportions_from_row(row: Mapping[str, str], row_number: int) -> list[float] | None:
    present = [column in row and str(row[column]).strip() for column in P_COLUMNS]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"row {row_number}: either provide all p_1...p_17 columns or none")
    return _normalise_proportions([_float_value(row[column], column, row_number) for column in P_COLUMNS], row_number)


def predict_csv(model_path: Path, input_path: Path, output_path: Path, mode: str) -> None:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        required = {"N_params_B"}
        if mode == "b":
            required.add("D_tokens_B")
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(missing)}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            parameter_count = _float_value(row.get("N_params_B"), "N_params_B", row_number)
            if parameter_count <= 0.0:
                raise ValueError(f"row {row_number}: N_params_B must be greater than zero")
            proportions = _proportions_from_row(row, row_number)
            if mode == "a":
                if proportions is None:
                    raise ValueError(f"row {row_number}: A mode requires p_1...p_17")
                prediction = predict_a(payload, parameter_count, proportions)
                correction = 0.0
            else:
                raw_tokens = row.get("D_tokens_B")
                d_imputed = raw_tokens is None or not str(raw_tokens).strip()
                if d_imputed:
                    token_count = float(payload["b_only_model"].get("D_floor_for_missing_scenarios", 0.134))
                else:
                    token_count = _float_value(raw_tokens, "D_tokens_B", row_number)
                    if token_count <= 0.0:
                        token_count = float(payload["b_only_model"].get("D_floor_for_missing_scenarios", 0.134))
                        d_imputed = True
                raw_quality = str(row.get("Q_score", "")).strip()
                quality_score = 1.0 if not raw_quality else _float_value(raw_quality, "Q_score", row_number)
                if not 0.0 <= quality_score <= 1.0:
                    raise ValueError(f"row {row_number}: Q_score must be in [0, 1]")
                prediction, correction = predict_b(payload, parameter_count, token_count, quality_score, proportions)
            output = dict(row)
            output.update({
                "prediction": prediction,
                "p_correction": correction,
                "prediction_mode": mode,
            })
            if mode == "b":
                output.update({
                    "D_tokens_B_used": token_count,
                    "D_tokens_B_imputed": d_imputed,
                    "prediction_status": "D_floor_imputed" if d_imputed else "ok",
                })
            rows.append(output)
    output_fields = fields + ["prediction", "p_correction", "prediction_mode"]
    if mode == "b":
        output_fields += ["D_tokens_B_used", "D_tokens_B_imputed", "prediction_status"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("a", "b"), default="b")
    args = parser.parse_args()
    predict_csv(args.model, args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
