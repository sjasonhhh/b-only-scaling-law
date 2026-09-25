from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "results" / "B_only_model.json"
DEFAULT_D_FLOOR = 0.134
DEFAULT_INTERVAL_Z = 1.6448536269514722


def _float_value(value: Optional[str], field: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: {field} is required")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return number


def _dot(coefficients: Sequence[float], features: Sequence[float]) -> float:
    if len(coefficients) != len(features):
        raise ValueError("model parameter length does not match feature length")
    return sum(float(coefficient) * float(feature) for coefficient, feature in zip(coefficients, features))


def _base_features(parameter_count: float, token_count: float) -> List[float]:
    log_parameter_count = math.log(parameter_count)
    log_token_count = math.log(token_count)
    return [
        1.0,
        parameter_count ** -0.25,
        token_count ** -0.25,
        parameter_count ** -0.50,
        token_count ** -0.50,
        log_parameter_count,
        log_token_count,
        log_parameter_count * log_token_count,
    ]


def _quality_features(parameter_count: float, token_count: float, quality_score: float) -> List[float]:
    log_parameter_count = math.log(parameter_count)
    log_token_count = math.log(token_count)
    quality_gap = 1.0 - quality_score
    features = [
        1.0,
        log_parameter_count,
        log_token_count,
        log_parameter_count * log_token_count,
    ]
    for power in range(1, 6):
        features.append(quality_gap**power)
    for power in range(1, 4):
        gap_power = quality_gap**power
        features.append(gap_power * log_parameter_count)
        features.append(gap_power * log_token_count)
    return features


def _prediction_for_row(
    payload: Mapping[str, object],
    row: Mapping[str, str],
    row_number: int,
    has_quality_column: bool,
) -> Dict[str, object]:
    parameter_count = _float_value(row.get("N_params_B"), "N_params_B", row_number)
    if parameter_count <= 0:
        raise ValueError(f"row {row_number}: N_params_B must be greater than zero")

    raw_tokens = row.get("D_tokens_B")
    d_imputed = raw_tokens is None or not raw_tokens.strip()
    if d_imputed:
        token_count = float(payload.get("D_floor_for_missing_scenarios", DEFAULT_D_FLOOR))
    else:
        token_count = _float_value(raw_tokens, "D_tokens_B", row_number)
        if token_count <= 0:
            token_count = float(payload.get("D_floor_for_missing_scenarios", DEFAULT_D_FLOOR))
            d_imputed = True

    base_payload = payload["base_model"]
    quality_payload = payload["quality_effect_model"]
    base_prediction = _dot(
        base_payload["coefficients"],
        _base_features(parameter_count, token_count),
    )

    raw_quality = row.get("Q_score") if has_quality_column else None
    use_quality = raw_quality is not None and bool(raw_quality.strip())
    if use_quality:
        quality_score = _float_value(raw_quality, "Q_score", row_number)
        quality_prediction = _dot(
            quality_payload["coefficients"],
            _quality_features(parameter_count, token_count, quality_score),
        )
        reference_prediction = _dot(
            quality_payload["coefficients"],
            _quality_features(parameter_count, token_count, 1.0),
        )
        prediction = base_prediction + quality_prediction - reference_prediction
        base_scale = float(base_payload.get("residual_scale", 0.0))
        quality_scale = float(quality_payload.get("residual_scale", 0.0))
        prediction_scale = math.sqrt(base_scale**2 + quality_scale**2)
    else:
        prediction = base_prediction
        prediction_scale = float(base_payload.get("residual_scale", 0.0))

    interval = payload.get("prediction_interval", {})
    interval_z = float(interval.get("z", DEFAULT_INTERVAL_Z))
    half_width = interval_z * prediction_scale
    output = dict(row)
    output.update(
        {
            "D_tokens_B_used": token_count,
            "D_tokens_B_imputed": d_imputed,
            "prediction": prediction,
            "prediction_std": prediction_scale,
            "prediction_low_90": prediction - half_width,
            "prediction_high_90": prediction + half_width,
            "prediction_interval_z": interval_z,
            "prediction_status": "D_floor_imputed" if d_imputed else "ok",
        }
    )
    return output


def predict_csv(model_path: Path, input_path: Path, output_path: Path) -> None:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        required = {"N_params_B", "D_tokens_B"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(missing)}")
        rows = [
            _prediction_for_row(payload, row, row_number, "Q_score" in fieldnames)
            for row_number, row in enumerate(reader, start=2)
        ]

    output_fields = fieldnames + [
        "D_tokens_B_used",
        "D_tokens_B_imputed",
        "prediction",
        "prediction_std",
        "prediction_low_90",
        "prediction_high_90",
        "prediction_interval_z",
        "prediction_status",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict validation Loss with the frozen B-only model")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing N_params_B, D_tokens_B and optional Q_score")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    predict_csv(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
