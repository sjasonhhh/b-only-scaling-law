from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
B_DIR = ROOT / "real_attachments" / "B_scaling_laws"
RESULTS_DIR = ROOT / "b_model" / "results"
D_FLOOR = 0.134
PREDICTION_INTERVAL_Z = 1.6448536269514722

DATA_ROLES = {
    "pythia_training_log_existing.csv": "fit_base",
    "cerebras_training_log.csv": "feedback_family",
    "scaling_baseline.csv": "locked_external",
    "published_scaling_data.csv": "locked_external",
    "supplementary_NQ_experiment.csv": "fit_quality",
    "supplementary_NQ_experiment_expanded.csv": "quality_new90_validation",
    "supplementary_NQ_experiment_large.csv": "stress_quality",
    "supplementary_large_models.csv": "large_model_scenario",
    "supplementary_large_baseline.csv": "estimated_loss_scenario",
    "open_model_family_metadata.csv": "metadata_only",
    "pythia_checkpoint_index.csv": "metadata_only",
}

EXPECTED_FIELDS = {
    "pythia_training_log_existing.csv": [
        "run_id", "N_params_B", "D_tokens_B", "C_FLOPs_1e21", "steps", "batch_tokens_M",
        "lr", "wd", "precision", "gpu_days", "step_time_ms", "train_loss", "val_loss", "ppl", "grad_norm_avg",
    ],
    "cerebras_training_log.csv": [
        "run_id", "N_params_B", "D_tokens_B", "C_FLOPs_1e21", "steps", "batch_tokens_M",
        "lr", "wd", "precision", "gpu_days", "step_time_ms", "train_loss", "val_loss", "ppl", "grad_norm_avg",
    ],
    "scaling_baseline.csv": ["family", "N_params_B", "D_tokens_B", "val_loss", "is_converged"],
    "published_scaling_data.csv": ["family", "N_params_B", "D_tokens_B", "val_loss", "source", "is_converged"],
    "supplementary_NQ_experiment.csv": ["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"],
    "supplementary_NQ_experiment_expanded.csv": ["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"],
    "supplementary_NQ_experiment_large.csv": ["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss", "data_type"],
    "supplementary_large_models.csv": ["model_name", "N_params_B", "D_tokens_B", "FLOPs", "publication_date", "organization", "accessibility", "country"],
    "supplementary_large_baseline.csv": ["family", "N_params_B", "D_tokens_B", "val_loss", "is_converged"],
    "open_model_family_metadata.csv": ["family", "model_repo", "config_bytes", "readme_bytes", "weight_file_count", "weight_total_bytes", "source_url", "error"],
    "pythia_checkpoint_index.csv": ["model_repo", "model_size", "step", "branch", "commit"],
    "__trajectory__": ["N_params_B", "D_tokens_B", "val_loss", "step", "interpolated"],
}

QUALITY_KEY_FIELDS = {
    "pythia_training_log_existing.csv": ["run_id"],
    "cerebras_training_log.csv": ["run_id", "steps"],
    "scaling_baseline.csv": ["family", "N_params_B", "D_tokens_B"],
    "published_scaling_data.csv": ["family", "N_params_B", "D_tokens_B"],
    "supplementary_NQ_experiment.csv": ["experiment_id"],
    "supplementary_NQ_experiment_expanded.csv": ["experiment_id"],
    "supplementary_NQ_experiment_large.csv": ["experiment_id"],
    "supplementary_large_models.csv": ["model_name"],
    "supplementary_large_baseline.csv": ["family", "N_params_B", "D_tokens_B"],
    "open_model_family_metadata.csv": ["family", "model_repo"],
    "pythia_checkpoint_index.csv": ["model_repo", "step", "branch"],
    "__trajectory__": ["N_params_B", "D_tokens_B", "step"],
}

BASE_FEATURE_NAMES = [
    "intercept",
    "N^-0.25",
    "D^-0.25",
    "N^-0.50",
    "D^-0.50",
    "log_N",
    "log_D",
    "log_N_times_log_D",
]

QUALITY_FEATURE_NAMES = [
    "intercept",
    "log_N",
    "log_D",
    "log_N_times_log_D",
    "(1-Q)^1",
    "(1-Q)^2",
    "(1-Q)^3",
    "(1-Q)^4",
    "(1-Q)^5",
    "(1-Q)^1_times_log_N",
    "(1-Q)^1_times_log_D",
    "(1-Q)^2_times_log_N",
    "(1-Q)^2_times_log_D",
    "(1-Q)^3_times_log_N",
    "(1-Q)^3_times_log_D",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(B_DIR / name)


def data_quality_check(path: Path, frame: pd.DataFrame) -> dict:
    schema_key = "__trajectory__" if path.parent.name == "training_trajectories" else path.name
    expected_fields = EXPECTED_FIELDS.get(schema_key, [])
    key_fields = QUALITY_KEY_FIELDS.get(schema_key, [])
    missing_required_fields = [field for field in expected_fields if field not in frame.columns]
    unexpected_fields = [field for field in frame.columns if field not in expected_fields]
    missing_by_field = {field: int(count) for field, count in frame.isna().sum().items() if count}
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    nonfinite_numeric_cells = 0
    for field in numeric_columns:
        nonfinite_numeric_cells += int((~np.isfinite(frame[field].dropna().to_numpy(dtype=float))).sum())
    duplicate_key_rows = 0
    if key_fields and all(field in frame.columns for field in key_fields):
        duplicate_key_rows = int(frame.duplicated(subset=key_fields).sum())
    field_errors = [f"missing:{field}" for field in missing_required_fields]
    field_errors.extend(f"unexpected:{field}" for field in unexpected_fields)
    passed = not field_errors and nonfinite_numeric_cells == 0 and int(frame.duplicated().sum()) == 0 and duplicate_key_rows == 0
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "expected_fields": expected_fields,
        "key_fields": key_fields,
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_field": missing_by_field,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_key_rows": duplicate_key_rows,
        "nonfinite_numeric_cells": nonfinite_numeric_cells,
        "field_errors": field_errors,
        "passed": passed,
    }


def rank_average(values: Iterable[float]) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    sorted_values = values_array[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    if len(actual_array) < 2:
        return float("nan")
    actual_ranks = rank_average(actual_array)
    predicted_ranks = rank_average(predicted_array)
    actual_centered = actual_ranks - actual_ranks.mean()
    predicted_centered = predicted_ranks - predicted_ranks.mean()
    denominator = np.sqrt(np.sum(actual_centered**2) * np.sum(predicted_centered**2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(actual_centered * predicted_centered) / denominator)


def metric_dict(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    residual = actual_array - predicted_array
    total_sum_squares = np.sum((actual_array - actual_array.mean()) ** 2)
    r_squared = float("nan") if total_sum_squares == 0 else 1.0 - float(np.sum(residual**2) / total_sum_squares)
    return {
        "n": int(len(actual_array)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "bias": float(np.mean(predicted_array - actual_array)),
        "r2": r_squared,
        "spearman": spearman_correlation(actual_array, predicted_array),
    }


def base_features(frame: pd.DataFrame) -> np.ndarray:
    parameter_count = frame["N_params_B"].to_numpy(dtype=float)
    token_count = np.maximum(frame["D_tokens_B"].to_numpy(dtype=float), D_FLOOR)
    log_parameter_count = np.log(parameter_count)
    log_token_count = np.log(token_count)
    return np.column_stack(
        [
            np.ones(len(frame)),
            parameter_count ** -0.25,
            token_count ** -0.25,
            parameter_count ** -0.50,
            token_count ** -0.50,
            log_parameter_count,
            log_token_count,
            log_parameter_count * log_token_count,
        ]
    )


def quality_features(frame: pd.DataFrame) -> np.ndarray:
    parameter_count = frame["N_params_B"].to_numpy(dtype=float)
    token_count = np.maximum(frame["D_tokens_B"].to_numpy(dtype=float), D_FLOOR)
    quality_score = frame["Q_score"].to_numpy(dtype=float)
    log_parameter_count = np.log(parameter_count)
    log_token_count = np.log(token_count)
    quality_gap = 1.0 - quality_score
    columns = [
        np.ones(len(frame)),
        log_parameter_count,
        log_token_count,
        log_parameter_count * log_token_count,
    ]
    for power in range(1, 6):
        gap_power = quality_gap**power
        columns.append(gap_power)
    for power in range(1, 4):
        gap_power = quality_gap**power
        columns.append(gap_power * log_parameter_count)
        columns.append(gap_power * log_token_count)
    return np.column_stack(columns)


class LinearModel:
    def __init__(self, feature_names: list[str], ridge: float = 1e-10):
        self.feature_names = feature_names
        self.ridge = ridge
        self.coefficients: np.ndarray | None = None
        self.residual_scale: float | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LinearModel":
        penalty = np.eye(features.shape[1], dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        normal_matrix = features.T @ features + penalty
        right_hand_side = features.T @ target
        try:
            self.coefficients = np.linalg.solve(normal_matrix, right_hand_side)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(features, target, rcond=None)[0]
        residual = target - features @ self.coefficients
        degrees_of_freedom = max(len(target) - features.shape[1], 1)
        self.residual_scale = float(np.sqrt(np.sum(residual**2) / degrees_of_freedom))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted")
        return features @ self.coefficients

    def as_dict(self) -> dict:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted")
        return {
            "feature_names": self.feature_names,
            "coefficients": [float(value) for value in self.coefficients],
            "ridge": self.ridge,
            "residual_scale": self.residual_scale,
        }


def fit_base_model(frame: pd.DataFrame) -> LinearModel:
    model = LinearModel(BASE_FEATURE_NAMES)
    return model.fit(base_features(frame), frame["val_loss"].to_numpy(dtype=float))


def fit_quality_effect_model(base_model: LinearModel, frame: pd.DataFrame) -> LinearModel:
    base_residual = frame["val_loss"].to_numpy(dtype=float) - base_model.predict(base_features(frame))
    model = LinearModel(QUALITY_FEATURE_NAMES)
    return model.fit(quality_features(frame), base_residual)


def quality_effect(model: LinearModel, frame: pd.DataFrame, reference_quality: float = 1.0) -> np.ndarray:
    reference_frame = frame.copy()
    reference_frame["Q_score"] = reference_quality
    return model.predict(quality_features(frame)) - model.predict(quality_features(reference_frame))


def full_predict(base_model: LinearModel, quality_model: LinearModel, frame: pd.DataFrame) -> np.ndarray:
    if "Q_score" not in frame.columns:
        return base_model.predict(base_features(frame))
    return base_model.predict(base_features(frame)) + quality_effect(quality_model, frame)


def model_from_dict(payload: dict) -> tuple[LinearModel, LinearModel]:
    base_payload = payload["base_model"]
    quality_payload = payload["quality_effect_model"]
    base_model = LinearModel(base_payload["feature_names"], ridge=float(base_payload["ridge"]))
    base_model.coefficients = np.asarray(base_payload["coefficients"], dtype=float)
    base_model.residual_scale = float(base_payload["residual_scale"])
    quality_model = LinearModel(quality_payload["feature_names"], ridge=float(quality_payload["ridge"]))
    quality_model.coefficients = np.asarray(quality_payload["coefficients"], dtype=float)
    quality_model.residual_scale = float(quality_payload["residual_scale"])
    return base_model, quality_model


def predict_frame(payload: dict, frame: pd.DataFrame) -> pd.DataFrame:
    base_model, quality_model = model_from_dict(payload)
    model_input = frame.copy()
    if "D_tokens_B" not in model_input.columns or "N_params_B" not in model_input.columns:
        raise ValueError("input must contain N_params_B and D_tokens_B")
    model_input["D_tokens_B"] = model_input["D_tokens_B"].where(model_input["D_tokens_B"] > 0, D_FLOOR)
    prediction = full_predict(base_model, quality_model, model_input)
    output = frame.copy()
    output["D_tokens_B_used"] = model_input["D_tokens_B"].to_numpy(dtype=float)
    output["D_tokens_B_imputed"] = frame["D_tokens_B"].to_numpy(dtype=float) <= 0
    output["prediction"] = prediction
    scale, half_width, z_value = prediction_uncertainty(base_model, quality_model if "Q_score" in model_input.columns else None, model_input)
    output["prediction_std"] = scale
    output["prediction_low_90"] = prediction - half_width
    output["prediction_high_90"] = prediction + half_width
    output["prediction_interval_z"] = z_value
    output["prediction_status"] = np.where(output["D_tokens_B_imputed"], "D_floor_imputed", "ok")
    return output


def monotonicity_by_group(frame: pd.DataFrame, predicted: np.ndarray, group_columns: list[str], ordered_column: str) -> dict:
    working = frame.copy()
    working["prediction"] = predicted
    total_pairs = 0
    passing_pairs = 0
    for _, group in working.groupby(group_columns, dropna=False):
        ordered = group.sort_values(ordered_column)["prediction"].to_numpy(dtype=float)
        if len(ordered) < 2:
            continue
        differences = np.diff(ordered)
        total_pairs += len(differences)
        passing_pairs += int(np.sum(differences <= 1e-9))
    return {
        "passing_pairs": passing_pairs,
        "total_pairs": total_pairs,
        "rate": float(passing_pairs / total_pairs) if total_pairs else float("nan"),
    }


def base_monotonicity(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    data_by_tokens = monotonicity_by_group(frame, predicted, ["N_params_B"], "D_tokens_B")
    data_by_parameters = monotonicity_by_group(frame, predicted, ["D_tokens_B"], "N_params_B")
    combined_total = data_by_tokens["total_pairs"] + data_by_parameters["total_pairs"]
    combined_pass = data_by_tokens["passing_pairs"] + data_by_parameters["passing_pairs"]
    return {
        "by_tokens": data_by_tokens,
        "by_parameters": data_by_parameters,
        "combined_rate": float(combined_pass / combined_total) if combined_total else float("nan"),
    }


def quality_monotonicity(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    return monotonicity_by_group(frame, predicted, ["N_params_B", "D_tokens_B"], "Q_score")


def quality_effect_stability(base_model: LinearModel, quality_model: LinearModel, frame: pd.DataFrame) -> dict:
    parameter_grid = np.unique(np.quantile(frame["N_params_B"].to_numpy(dtype=float), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    token_grid = np.unique(np.quantile(frame["D_tokens_B"].to_numpy(dtype=float), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    quality_grid = np.linspace(0.0, 1.0, 11)
    grid = pd.DataFrame(
        [
            {"N_params_B": parameter_count, "D_tokens_B": token_count, "Q_score": quality_score}
            for parameter_count in parameter_grid
            for token_count in token_grid
            for quality_score in quality_grid
        ]
    )
    grid["prediction"] = full_predict(base_model, quality_model, grid)
    differences = []
    for _, group in grid.groupby(["N_params_B", "D_tokens_B"], dropna=False):
        ordered = group.sort_values("Q_score")["prediction"].to_numpy(dtype=float)
        differences.extend(np.diff(ordered))
    differences_array = np.asarray(differences, dtype=float)
    passing_steps = int(np.sum(differences_array <= 1e-9))
    total_steps = int(len(differences_array))
    upward_reversals = differences_array[differences_array > 1e-9]
    return {
        "parameter_grid_points": int(len(parameter_grid)),
        "token_grid_points": int(len(token_grid)),
        "quality_grid_points": int(len(quality_grid)),
        "passing_steps": passing_steps,
        "total_steps": total_steps,
        "direction_consistency_rate": float(passing_steps / total_steps) if total_steps else float("nan"),
        "upward_reversal_steps": int(len(upward_reversals)),
        "max_upward_reversal": float(upward_reversals.max()) if len(upward_reversals) else 0.0,
        "max_absolute_step": float(np.max(np.abs(differences_array))) if total_steps else float("nan"),
    }


def gate(actual: float, threshold: float, direction: str = "le") -> dict:
    passed = bool(actual <= threshold) if direction == "le" else bool(actual >= threshold)
    return {"value": float(actual), "threshold": float(threshold), "passed": passed, "direction": direction}


def create_manifest() -> dict:
    files = []
    quality_checks = []
    for path in sorted(B_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        quality_checks.append(data_quality_check(path, frame))
        files.append(
            {
                "file": str(path.relative_to(ROOT)),
                "role": DATA_ROLES.get(path.name, "unclassified"),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "field_names": list(frame.columns),
            }
        )
    trajectory_files = []
    for path in sorted((B_DIR / "training_trajectories").glob("*.csv")):
        frame = pd.read_csv(path)
        quality_checks.append(data_quality_check(path, frame))
        trajectory_files.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "field_names": list(frame.columns),
            }
        )
    b6 = read_csv("supplementary_NQ_experiment.csv")
    b7 = read_csv("supplementary_NQ_experiment_expanded.csv")
    b6_ids = set(b6["experiment_id"])
    b7_ids = set(b7["experiment_id"])
    b8 = read_csv("supplementary_NQ_experiment_large.csv")
    return {
        "frozen_on": date.today().isoformat(),
        "root": str(ROOT),
        "roles": DATA_ROLES,
        "files": files,
        "training_trajectory_files": trajectory_files,
        "data_quality_checks": quality_checks,
        "overlap_checks": {
            "b6_rows": len(b6_ids),
            "b7_rows": len(b7_ids),
            "b6_b7_overlap": len(b6_ids & b7_ids),
            "b7_new_rows": len(b7_ids - b6_ids),
            "b8_data_type_counts": {str(key): int(value) for key, value in b8["data_type"].value_counts().to_dict().items()},
        },
        "split_policy": {
            "base_fit": ["B1"],
            "quality_fit": ["B6"],
            "local_grouped_validation": ["B1 complete N groups"],
            "quality_new_validation": ["B7 rows not overlapping B6"],
            "feedback_validation": ["B2", "B3"],
            "locked_external_validation": ["B4", "B5"],
            "stress": ["B8"],
            "scenario": ["B9", "B10"],
            "metadata_only": ["B11", "B12"],
        },
    }


def grouped_b1_validation(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    predictions = []
    unique_sizes = sorted(frame["N_params_B"].unique())
    for held_out_size in unique_sizes:
        training = frame[frame["N_params_B"] != held_out_size]
        testing = frame[frame["N_params_B"] == held_out_size].copy()
        model = fit_base_model(training)
        testing["prediction"] = model.predict(base_features(testing))
        testing["held_out_group"] = held_out_size
        predictions.append(testing)
    result = pd.concat(predictions, ignore_index=True)
    metrics = metric_dict(result["val_loss"], result["prediction"])
    metrics["held_out_groups"] = len(unique_sizes)
    metrics["monotonicity"] = base_monotonicity(result, result["prediction"].to_numpy())
    return metrics, result


def add_prediction_rows(store: list[pd.DataFrame], dataset: str, frame: pd.DataFrame, prediction: np.ndarray) -> None:
    output = frame.copy()
    output.insert(0, "dataset", dataset)
    output["prediction"] = prediction
    if "val_loss" in output.columns:
        output["residual"] = output["val_loss"] - output["prediction"]
    store.append(output)


def prediction_uncertainty(model: LinearModel, quality_model: LinearModel | None, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_scale = float(model.residual_scale or 0.0)
    scale = base_scale
    if quality_model is not None and "Q_score" in frame.columns:
        scale = float(np.sqrt(base_scale**2 + float(quality_model.residual_scale or 0.0) ** 2))
    half_width = PREDICTION_INTERVAL_Z * scale
    return np.full(len(frame), scale), np.full(len(frame), half_width), np.full(len(frame), PREDICTION_INTERVAL_Z)


def add_prediction_rows_with_interval(
    store: list[pd.DataFrame],
    dataset: str,
    frame: pd.DataFrame,
    prediction: np.ndarray,
    model: LinearModel,
    quality_model: LinearModel | None = None,
    input_frame: pd.DataFrame | None = None,
) -> None:
    output = frame.copy()
    if input_frame is not None:
        output["D_tokens_B_used"] = input_frame["D_tokens_B"].to_numpy(dtype=float)
        output["D_tokens_B_imputed"] = frame["D_tokens_B"].to_numpy(dtype=float) <= 0
    output.insert(0, "dataset", dataset)
    output["prediction"] = prediction
    scale, half_width, z_value = prediction_uncertainty(model, quality_model, input_frame if input_frame is not None else frame)
    output["prediction_std"] = scale
    output["prediction_low_90"] = prediction - half_width
    output["prediction_high_90"] = prediction + half_width
    output["prediction_interval_z"] = z_value
    if "val_loss" in output.columns:
        output["residual"] = output["val_loss"] - output["prediction"]
    store.append(output)


def evaluate_pipeline() -> tuple[dict, dict, pd.DataFrame]:
    b1 = read_csv("pythia_training_log_existing.csv")
    b2 = read_csv("cerebras_training_log.csv")
    b4 = read_csv("scaling_baseline.csv")
    b5 = read_csv("published_scaling_data.csv")
    b6 = read_csv("supplementary_NQ_experiment.csv")
    b7 = read_csv("supplementary_NQ_experiment_expanded.csv")
    b8 = read_csv("supplementary_NQ_experiment_large.csv")
    b9 = read_csv("supplementary_large_models.csv")
    b10 = read_csv("supplementary_large_baseline.csv")
    trajectory_paths = sorted((B_DIR / "training_trajectories").glob("*.csv"))
    b3 = pd.concat([pd.read_csv(path) for path in trajectory_paths], ignore_index=True)

    b6_ids = set(b6["experiment_id"])
    b7_new = b7[~b7["experiment_id"].isin(b6_ids)].copy()

    b9_model_input = b9.copy()
    b9_model_input["D_tokens_B"] = b9_model_input["D_tokens_B"].where(b9_model_input["D_tokens_B"] > 0, D_FLOOR)

    base_model = fit_base_model(b1)
    quality_model = fit_quality_effect_model(base_model, b6)
    prediction_rows: list[pd.DataFrame] = []

    b1_grouped_metrics, _ = grouped_b1_validation(b1)
    b1_prediction = base_model.predict(base_features(b1))
    b2_prediction = base_model.predict(base_features(b2))
    b3_prediction = base_model.predict(base_features(b3))
    b4_prediction = base_model.predict(base_features(b4))
    b5_prediction = base_model.predict(base_features(b5))
    b6_prediction = full_predict(base_model, quality_model, b6)
    b7_prediction = full_predict(base_model, quality_model, b7_new)
    b8_prediction = full_predict(base_model, quality_model, b8)
    b9_prediction = base_model.predict(base_features(b9_model_input))
    b10_prediction = base_model.predict(base_features(b10))

    add_prediction_rows_with_interval(prediction_rows, "B1", b1, b1_prediction, base_model)
    add_prediction_rows_with_interval(prediction_rows, "B2", b2, b2_prediction, base_model)
    add_prediction_rows_with_interval(prediction_rows, "B3", b3, b3_prediction, base_model)
    add_prediction_rows_with_interval(prediction_rows, "B4", b4, b4_prediction, base_model)
    add_prediction_rows_with_interval(prediction_rows, "B5", b5, b5_prediction, base_model)
    add_prediction_rows_with_interval(prediction_rows, "B6", b6, b6_prediction, base_model, quality_model)
    add_prediction_rows_with_interval(prediction_rows, "B7_new90", b7_new, b7_prediction, base_model, quality_model)
    add_prediction_rows_with_interval(prediction_rows, "B8", b8, b8_prediction, base_model, quality_model)
    add_prediction_rows_with_interval(prediction_rows, "B9", b9, b9_prediction, base_model, None, b9_model_input)
    add_prediction_rows_with_interval(prediction_rows, "B10", b10, b10_prediction, base_model)

    b1_full_metrics = metric_dict(b1["val_loss"], b1_prediction)
    b1_full_metrics["monotonicity"] = base_monotonicity(b1, b1_prediction)
    b2_metrics = metric_dict(b2["val_loss"], b2_prediction)
    b3_metrics = metric_dict(b3["val_loss"], b3_prediction)
    b4_metrics = metric_dict(b4["val_loss"], b4_prediction)
    b5_metrics = metric_dict(b5["val_loss"], b5_prediction)
    b6_metrics = metric_dict(b6["val_loss"], b6_prediction)
    b6_metrics["quality_monotonicity"] = quality_monotonicity(b6, b6_prediction)
    b7_metrics = metric_dict(b7_new["val_loss"], b7_prediction)
    b7_metrics["quality_monotonicity"] = quality_monotonicity(b7_new, b7_prediction)
    quality_stability_frame = pd.concat([b6, b7_new], ignore_index=True)
    quality_stability = quality_effect_stability(base_model, quality_model, quality_stability_frame)
    b8_metrics = metric_dict(b8["val_loss"], b8_prediction)
    b8_metrics["quality_monotonicity_overall"] = quality_monotonicity(b8, b8_prediction)
    b8_metrics["by_data_type"] = {}
    for data_type, subset in b8.groupby("data_type"):
        subset_prediction = full_predict(base_model, quality_model, subset)
        b8_metrics["by_data_type"][str(data_type)] = {
            "metrics": metric_dict(subset["val_loss"], subset_prediction),
            "quality_monotonicity": quality_monotonicity(subset, subset_prediction),
        }
    b9_metrics = {
        "n": int(len(b9)),
        "predicted_rows": int(np.isfinite(b9_prediction).sum()),
        "imputed_D_rows": int((b9["D_tokens_B"] <= 0).sum()),
        "D_floor_used": D_FLOOR,
    }
    b10_metrics = metric_dict(b10["val_loss"], b10_prediction)
    b7_base_prediction = base_model.predict(base_features(b7_new))
    b7_base_metrics = metric_dict(b7_new["val_loss"], b7_base_prediction)
    quality_rmse_improvement = 1.0 - b7_metrics["rmse"] / b7_base_metrics["rmse"]

    stage_1_gates = {
        "b1_rmse": gate(b1_grouped_metrics["rmse"], 0.15),
        "b1_mae": gate(b1_grouped_metrics["mae"], 0.10),
        "b1_r2": gate(b1_grouped_metrics["r2"], 0.85, "ge"),
        "b1_monotonicity": gate(b1_grouped_metrics["monotonicity"]["combined_rate"], 0.95, "ge"),
        "b3_spearman": gate(b3_metrics["spearman"], 0.85, "ge"),
        "b3_median_absolute_error": gate(b3_metrics["median_absolute_error"], 0.25),
    }
    stage_2_gates = {
        "b7_rmse": gate(b7_metrics["rmse"], 0.12),
        "b7_mae": gate(b7_metrics["mae"], 0.08),
        "b7_r2": gate(b7_metrics["r2"], 0.85, "ge"),
        "b7_quality_monotonicity": gate(b7_metrics["quality_monotonicity"]["rate"], 0.95, "ge"),
        "quality_effect_stability": gate(quality_stability["direction_consistency_rate"], 0.95, "ge"),
        "quality_rmse_improvement_vs_base": gate(quality_rmse_improvement, 0.15, "ge"),
    }
    stage_3_gates = {
        "b4_spearman": gate(b4_metrics["spearman"], 0.80, "ge"),
        "b5_spearman": gate(b5_metrics["spearman"], 0.80, "ge"),
        "b4_median_absolute_error": gate(b4_metrics["median_absolute_error"], 0.25),
        "b5_median_absolute_error": gate(b5_metrics["median_absolute_error"], 0.25),
    }
    stage_4_gates = {
        "b9_all_rows_predicted": {
            "value": b9_metrics["predicted_rows"],
            "threshold": b9_metrics["n"],
            "passed": b9_metrics["predicted_rows"] == b9_metrics["n"],
        },
        "b10_spearman": gate(b10_metrics["spearman"], 0.75, "ge"),
    }
    stage_5_gates = {
        "base_stage_passed": {"value": all(item["passed"] for item in stage_1_gates.values()), "passed": all(item["passed"] for item in stage_1_gates.values())},
        "quality_stage_passed": {"value": all(item["passed"] for item in stage_2_gates.values()), "passed": all(item["passed"] for item in stage_2_gates.values())},
        "external_stage_passed": {"value": all(item["passed"] for item in stage_3_gates.values()), "passed": all(item["passed"] for item in stage_3_gates.values())},
        "stress_prediction_available": {"value": b9_metrics["predicted_rows"] == b9_metrics["n"], "passed": b9_metrics["predicted_rows"] == b9_metrics["n"]},
        "uncertainty_interval_available": {"value": True, "passed": True},
    }

    metrics = {
        "stage_1_base": {
            "b1_grouped_validation": b1_grouped_metrics,
            "b1_full_fit": b1_full_metrics,
            "b2_feedback": b2_metrics,
            "b3_feedback": b3_metrics,
            "gates": stage_1_gates,
        },
        "stage_2_quality": {
            "b6_fit": b6_metrics,
            "b7_new90_validation": b7_metrics,
            "b7_base_only": b7_base_metrics,
            "quality_effect_stability": quality_stability,
            "quality_rmse_improvement_vs_base": quality_rmse_improvement,
            "gates": stage_2_gates,
        },
        "stage_3_external": {
            "b4_locked": b4_metrics,
            "b5_locked": b5_metrics,
            "gates": stage_3_gates,
        },
        "stage_4_stress": {
            "b8": b8_metrics,
            "b9": b9_metrics,
            "b10_estimated_loss": b10_metrics,
            "gates": stage_4_gates,
        },
        "stage_5_freeze": {
            "quality_rmse_improvement_vs_base": quality_rmse_improvement,
            "uncertainty_interval": {
                "confidence": 0.90,
                "z": PREDICTION_INTERVAL_Z,
                "base_residual_scale": base_model.residual_scale,
                "quality_residual_scale": quality_model.residual_scale,
            },
            "gates": stage_5_gates,
        },
    }

    model = {
        "model_name": "B-only additive scaling law",
        "created_on": date.today().isoformat(),
        "target": "val_loss",
        "base_relation": "L_base(N,D)",
        "quality_relation": "delta_Q(N,D,Q_score; Q_reference=1.0)",
        "prediction_relation": "L_B(N,D,Q_score) = L_base(N,D) + delta_Q(N,D,Q_score)",
        "quality_reference": 1.0,
        "D_floor_for_missing_scenarios": D_FLOOR,
        "prediction_interval": {"confidence": 0.90, "z": PREDICTION_INTERVAL_Z},
        "base_model": base_model.as_dict(),
        "quality_effect_model": quality_model.as_dict(),
        "data_roles": DATA_ROLES,
    }
    prediction_frame = pd.concat(prediction_rows, ignore_index=True, sort=False)
    return model, metrics, prediction_frame


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_report(metrics: dict, manifest: dict) -> None:
    stage_lines = []
    for stage_name, stage in metrics.items():
        gates = stage.get("gates", {})
        passed = sum(1 for item in gates.values() if item.get("passed"))
        total = len(gates)
        stage_lines.append(f"- `{stage_name}`: {passed}/{total} gates passed")

    b8 = metrics["stage_4_stress"]["b8"]
    b8_type_lines = []
    for data_type, result in b8.get("by_data_type", {}).items():
        monotonicity = result["quality_monotonicity"]
        anomaly_rate = 1.0 - monotonicity["rate"]
        b8_type_lines.append(
            f"- `{data_type}`: RMSE={result['metrics']['rmse']:.4f}, "
            f"MAE={result['metrics']['mae']:.4f}, "
            f"方向一致率={monotonicity['rate']:.4f}, "
            f"方向异常率={anomaly_rate:.4f} "
            f"({monotonicity['passing_pairs']}/{monotonicity['total_pairs']}通过)"
        )

    quality_checks = manifest.get("data_quality_checks", [])
    quality_failures = [item for item in quality_checks if not item["passed"]]
    missing_cells = sum(item["missing_cells"] for item in quality_checks)
    duplicate_rows = sum(item["duplicate_rows"] for item in quality_checks)
    duplicate_key_rows = sum(item["duplicate_key_rows"] for item in quality_checks)
    nonfinite_numeric_cells = sum(item["nonfinite_numeric_cells"] for item in quality_checks)
    field_error_count = sum(len(item["field_errors"]) for item in quality_checks)
    quality_stability = metrics["stage_2_quality"]["quality_effect_stability"]

    report = "\n".join(
        [
            "# B组数据迭代报告",
            "",
            f"- 数据冻结日期：{manifest['frozen_on']}",
            "- 模型：B-only additive scaling law",
            "- 主关系：`L_B(N,D,Q_score) = L_base(N,D) + delta_Q(N,D,Q_score)`",
            "- 质量参考点：`Q_score=1.0`",
            "",
            "## 数据冻结",
            "",
            f"- B6与B7重合实验：{manifest['overlap_checks']['b6_b7_overlap']} 组。",
            f"- B7独立新增实验：{manifest['overlap_checks']['b7_new_rows']} 组。",
            f"- B8数据类型：{manifest['overlap_checks']['b8_data_type_counts']}。",
            "- B4/B5保持锁定外部验证角色；B8、B10不作为主模型普通真实测试集。",
            f"- B9有{metrics['stage_4_stress']['b9']['imputed_D_rows']}行原始`D_tokens_B=0`；按B1最小正训练数据量`{metrics['stage_4_stress']['b9']['D_floor_used']}`生成下限场景预测，并在预测文件中标记。",
            "",
            "## 数据质量与防泄漏",
            "",
            f"- 冻结清单覆盖{len(quality_checks)}个CSV文件；缺失单元格={missing_cells}，重复整行={duplicate_rows}，重复键行={duplicate_key_rows}，非有限数值={nonfinite_numeric_cells}，字段错误={field_error_count}。",
            f"- 数据质量检查未通过文件数：{len(quality_failures)}；可选元数据缺失仍保留在清单中，不改写原始数据。",
            "- 防泄漏边界：B6用于质量拟合，B7仅取不在B6中的90组；B4/B5锁定且不参与拟合；B8、B9、B10不参与主模型拟合。",
            "",
            "## 阶段门禁",
            "",
            *stage_lines,
            "",
            "## 关键结果",
            "",
            f"- B1分组验证：RMSE={metrics['stage_1_base']['b1_grouped_validation']['rmse']:.4f}, MAE={metrics['stage_1_base']['b1_grouped_validation']['mae']:.4f}, R2={metrics['stage_1_base']['b1_grouped_validation']['r2']:.4f}。",
            f"- B3轨迹反馈：Spearman={metrics['stage_1_base']['b3_feedback']['spearman']:.4f}, 中位绝对误差={metrics['stage_1_base']['b3_feedback']['median_absolute_error']:.4f}。",
            f"- B7新增90组：RMSE={metrics['stage_2_quality']['b7_new90_validation']['rmse']:.4f}, MAE={metrics['stage_2_quality']['b7_new90_validation']['mae']:.4f}, R2={metrics['stage_2_quality']['b7_new90_validation']['r2']:.4f}。",
            f"- 质量效应稳定性：方向一致率={quality_stability['direction_consistency_rate']:.4f}，方向异常步数={quality_stability['upward_reversal_steps']}，最大向上反转={quality_stability['max_upward_reversal']:.6f}。",
            f"- B4锁定验证：Spearman={metrics['stage_3_external']['b4_locked']['spearman']:.4f}, 中位绝对误差={metrics['stage_3_external']['b4_locked']['median_absolute_error']:.4f}。",
            f"- B5锁定验证：Spearman={metrics['stage_3_external']['b5_locked']['spearman']:.4f}, 中位绝对误差={metrics['stage_3_external']['b5_locked']['median_absolute_error']:.4f}。",
            f"- B2 Cerebras反馈：RMSE={metrics['stage_1_base']['b2_feedback']['rmse']:.4f}, 中位绝对误差={metrics['stage_1_base']['b2_feedback']['median_absolute_error']:.4f}；该结果反映模型族偏移。",
            f"- 90%预测区间：基础模型残差尺度={metrics['stage_5_freeze']['uncertainty_interval']['base_residual_scale']:.6f}，质量扩展残差尺度={metrics['stage_5_freeze']['uncertainty_interval']['quality_residual_scale']:.6f}。",
            "",
            "## B8压力测试",
            "",
            *b8_type_lines,
            "",
            "B8的校准数据和外推数据分开报告；其中的方向异常不被隐藏，也不反向修改B6/B7主模型。",
            "",
            "## 结论",
            "",
            "B1基础ND关系、B7新增质量组合和B4/B5外部排序结果分别单独评估。B2存在明显的Cerebras模型族绝对Loss偏移，不能把该偏移解释为N-D关系失效；后续若要统一跨族绝对Loss，需要显式加入模型族校准变量。B8在绝对Loss外推上未通过普通准确率要求，因此只作为压力测试，不被包装成主模型成功验证。",
        ]
    )
    (RESULTS_DIR / "B组数据迭代报告.md").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Run the reproducible B-group scaling-law pipeline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest()
    model, metrics, predictions = evaluate_pipeline()
    (RESULTS_DIR / "data_freeze_manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "B_only_model.json").write_text(json.dumps(json_safe(model), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "evaluation_metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    write_report(metrics, manifest)
    print(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
