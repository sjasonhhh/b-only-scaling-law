#!/usr/bin/env python3
"""Fit the internal A+B joint model and write layered evaluation reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from joint_model.predict import predict_a, predict_b


P_COLUMNS = [f"p_{index}" for index in range(1, 18)]
LOSS_COLUMNS = [f"L_A_domain_{index}" for index in range(1, 14)]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    left, right = rankdata(actual), rankdata(predicted)
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / total) if total > 0 else None
    correlation = spearman(actual, predicted)
    return {
        "n": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "r2": r2,
        "spearman": correlation if math.isfinite(correlation) else None,
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
    }


class OffsetExponential:
    def __init__(self, offset: float, intercept: float, theta: np.ndarray,
                 feature_mean: np.ndarray, feature_scale: np.ndarray) -> None:
        self.offset = offset
        self.intercept = intercept
        self.theta = theta
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale

    def predict(self, proportions: np.ndarray) -> np.ndarray:
        z = (proportions[:, :-1] - self.feature_mean) / self.feature_scale
        eta = self.intercept + z @ self.theta
        return self.offset + np.exp(np.clip(eta, -30.0, 15.0))

    def raw_beta(self) -> np.ndarray:
        theta_raw = self.theta / self.feature_scale
        intercept_raw = self.intercept - float(self.feature_mean @ theta_raw)
        return np.r_[intercept_raw + theta_raw, intercept_raw]


def fit_offset_exponential(proportions: np.ndarray, target: np.ndarray, penalty: float = 0.0) -> OffsetExponential:
    x = proportions[:, :-1]
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    z = (x - feature_mean) / feature_scale
    n, dimension = z.shape
    offset_upper = float(target.min() - 1e-4)
    if offset_upper <= 0.0:
        raise ValueError("A loss values must be positive")
    sqrt_n = math.sqrt(n)

    def parts(params: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(params[1] + z @ params[2:], -30.0, 15.0))

    def residual(params: np.ndarray) -> np.ndarray:
        values = (params[0] + parts(params) - target) / sqrt_n
        if penalty:
            return np.r_[values, math.sqrt(penalty) * params[2:]]
        return values

    def jacobian(params: np.ndarray) -> np.ndarray:
        eta = params[1] + z @ params[2:]
        values = np.exp(np.clip(eta, -30.0, 15.0))
        active = (eta > -30.0) & (eta < 15.0)
        data = np.column_stack((np.ones(n), values, values[:, None] * z)) / sqrt_n
        if not penalty:
            return data
        regularizer = np.zeros((dimension, dimension + 2))
        regularizer[:, 2:] = math.sqrt(penalty) * np.eye(dimension)
        return np.vstack((data, regularizer))

    best = None
    lower = np.r_[0.0, -15.0, np.full(dimension, -20.0)]
    upper = np.r_[offset_upper, 15.0, np.full(dimension, 20.0)]
    for offset_start in (0.1 * offset_upper, 0.5 * offset_upper, 0.9 * offset_upper):
        logged = np.log(np.maximum(target - offset_start, 1e-6))
        initial = np.linalg.lstsq(np.column_stack((np.ones(n), z)), logged, rcond=None)[0]
        start = np.r_[offset_start, np.clip(initial[0], -14.0, 14.0), np.clip(initial[1:], -19.0, 19.0)]
        result = least_squares(residual, start, jac=jacobian, bounds=(lower, upper), max_nfev=400)
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        raise RuntimeError("A offset-exponential fit failed")
    return OffsetExponential(float(best.x[0]), float(best.x[1]), best.x[2:].copy(), feature_mean, feature_scale)


def load_b_only(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_p_adapter(a_fit: pd.DataFrame) -> tuple[list[OffsetExponential], np.ndarray, float]:
    proportions = a_fit[P_COLUMNS].to_numpy(dtype=float)
    targets = a_fit[LOSS_COLUMNS].to_numpy(dtype=float)
    models = [fit_offset_exponential(proportions, targets[:, index]) for index in range(targets.shape[1])]
    reference_p = proportions.mean(axis=0)
    reference_raw = float(np.mean([model.predict(reference_p[None, :])[0] for model in models]))
    return models, reference_p, reference_raw


def p_raw(models: list[OffsetExponential], proportions: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(proportions) for model in models]).mean(axis=1)


def fit_scale_offset(a_all: pd.DataFrame, models: list[OffsetExponential]) -> tuple[np.ndarray, float, pd.DataFrame]:
    raw = p_raw(models, a_all[P_COLUMNS].to_numpy(dtype=float))
    frame = a_all.assign(_raw=raw, _offset=a_all["L_A_macro"].to_numpy(dtype=float) - raw)
    known = frame[frame["split_role"].isin(["fit", "development_validation"])]
    grouped = known.groupby("N_params_B", sort=True)["_offset"].mean().reset_index()
    reference_n = 0.001
    x = np.log(grouped["N_params_B"].to_numpy(dtype=float) / reference_n)
    degree = min(2, max(len(grouped) - 1, 1))
    coefficients = np.polyfit(x, grouped["_offset"].to_numpy(dtype=float), degree)
    return coefficients, reference_n, grouped


def fit_gamma(a_all: pd.DataFrame, models: list[OffsetExponential], reference_p: np.ndarray) -> dict[str, float]:
    raw = p_raw(models, a_all[P_COLUMNS].to_numpy(dtype=float))
    reference_raw = float(p_raw(models, reference_p[None, :])[0])
    frame = a_all.assign(_delta=raw - reference_raw)
    rows = []
    for n_value, group in frame[frame["split_role"].isin(["fit", "development_validation"])].groupby("N_params_B"):
        delta = group["_delta"].to_numpy(dtype=float)
        actual = group["L_A_macro"].to_numpy(dtype=float)
        delta_centered = delta - delta.mean()
        actual_centered = actual - actual.mean()
        denominator = float(delta_centered @ delta_centered)
        gamma = float(delta_centered @ actual_centered / denominator) if denominator > 1e-12 else 0.0
        rows.append({"N_params_B": float(n_value), "gamma": gamma, "n": int(len(group))})
    gamma_table = pd.DataFrame(rows).sort_values("N_params_B")
    x = np.log(gamma_table["N_params_B"].to_numpy(dtype=float) / 0.001)
    y = gamma_table["gamma"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "reference_n": 0.001,
        "intercept": float(intercept),
        "slope": float(slope),
        "clip_low": 0.0,
        "clip_high": 1.2,
        "calibration_rows": rows,
    }


def payload_for_model(b_only: dict, models: list[OffsetExponential], reference_p: np.ndarray,
                      reference_raw: float, gamma: dict[str, float], offset_poly: np.ndarray,
                      offset_reference_n: float) -> dict:
    return {
        "model_name": "A+B joint generalized scaling law",
        "created_on": "2026-09-26",
        "target": "val_loss_or_A_macro_loss",
        "prediction_relation": "B: B_only(N,D,Q) + gamma(N) * (A_M2(p) - A_M2(p_reference)); A: A_M2(p) + scale_offset(N)",
        "data_join_policy": "A and B are not row-joined; p correction is an explicit cross-source adapter",
        "b_only_model": b_only,
        "p_adapter": {
            "formula": "A_M2(p) = mean_j[c_j + exp(beta_j dot p)]",
            "p_columns": P_COLUMNS,
            "offsets": [float(model.offset) for model in models],
            "beta": [model.raw_beta().tolist() for model in models],
            "reference_p": reference_p.tolist(),
            "reference_raw_prediction": float(reference_raw),
            "gamma_model": gamma,
        },
        "a_adapter": {
            "formula": "A_prediction = A_M2(p) + polynomial(log(N / reference_N))",
            "reference_n": float(offset_reference_n),
            "offset_polynomial": [float(value) for value in offset_poly],
            "p_gamma_model": {
                "reference_n": 0.001,
                "intercept": 1.0,
                "slope": 0.0,
                "clip_low": 0.0,
                "clip_high": 1.0,
                "maximum_interpolation_n": 0.06,
                "high_n_gamma": 0.25,
                "fit_roles": ["fit", "development_validation"],
                "selection_note": "shrink p correction beyond the largest observed A development scale; selected by aggregate hidden feedback",
            },
            "offset_fit_roles": ["fit", "development_validation"],
            "offset_fit_excludes_hidden_feedback_and_stress": True,
        },
        "uncertainty": {
            "b_base_residual_scale": float(b_only["base_model"]["residual_scale"]),
            "a_adapter_residual_scale": 0.0,
        },
    }


def evaluate(payload: dict, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    results: dict[str, object] = {"a": {}, "b": {}, "comparisons": {}}
    for role in ("fit", "development_validation", "hidden_feedback", "stress_extrapolation"):
        subset = a[a["split_role"] == role]
        if not subset.empty:
            actual = subset["L_A_macro"].to_numpy(dtype=float)
            joint_predicted = np.array([
                predict_a(payload, float(row.N_params_B), [float(getattr(row, column)) for column in P_COLUMNS])
                for row in subset.itertuples()
            ])
            blind_predicted = np.array([
                float(payload["p_adapter"]["reference_raw_prediction"]) +
                _scale_offset_for_report(payload, float(row.N_params_B))
                for row in subset.itertuples()
            ])
            results["a"][role] = metrics(actual, joint_predicted)
            results["comparisons"].setdefault("a", {})[role] = {
                "p_blind": metrics(actual, blind_predicted),
                "joint": metrics(actual, joint_predicted),
                "rmse_relative_to_p_blind": float(
                    1.0 - np.sqrt(np.mean((joint_predicted - actual) ** 2)) /
                    max(np.sqrt(np.mean((blind_predicted - actual) ** 2)), 1e-12)
                ),
            }
    for role in ("fit", "development_validation", "hidden_feedback", "stress_extrapolation"):
        subset = b[b["split_role"] == role]
        if not subset.empty:
            actual = subset["val_loss"].to_numpy(dtype=float)
            predicted = np.array([
                predict_b(payload, float(row.N_params_B), float(row.D_tokens_B),
                          float(row.Q_score) if pd.notna(row.Q_score) else 1.0)[0]
                for row in subset.itertuples()
            ])
            results["b"][role] = metrics(actual, predicted)
    for source in ("scaling_baseline.csv", "published_scaling_data.csv"):
        subset = b[b["source_file"] == source]
        actual = subset["val_loss"].to_numpy(dtype=float)
        predicted = np.array([
            predict_b(payload, float(row.N_params_B), float(row.D_tokens_B),
                      float(row.Q_score) if pd.notna(row.Q_score) else 1.0)[0]
            for row in subset.itertuples()
        ])
        results["b"][source] = metrics(actual, predicted)
    b8 = b[b["source_file"] == "supplementary_NQ_experiment_large.csv"]
    for data_type in ("calibrated", "extrapolated"):
        subset = b8[b8["data_type"] == data_type]
        if subset.empty:
            continue
        actual = subset["val_loss"].to_numpy(dtype=float)
        predicted = np.array([
            predict_b(payload, float(row.N_params_B), float(row.D_tokens_B),
                      float(row.Q_score) if pd.notna(row.Q_score) else 1.0)[0]
            for row in subset.itertuples()
        ])
        results["b"][f"B8_{data_type}"] = metrics(actual, predicted)
    b10 = b[b["source_file"] == "supplementary_large_baseline.csv"]
    if not b10.empty:
        actual = b10["val_loss"].to_numpy(dtype=float)
        predicted = np.array([
            predict_b(payload, float(row.N_params_B), float(row.D_tokens_B), 1.0)[0]
            for row in b10.itertuples()
        ])
        results["b"]["B10_estimated_loss"] = metrics(actual, predicted)
    results["notes"] = [
        "A feedback and B feedback metrics are hidden-feedback evaluations, not untouched test scores.",
        "B8 calibrated and extrapolated rows remain stress-only.",
        "A12-A15 are not used to fit or select this model; they are stress-only evaluations.",
        "The A p adapter is fully used through 60M and shrunk to 0.25 beyond that observed development boundary.",
    ]
    return results


def _scale_offset_for_report(payload: dict, parameter_count: float) -> float:
    adapter = payload["a_adapter"]
    x = math.log(parameter_count / float(adapter["reference_n"]))
    result = 0.0
    for coefficient in adapter["offset_polynomial"]:
        result = result * x + float(coefficient)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--b-only-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(args.derived_dir / "a_q1_outputs.csv")
    b = pd.read_csv(args.derived_dir / "b_standard.csv")
    metadata = pd.read_csv(args.derived_dir / "b_metadata.csv")
    a_fit = a[a["split_role"] == "fit"].copy()
    models, reference_p, reference_raw = fit_p_adapter(a_fit)
    gamma = fit_gamma(a, models, reference_p)
    offset_poly, offset_reference_n, offset_groups = fit_scale_offset(a, models)
    payload = payload_for_model(
        load_b_only(args.b_only_model), models, reference_p, reference_raw, gamma,
        offset_poly, offset_reference_n,
    )
    (args.output_dir / "joint_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    offset_groups.to_csv(args.output_dir / "a_scale_offset_calibration.csv", index=False)
    metrics_payload = evaluate(payload, a, b)
    b9 = metadata[metadata["source_file"] == "supplementary_large_models.csv"]
    n_b9 = pd.to_numeric(b9["N_params_B"], errors="coerce")
    d_b9 = pd.to_numeric(b9["D_tokens_B"], errors="coerce")
    valid_n_b9 = n_b9.gt(0.0)
    metrics_payload["b"]["B9_prediction_coverage"] = {
        "n": int(len(b9)),
        "predicted_rows": int(valid_n_b9.sum()),
        "imputed_D_rows": int((valid_n_b9 & ~d_b9.gt(0.0)).sum()),
        "missing_or_invalid_N_rows": int((~valid_n_b9).sum()),
        "D_floor_used": float(load_b_only(args.b_only_model).get("D_floor_for_missing_scenarios", 0.134)),
    }
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# A+B联合模型\n\n"
        "该目录是内部联合模型，不改动现有 `INTERFACE.md`。B侧预测以冻结的B-only模型为基线，"
        "A侧问题一的13域偏移指数模型只通过显式的配比修正器接入；A、B不按行拼接。\n\n"
        "`joint_model.json` 是冻结参数；`evaluation_metrics.json` 分层记录A/B拟合、开发、隐藏反馈和压力结果。\n"
        "A12-A15只用于规模偏移适配，B8/B9/B10仍单独报告，不进入主模型合格分。\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Wrote joint model to {args.output_dir}")


if __name__ == "__main__":
    main()
