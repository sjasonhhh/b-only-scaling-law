#!/usr/bin/env python3
"""Evaluate a submission without exposing private row-level labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from joint_model.predict import P_COLUMNS, predict_a, predict_b


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(actual: list[float], predicted: list[float]) -> float:
    left, right = rankdata(actual), rankdata(predicted)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value**2 for value in left_centered) * sum(value**2 for value in right_centered))
    return sum(left_value * right_value for left_value, right_value in zip(left_centered, right_centered)) / denominator if denominator else 0.0


def group_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - target for prediction, target in zip(predicted, actual)]
    actual_mean = sum(actual) / len(actual)
    total = sum((target - actual_mean) ** 2 for target in actual)
    ordered = sorted(abs(error) for error in errors)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    mae = sum(abs(error) for error in errors) / len(errors)
    rho = spearman(actual, predicted)
    score = 0.5 / (1.0 + mae) + 0.5 * max(0.0, min(1.0, (rho + 1.0) / 2.0))
    return {
        "n": len(actual),
        "rmse": math.sqrt(sum(error**2 for error in errors) / len(errors)),
        "mae": mae,
        "median_absolute_error": median,
        "r2": 1.0 - sum(error**2 for error in errors) / total if total else 0.0,
        "spearman": rho,
        "score": score,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate_a(path: Path, payload: dict) -> dict[str, float]:
    actual: list[float] = []
    predicted: list[float] = []
    for row in read_rows(path):
        actual.append(float(row["L_A_macro"]))
        proportions = [float(row[column]) for column in P_COLUMNS]
        predicted.append(predict_a(payload, float(row["N_params_B"]), proportions))
    return group_metrics(actual, predicted)


def evaluate_b(path: Path, payload: dict) -> dict[str, float]:
    actual: list[float] = []
    predicted: list[float] = []
    for row in read_rows(path):
        actual.append(float(row["val_loss"]))
        quality = float(row["Q_score"]) if row.get("Q_score", "").strip() else 1.0
        predicted.append(predict_b(payload, float(row["N_params_B"]), float(row["D_tokens_B"]), quality)[0])
    return group_metrics(actual, predicted)


def evaluate_b9_coverage(path: Path, _payload: dict | None = None) -> dict[str, float]:
    rows = read_rows(path)
    valid_n = 0
    imputed_d = 0
    for row in rows:
        try:
            n_value = float(row["N_params_B"])
            d_value = float(row["D_tokens_B"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(n_value) and n_value > 0.0:
            valid_n += 1
            if not math.isfinite(d_value) or d_value <= 0.0:
                imputed_d += 1
    return {
        "n": len(rows),
        "predicted_rows": valid_n,
        "imputed_D_rows": imputed_d,
        "missing_or_invalid_N_rows": len(rows) - valid_n,
        "D_floor_used": 0.134,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--submission-id", required=True)
    args = parser.parse_args()
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, float]] = {}
    for name, filename, evaluator in (
        ("A_hidden_feedback", "a_feedback.csv", evaluate_a),
        ("B_scaling_baseline", "b_scaling_baseline.csv", evaluate_b),
        ("B_published_scaling", "b_published_scaling.csv", evaluate_b),
    ):
        path = args.labels_dir / filename
        if path.is_file():
            groups[name] = evaluator(path, payload)
    stress: dict[str, dict[str, float]] = {}
    for name, filename, evaluator in (
        ("A_stress", "a_stress.csv", evaluate_a),
        ("B8_calibrated", "b8_calibrated.csv", evaluate_b),
        ("B8_extrapolated", "b8_extrapolated.csv", evaluate_b),
        ("B10_estimated_loss", "b10_estimated_loss.csv", evaluate_b),
        ("B9_prediction_coverage", "b9_large_models_input.csv", evaluate_b9_coverage),
    ):
        path = args.labels_dir / filename
        if path.is_file():
            stress[name] = evaluator(path, payload)
    primary = list(groups.values())
    overall = sum(item["score"] for item in primary) / len(primary) if primary else 0.0
    gate_status = "pass" if (
        overall >= 0.90 and
        groups.get("A_hidden_feedback", {}).get("mae", float("inf")) <= 0.10 and
        groups.get("B_scaling_baseline", {}).get("spearman", -1.0) >= 0.80 and
        groups.get("B_published_scaling", {}).get("spearman", -1.0) >= 0.80
    ) else "review"
    feedback = []
    if groups.get("A_hidden_feedback", {}).get("mae", 0.0) > 0.25:
        feedback.append("A隐藏反馈的绝对标定误差仍偏高，应优先检查规模偏移而不是逐行调参。")
    if groups.get("B_scaling_baseline", {}).get("spearman", 1.0) < 0.80:
        feedback.append("B跨族排序相关性未达0.80，应检查模型族偏移。")
    if groups.get("B_published_scaling", {}).get("spearman", 1.0) < 0.80:
        feedback.append("B文献排序相关性未达0.80，应保留来源分层，不要与Pythia日志混合。")
    if not feedback:
        feedback.append("主评测组没有触发预设诊断；继续保留当前版本并检查压力组。")
    result = {
        "submission_id": args.submission_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(args.model),
        "overall_score": overall,
        "gate_status": gate_status,
        "group_scores": groups,
        "stress_scores": stress,
        "feedback_messages": feedback,
        "label_policy": "private labels are not included in this output",
    }
    args.history_dir.mkdir(parents=True, exist_ok=True)
    output = args.history_dir / f"{args.submission_id}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
