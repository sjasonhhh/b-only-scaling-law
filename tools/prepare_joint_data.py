#!/usr/bin/env python3
"""Build auditable A/B tables for the joint scaling-law experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


MIXTURE_PREFIX = "train_the_pile_"
LOSS_PREFIX = "metric/the_pile_"
P_COLUMNS = [f"p_{index}" for index in range(1, 18)]

A_PAIRS = (
    ("A4_A5", "train_mixture_1m.csv", "train_pile_loss_1m.csv", "fit", "1m", 0.001, False),
    ("A6_A7", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "development_validation", "1m", 0.001, False),
    ("A8_A9", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "development_validation", "60m", 0.060, False),
    ("A10_A11", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "hidden_feedback", "1B", 1.0, False),
    ("A12_A13", "est_mixture_10b.csv", "est_pile_loss_10b.csv", "stress_extrapolation", "10B", 10.0, True),
    ("A14_A15", "est_mixture_70b.csv", "est_pile_loss_70b.csv", "stress_extrapolation", "70B", 70.0, True),
)

B_ROLE = {
    "pythia_training_log_existing.csv": ("fit", "真实", False, False),
    "cerebras_training_log.csv": ("development_validation", "半合成", True, False),
    "scaling_baseline.csv": ("hidden_feedback", "真实公开观测", False, False),
    "published_scaling_data.csv": ("hidden_feedback", "真实公开观测", False, False),
    "supplementary_NQ_experiment.csv": ("fit", "半合成", True, False),
    "supplementary_NQ_experiment_expanded.csv": ("development_validation", "半合成", True, False),
    "supplementary_NQ_experiment_large.csv": ("stress_extrapolation", "半合成", True, True),
    "supplementary_large_models.csv": ("metadata", "真实元数据", False, False),
    "supplementary_large_baseline.csv": ("stress_extrapolation", "估算外推", True, True),
    "open_model_family_metadata.csv": ("metadata", "真实元数据", False, False),
    "pythia_checkpoint_index.csv": ("metadata", "真实元数据", False, False),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(values: Iterable[float]) -> str:
    rounded = [round(float(value), 8) for value in values]
    return hashlib.sha256(stable_json(rounded).encode("utf-8")).hexdigest()[:20]


def finite_float(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def read_quality_means(q1_dir: Path) -> tuple[pd.DataFrame, dict[str, float], float]:
    scores_path = q1_dir / "sample_scores.csv.gz"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Q1 output not found: {scores_path}")
    scores = pd.read_csv(scores_path, usecols=["id", "domain", "q_score"])
    if scores["id"].duplicated().any() or scores["id"].isna().any():
        raise ValueError("Q1 sample scores must contain unique non-null IDs")
    scores["q_score"] = scores["q_score"].map(finite_float)
    if not scores["q_score"].between(0.0, 1.0).all():
        raise ValueError("Q1 q_score must be in [0, 1]")
    domain_means = scores.groupby("domain", sort=True)["q_score"].mean().to_dict()
    if not domain_means:
        raise ValueError("Q1 sample scores contain no domains")
    equal_domain_prior = float(sum(domain_means.values()) / len(domain_means))
    return scores, domain_means, equal_domain_prior


def build_quality_mapping(a_root: Path, domain_means: dict[str, float], prior: float) -> pd.DataFrame:
    mapping = pd.read_csv(a_root / "domain_mapping_guide.csv").copy()
    required = {"mixture_domain", "quality_domain", "mapping_type"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"A16 mapping is missing {required - set(mapping.columns)}")
    rows: list[dict[str, object]] = []
    for row in mapping.to_dict("records"):
        mixture_domain = str(row["mixture_domain"])
        quality_domain = str(row["quality_domain"])
        mapping_type = str(row["mapping_type"])
        direct_value = domain_means.get(quality_domain)
        if mapping_type in {"direct", "near_direct"} and direct_value is None:
            raise ValueError(f"A16 maps {mixture_domain} to missing Q1 domain {quality_domain}")
        if direct_value is None:
            value = prior
            value_source = "equal_domain_prior_for_inferred_mapping"
            effective_type = "inferred_global_prior"
        else:
            value = direct_value
            value_source = "Q1_domain_mean"
            effective_type = mapping_type
        rows.append({
            "mixture_domain": mixture_domain,
            "quality_domain": quality_domain,
            "mapping_type": mapping_type,
            "effective_mapping_type": effective_type,
            "quality_value": float(value),
            "quality_value_source": value_source,
            "note": str(row.get("note", "")),
        })
    result = pd.DataFrame(rows)
    if result["mixture_domain"].duplicated().any():
        raise ValueError("A16 contains duplicate mixture domains")
    return result


def build_a_tables(a_root: Path, q1_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _, domain_means, equal_domain_prior = read_quality_means(q1_dir)
    mapping = build_quality_mapping(a_root, domain_means, equal_domain_prior)
    quality_by_mixture = mapping.set_index("mixture_domain")["quality_value"].to_dict()
    mixture_type = mapping.set_index("mixture_domain")["effective_mapping_type"].to_dict()

    all_rows: list[pd.DataFrame] = []
    p_columns: list[str] | None = None
    loss_columns: list[str] | None = None
    for pair_id, mixture_name, loss_name, split_role, scale, scale_nominal_b, extrapolated in A_PAIRS:
        mixture_path = a_root / "regmix_tables" / mixture_name
        loss_path = a_root / "regmix_tables" / loss_name
        mixture = pd.read_csv(mixture_path)
        losses = pd.read_csv(loss_path)
        if "index" not in mixture.columns or "index" not in losses.columns:
            raise ValueError(f"{pair_id}: both files must contain index")
        if len(mixture) != len(losses) or set(mixture["index"]) != set(losses["index"]):
            raise ValueError(f"{pair_id}: mixture/loss index sets do not match")
        p_columns_now = [column for column in mixture.columns if column.startswith(MIXTURE_PREFIX)]
        loss_columns_now = [column for column in losses.columns if column.startswith(LOSS_PREFIX)]
        if p_columns is None:
            p_columns, loss_columns = p_columns_now, loss_columns_now
        if p_columns_now != p_columns or loss_columns_now != loss_columns:
            raise ValueError(f"{pair_id}: A4-A15 field order is inconsistent")
        loss_by_index = losses.set_index("index")
        records: list[dict[str, object]] = []
        for row in mixture.to_dict("records"):
            original_index = row["index"]
            proportions = [finite_float(row[column]) for column in p_columns]
            raw_sum = sum(proportions)
            if raw_sum <= 0:
                raise ValueError(f"{pair_id}/{original_index}: mixture sum is not positive")
            normalized = [value / raw_sum for value in proportions]
            loss_row = loss_by_index.loc[original_index]
            target_values = [finite_float(loss_row[column]) for column in loss_columns]
            p_fingerprint = fingerprint(normalized)
            mapped_values = [quality_by_mixture[column.removeprefix(MIXTURE_PREFIX)] for column in p_columns]
            inferred_mass = sum(
                proportion for proportion, column in zip(normalized, p_columns)
                if mixture_type[column.removeprefix(MIXTURE_PREFIX)] == "inferred_global_prior"
            )
            q_a = sum(proportion * quality for proportion, quality in zip(normalized, mapped_values))
            record: dict[str, object] = {
                "observation_id": f"A_{pair_id}_{original_index}",
                "source_group": "A",
                "source_file": f"regmix_tables/{mixture_name}|regmix_tables/{loss_name}",
                "provenance": "derived_Q1_quality_and_A4_A15_loss",
                "split_role": split_role,
                "group_id": f"A_{pair_id}",
                "lineage_group": f"A_p_{p_fingerprint}",
                "observed_fields": stable_json(["p_1...p_17", "L_A_domain_1...L_A_domain_13"]),
                "target_fields": stable_json(["L_A_macro", "L_A_domain_1...L_A_domain_13"]),
                "is_duplicate": False,
                "is_derived": True,
                "is_extrapolated": extrapolated,
                "is_condition_overlap": False,
                "a_pair_id": pair_id,
                "a_index": str(original_index),
                "scale": scale,
                "scale_nominal_B": scale_nominal_b,
                "N_params_B": scale_nominal_b,
                "D_tokens_B": "",
                "N_params_B_source": "A_filename_scale",
                "Q_A": q_a,
                "Q_A_inferred_mass": inferred_mass,
                "Q_A_quality_prior": equal_domain_prior,
                "L_A_macro": sum(target_values) / len(target_values),
                "p_sum_raw": raw_sum,
                "p_fingerprint": p_fingerprint,
            }
            for number, (column, value) in enumerate(zip(p_columns, normalized), start=1):
                record[f"p_{number}"] = value
                record[f"p_domain_{number}"] = column.removeprefix(MIXTURE_PREFIX)
            for number, value in enumerate(target_values, start=1):
                record[f"L_A_domain_{number}"] = value
            records.append(record)
        all_rows.append(pd.DataFrame(records))
    a_all = pd.concat(all_rows, ignore_index=True)
    a_all["p_repeated_across_scales"] = a_all["p_fingerprint"].duplicated(keep=False)
    a_all.to_csv(output_dir / "a_q1_outputs.csv", index=False)
    for split_role, filename in (
        ("fit", "a_fit.csv"),
        ("development_validation", "a_dev.csv"),
        ("hidden_feedback", "a_feedback.csv"),
        ("stress_extrapolation", "a_stress.csv"),
    ):
        a_all.loc[a_all["split_role"] == split_role].to_csv(output_dir / filename, index=False)

    quality_table = pd.DataFrame(
        [{"quality_domain": name, "q_domain_mean": value, "source": "Q1 sample_scores.csv.gz"}
         for name, value in sorted(domain_means.items())]
    )
    quality_table.to_csv(output_dir / "a_quality_domain_means.csv", index=False)
    mapping.to_csv(output_dir / "a16_quality_mapping_resolved.csv", index=False)
    return a_all, mapping, {
        "q1_scores_file": str((q1_dir / "sample_scores.csv.gz").resolve()),
        "q1_domain_means": domain_means,
        "equal_domain_prior": equal_domain_prior,
        "a_rows": int(len(a_all)),
        "a_unique_p_fingerprints": int(a_all["p_fingerprint"].nunique()),
        "a_repeated_p_rows": int(a_all["p_repeated_across_scales"].sum()),
        "a_inferred_mapping_domains": int((mapping["effective_mapping_type"] == "inferred_global_prior").sum()),
    }


def add_b_record(
    records: list[dict[str, object]],
    *,
    observation_id: str,
    source_file: str,
    provenance: str,
    split_role: str,
    group_id: str,
    lineage_group: str,
    n_params: object = "",
    d_tokens: object = "",
    q_score: object = "",
    val_loss: object = "",
    data_type: object = "",
    family: object = "",
    experiment_id: object = "",
    step: object = "",
    is_duplicate: bool = False,
    is_derived: bool = False,
    is_extrapolated: bool = False,
    is_condition_overlap: bool = False,
) -> None:
    record = {
        "observation_id": observation_id,
        "source_group": "B",
        "source_file": source_file,
        "provenance": provenance,
        "split_role": split_role,
        "group_id": group_id,
        "lineage_group": lineage_group,
        "observed_fields": stable_json([field for field, value in (
            ("N_params_B", n_params), ("D_tokens_B", d_tokens), ("Q_score", q_score), ("val_loss", val_loss)
        ) if value != ""]),
        "target_fields": stable_json(["val_loss"] if val_loss != "" else []),
        "is_duplicate": is_duplicate,
        "is_derived": is_derived,
        "is_extrapolated": is_extrapolated,
        "is_condition_overlap": is_condition_overlap,
        "N_params_B": n_params,
        "D_tokens_B": d_tokens,
        "Q_score": q_score,
        "val_loss": val_loss,
        "data_type": data_type,
        "family": family,
        "experiment_id": experiment_id,
        "step": step,
    }
    records.append(record)


def build_b_tables(b_root: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    records: list[dict[str, object]] = []
    b6 = pd.read_csv(b_root / "supplementary_NQ_experiment.csv")
    b6_keys = {
        (round(finite_float(row.N_params_B), 8), round(finite_float(row.D_tokens_B), 8), round(finite_float(row.Q_score), 8))
        for row in b6.itertuples()
    }
    b7 = pd.read_csv(b_root / "supplementary_NQ_experiment_expanded.csv")
    b7_keys = {
        (round(finite_float(row.N_params_B), 8), round(finite_float(row.D_tokens_B), 8), round(finite_float(row.Q_score), 8))
        for row in b7.itertuples()
    }
    b7_condition_keys = {
        (round(finite_float(row.N_params_B), 8), round(finite_float(row.D_tokens_B), 8))
        for row in b7.itertuples()
    }

    pythia = pd.read_csv(b_root / "pythia_training_log_existing.csv")
    for row_number, row in enumerate(pythia.to_dict("records"), start=1):
        n_value, d_value, loss = map(finite_float, (row["N_params_B"], row["D_tokens_B"], row["val_loss"]))
        model_group = f"pythia_N_{n_value:.8g}"
        add_b_record(records, observation_id=f"B1_{row_number}", source_file="pythia_training_log_existing.csv",
                     provenance="真实", split_role="fit", group_id=model_group, lineage_group=model_group,
                     n_params=n_value, d_tokens=d_value, val_loss=loss, family="Pythia", step=row.get("steps", ""))

    cerebras = pd.read_csv(b_root / "cerebras_training_log.csv")
    for row_number, row in enumerate(cerebras.to_dict("records"), start=1):
        n_value, d_value, loss = map(finite_float, (row["N_params_B"], row["D_tokens_B"], row["val_loss"]))
        model_group = f"cerebras_N_{n_value:.8g}"
        add_b_record(records, observation_id=f"B2_{row_number}", source_file="cerebras_training_log.csv",
                     provenance="半合成", split_role="development_validation", group_id=model_group,
                     lineage_group=model_group, n_params=n_value, d_tokens=d_value, val_loss=loss,
                     family="Cerebras-GPT", step=row.get("steps", ""), is_derived=True)

    trajectory_dir = b_root / "training_trajectories"
    for trajectory_path in sorted(trajectory_dir.glob("*.csv")):
        trajectory = pd.read_csv(trajectory_path)
        for row_number, row in enumerate(trajectory.to_dict("records"), start=1):
            n_value, d_value, loss = map(finite_float, (row["N_params_B"], row["D_tokens_B"], row["val_loss"]))
            model_group = f"pythia_N_{n_value:.8g}"
            add_b_record(records, observation_id=f"B3_{trajectory_path.stem}_{row_number}",
                         source_file=f"training_trajectories/{trajectory_path.name}", provenance="插值",
                         split_role="development_validation", group_id=model_group, lineage_group=model_group,
                         n_params=n_value, d_tokens=d_value, val_loss=loss, family="Pythia",
                         step=row.get("step", ""), is_derived=True)

    for source_name, family_column in (("scaling_baseline.csv", "family"), ("published_scaling_data.csv", "family")):
        frame = pd.read_csv(b_root / source_name)
        for row_number, row in enumerate(frame.to_dict("records"), start=1):
            n_value, d_value, loss = map(finite_float, (row["N_params_B"], row["D_tokens_B"], row["val_loss"]))
            family = str(row.get(family_column, "unknown"))
            group = f"{source_name[:-4]}_{family}"
            add_b_record(records, observation_id=f"{source_name[:-4]}_{row_number}", source_file=source_name,
                         provenance="真实公开观测", split_role="hidden_feedback", group_id=group,
                         lineage_group=group, n_params=n_value, d_tokens=d_value, val_loss=loss, family=family)

    for source_name, split_role, provenance, extrapolated in (
        ("supplementary_NQ_experiment.csv", "fit", "半合成", False),
        ("supplementary_NQ_experiment_expanded.csv", "development_validation", "半合成", False),
        ("supplementary_NQ_experiment_large.csv", "stress_extrapolation", "半合成", True),
    ):
        frame = pd.read_csv(b_root / source_name)
        for row_number, row in enumerate(frame.to_dict("records"), start=1):
            n_value, d_value, q_value, loss = map(
                finite_float, (row["N_params_B"], row["D_tokens_B"], row["Q_score"], row["val_loss"])
            )
            key = (round(n_value, 8), round(d_value, 8), round(q_value, 8))
            duplicate = source_name == "supplementary_NQ_experiment_expanded.csv" and key in b6_keys
            condition_overlap = (
                source_name == "supplementary_NQ_experiment_large.csv"
                and (round(n_value, 8), round(d_value, 8)) in b7_condition_keys
            )
            if duplicate:
                effective_role = "overlap_not_scored"
            else:
                effective_role = split_role
            group = f"NQ_N_{n_value:.8g}_D_{d_value:.8g}"
            add_b_record(records, observation_id=f"{source_name[:-4]}_{row_number}", source_file=source_name,
                         provenance=provenance, split_role=effective_role, group_id=group,
                         lineage_group=f"{source_name[:-4]}_{group}", n_params=n_value, d_tokens=d_value,
                         q_score=q_value, val_loss=loss, data_type=row.get("data_type", ""),
                         experiment_id=row.get("experiment_id", ""), is_duplicate=duplicate,
                         is_derived=True, is_extrapolated=extrapolated, is_condition_overlap=condition_overlap)

    for source_name in ("supplementary_large_baseline.csv",):
        frame = pd.read_csv(b_root / source_name)
        for row_number, row in enumerate(frame.to_dict("records"), start=1):
            n_value, d_value, loss = map(finite_float, (row["N_params_B"], row["D_tokens_B"], row["val_loss"]))
            family = str(row.get("family", "unknown"))
            group = f"{source_name[:-4]}_{family}"
            add_b_record(records, observation_id=f"{source_name[:-4]}_{row_number}", source_file=source_name,
                         provenance="估算外推", split_role="stress_extrapolation", group_id=group,
                         lineage_group=group, n_params=n_value, d_tokens=d_value, val_loss=loss,
                         family=family, is_derived=True, is_extrapolated=True)

    metadata_specs = (
        ("supplementary_large_models.csv", "真实元数据"),
        ("open_model_family_metadata.csv", "真实元数据"),
        ("pythia_checkpoint_index.csv", "真实元数据"),
    )
    metadata_rows: list[dict[str, object]] = []
    for source_name, provenance in metadata_specs:
        frame = pd.read_csv(b_root / source_name)
        for row_number, row in enumerate(frame.to_dict("records"), start=1):
            model_key = str(row.get("model_name", row.get("model_repo", row.get("family", row_number))))
            metadata_rows.append({
                "observation_id": f"{source_name[:-4]}_{row_number}",
                "source_group": "B",
                "source_file": source_name,
                "provenance": provenance,
                "split_role": "metadata",
                "group_id": model_key,
                "lineage_group": model_key,
                "observed_fields": stable_json(list(row.keys())),
                "target_fields": stable_json([]),
                "is_duplicate": False,
                "is_derived": False,
                "is_extrapolated": False,
                "is_condition_overlap": False,
                **row,
            })
    metadata = pd.DataFrame(metadata_rows)
    b_all = pd.DataFrame(records)
    b_all["condition_key"] = b_all.apply(
        lambda row: "|".join(str(row[field]) for field in ("N_params_B", "D_tokens_B", "Q_score")), axis=1
    )
    b_all.to_csv(output_dir / "b_standard.csv", index=False)
    for role, filename in (
        ("fit", "b_fit.csv"),
        ("development_validation", "b_dev.csv"),
        ("hidden_feedback", "b_feedback.csv"),
        ("stress_extrapolation", "b_stress.csv"),
        ("overlap_not_scored", "b_overlap.csv"),
    ):
        b_all.loc[b_all["split_role"] == role].to_csv(output_dir / filename, index=False)
    metadata.to_csv(output_dir / "b_metadata.csv", index=False)
    return b_all, {
        "b_rows_with_targets": int(len(b_all)),
        "b_fit_rows": int((b_all["split_role"] == "fit").sum()),
        "b_dev_rows": int((b_all["split_role"] == "development_validation").sum()),
        "b_b7_overlap_rows": int((b_all["split_role"] == "overlap_not_scored").sum()),
        "b8_b7_condition_overlap_rows": int(b_all["is_condition_overlap"].sum()),
        "b_metadata_rows": int(len(metadata)),
    }


def build_manifest(
    *,
    a_root: Path,
    b_root: Path,
    q1_dir: Path,
    output_dir: Path,
    public_input_dir: Path | None,
    a_summary: dict,
    b_summary: dict,
) -> dict:
    source_files = [a_root / "domain_mapping_guide.csv", a_root / "regmix_domain_summary.csv",
                    q1_dir / "sample_scores.csv.gz"]
    for _, mixture_name, loss_name, *_ in A_PAIRS:
        source_files.extend((a_root / "regmix_tables" / mixture_name,
                             a_root / "regmix_tables" / loss_name))
    source_files.extend(b_root / name for name in B_ROLE)
    source_files.extend(sorted((b_root / "training_trajectories").glob("*.csv")))
    hashes = []
    for path in source_files:
        if path.is_file():
            hashes.append({"file": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    derived_files = []
    for path in sorted(output_dir.glob("*.csv")):
        derived_files.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    public_files = []
    if public_input_dir and public_input_dir.is_dir():
        for path in sorted(public_input_dir.glob("*.csv")):
            public_files.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "created_on": "2026-09-25",
        "purpose": "A+B joint scaling-law data preparation",
        "source_hashes": hashes,
        "derived_hashes": derived_files,
        "public_input_hashes": public_files,
        "a_summary": a_summary,
        "b_summary": b_summary,
        "rules": {
            "a_b_row_join": "forbidden",
            "missing_a_N_or_D": "left_missing",
            "q_a_inferred_domain_handling": "equal_domain_prior",
            "b7_repeated_b6_rows": "overlap_not_scored",
            "b8_condition_overlap": "stress_only_and_reported",
            "a_shared_mixture_fingerprints": "shared_lineage_group_not_independent_rows",
        },
    }


def export_public_inputs(derived_dir: Path, public_dir: Path) -> dict[str, int]:
    public_dir.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(derived_dir / "a_feedback.csv")
    b = pd.read_csv(derived_dir / "b_feedback.csv")
    a_stress = pd.read_csv(derived_dir / "a_stress.csv")
    b_stress = pd.read_csv(derived_dir / "b_stress.csv")

    a_columns = ["observation_id", "N_params_B", "scale", "p_fingerprint", *P_COLUMNS]
    a.loc[:, a_columns].to_csv(public_dir / "a_hidden_feedback_input.csv", index=False)
    a_stress.loc[:, a_columns].to_csv(public_dir / "a_stress_input.csv", index=False)

    b_columns = ["observation_id", "N_params_B", "D_tokens_B", "Q_score", "source_file", "family"]
    b.loc[:, b_columns].to_csv(public_dir / "b_hidden_feedback_input.csv", index=False)
    b_stress.loc[:, b_columns].to_csv(public_dir / "b_stress_input.csv", index=False)

    b9 = pd.read_csv(derived_dir / "b_metadata.csv")
    b9 = b9[b9["source_file"] == "supplementary_large_models.csv"]
    b9_columns = ["observation_id", "N_params_B", "D_tokens_B", "source_file", "model_name"]
    b9.loc[:, [column for column in b9_columns if column in b9.columns]].to_csv(
        public_dir / "b9_large_models_input.csv", index=False
    )
    return {
        "a_hidden_feedback_input": int(len(a)),
        "b_hidden_feedback_input": int(len(b)),
        "a_stress_input": int(len(a_stress)),
        "b_stress_input": int(len(b_stress)),
        "b9_large_models_input": int(len(b9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path, required=True)
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--q1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-input-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a_all, mapping, a_summary = build_a_tables(args.a_root, args.q1_dir, args.output_dir)
    b_all, b_summary = build_b_tables(args.b_root, args.output_dir)
    public_counts = None
    if args.public_input_dir:
        public_counts = export_public_inputs(args.output_dir, args.public_input_dir)
    manifest = build_manifest(
        a_root=args.a_root, b_root=args.b_root, q1_dir=args.q1_dir, output_dir=args.output_dir,
        public_input_dir=args.public_input_dir,
        a_summary=a_summary, b_summary=b_summary,
    )
    manifest["a_rows_by_role"] = a_all["split_role"].value_counts().to_dict()
    manifest["b_rows_by_role"] = b_all["split_role"].value_counts().to_dict()
    if public_counts:
        manifest["public_input_counts"] = public_counts
    (args.output_dir / "joint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"a": a_summary, "b": b_summary}, ensure_ascii=False, indent=2))
    print(f"Wrote joint tables to {args.output_dir}")


if __name__ == "__main__":
    main()
