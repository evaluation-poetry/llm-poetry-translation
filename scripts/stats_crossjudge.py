#!/usr/bin/env python3
"""Cross-judge agreement, human alignment, and direct self-preference tests.

The direct bias diagnostic is a paired difference-in-differences by poem:
``(DeepSeekJudge - GPTJudge)`` for DeepSeek-family candidates minus the same
quantity for all other candidates.  Positive agreement alone is never treated
as evidence that self-preference is absent.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from stats_intensity import (
    BOOTSTRAP_SAMPLES,
    JUDGE_DIMENSIONS,
    SEED,
    _finite,
    _wilcoxon_p,
    holm_adjust,
    paired_summary,
    read_rows,
    write_csv,
)


OUTPUT_NAMES = [
    "agreement",
    "self_preference",
    "judge_human_alignment",
    "correlation_difference_bootstrap",
    "within_family",
    "rank_stability",
    "dimension_sensitivity",
]
PERMUTATION_SAMPLES = 10_000
MIN_VALID_FRACTION = 0.80
HUMAN_DIMENSIONS = ["MF", "IR", "EV", "LR", "MD", "PT"]
EXPECTED_ANNOTATORS = [f"A{index:02d}" for index in range(1, 8)]
HUMAN_MISSING_POLICY = (
    "20x7x6x7_expected; >=6_of_7_unique_annotators_per_dimension; "
    "equal_mean_of_six_dimension_means"
)

ORIGINAL_SYSTEMS = [
    "baidu_translate",
    "deepseek_v4_flash_non_thinking",
    "deepseek_v4_flash_thinking",
    "qwen36_plus_non_thinking",
    "qwen36_plus_thinking",
    "claude_sonnet46_non_thinking",
    "claude_sonnet46_thinking",
]
FAMILY_PAIRS = {
    "deepseek": ("deepseek_v4_flash_non_thinking", "deepseek_v4_flash_thinking"),
    "qwen": ("qwen36_plus_non_thinking", "qwen36_plus_thinking"),
    "claude": ("claude_sonnet46_non_thinking", "claude_sonnet46_thinking"),
}
RANK_FAMILIES = {
    "Baidu": ["baidu_translate"],
    "DeepSeek": ["deepseek_v4_flash_non_thinking", "deepseek_v4_flash_thinking"],
    "Qwen": ["qwen36_plus_non_thinking", "qwen36_plus_thinking"],
    "Claude": ["claude_sonnet46_non_thinking", "claude_sonnet46_thinking"],
}


def _judge_index(
    rows: list[dict[str, Any]],
    *,
    label: str,
    expected_record_count: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        record_id = str(row.get("record_id") or "")
        system_id = str(row.get("system_id") or "")
        if not record_id or not system_id:
            raise ValueError(f"{label} row {row_number} has an empty record_id or system_id")
        if system_id not in ORIGINAL_SYSTEMS:
            raise ValueError(f"{label} contains unexpected system_id: {system_id}")
        key = (record_id, system_id)
        if key in indexed:
            raise ValueError(f"{label} duplicate {key}")
        status = row.get("status")
        if status not in (None, "", "ok"):
            raise ValueError(f"{label} non-ok status at {key}: {status}")
        for field in ["average_score", *JUDGE_DIMENSIONS]:
            _finite(row.get(field), f"{label} {record_id}/{system_id}/{field}")
        indexed[key] = row

    expected_rows = expected_record_count * len(ORIGINAL_SYSTEMS)
    if len(indexed) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} unique candidates, found {len(indexed)}")
    record_ids = sorted({record_id for record_id, _ in indexed})
    if len(record_ids) != expected_record_count:
        raise ValueError(f"{label}: expected {expected_record_count} poems, found {len(record_ids)}")
    expected_keys = {(record_id, system_id) for record_id in record_ids for system_id in ORIGINAL_SYSTEMS}
    if set(indexed) != expected_keys:
        raise ValueError(f"{label}: candidates are not an exact poem x 7-system panel")
    return indexed


def _human_composite(
    rows: list[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    seen_cells: set[tuple[str, str, str, str]] = set()
    seen_rows: set[tuple[str, str, str, str]] = set()
    expected_records = {record_id for record_id, _ in expected_keys}
    expected_systems = {system_id for _, system_id in expected_keys}
    seen_annotators: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        record_id = str(row.get("record_id") or "")
        system_id = str(row.get("system_id") or "")
        annotator = str(row.get("annotator") or "")
        dimension = str(row.get("dim") or "")
        if record_id not in expected_records:
            raise ValueError(f"human row {row_number} has unknown/missing record_id: {record_id!r}")
        if system_id not in expected_systems:
            raise ValueError(f"human row {row_number} has unknown/missing system_id: {system_id!r}")
        if annotator not in EXPECTED_ANNOTATORS:
            raise ValueError(f"human row {row_number} has unknown/missing annotator: {annotator!r}")
        if dimension not in HUMAN_DIMENSIONS:
            raise ValueError(f"human row {row_number} has unknown/missing dimension: {dimension!r}")
        key = (record_id, system_id)
        if key not in expected_keys:
            raise ValueError(f"human data contains unexpected candidate {key}")
        cell = (annotator, record_id, system_id, dimension)
        if cell in seen_rows:
            raise ValueError(f"human data duplicate cell: {cell}")
        seen_rows.add(cell)
        score = row.get("score")
        if score is None or not str(score).strip():
            continue
        numeric = _finite(score, f"human {record_id}/{system_id}/{annotator}/{dimension}")
        if not numeric.is_integer() or not 1 <= numeric <= 6:
            raise ValueError(f"human score must be an integer in [1, 6], got {score!r}")
        grouped[(record_id, system_id, dimension)][annotator] = int(numeric)
        seen_cells.add(cell)
        seen_annotators.add(annotator)
    if seen_annotators != set(EXPECTED_ANNOTATORS):
        raise ValueError("human panel must contain all seven expected annotators")

    composites: dict[tuple[str, str], float] = {}
    for record_id, system_id in sorted(expected_keys):
        dimension_means: list[float] = []
        for dimension in HUMAN_DIMENSIONS:
            scores = grouped.get((record_id, system_id, dimension), {})
            if len(scores) < 6:
                raise ValueError(
                    f"human panel {record_id}/{system_id}/{dimension} has {len(scores)} annotators; require >=6"
                )
            dimension_means.append(float(np.mean(list(scores.values()))))
        composites[(record_id, system_id)] = float(np.mean(dimension_means))
    expected_cells = len(expected_keys) * len(HUMAN_DIMENSIONS) * len(EXPECTED_ANNOTATORS)
    metadata = {
        "human_missing_cells": expected_cells - len(seen_cells),
        "human_expected_cells": expected_cells,
        "human_observed_cells": len(seen_cells),
        "human_missing_policy": HUMAN_MISSING_POLICY,
    }
    return composites, metadata


def _correlation(method: str, x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("correlations require equal arrays with at least three observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            result = stats.pearsonr(x, y)
        elif method == "spearman":
            result = stats.spearmanr(x, y)
        elif method == "kendall":
            result = stats.kendalltau(x, y)
        else:
            raise ValueError(f"unknown correlation method: {method}")
    estimate = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(estimate) or not math.isfinite(p_value):
        return None, None
    return estimate, p_value


def _block_indices(keys: Sequence[tuple[str, str]]) -> tuple[list[str], dict[str, np.ndarray]]:
    record_ids = sorted({record_id for record_id, _ in keys})
    blocks = {
        record_id: np.asarray([index for index, key in enumerate(keys) if key[0] == record_id], dtype=int)
        for record_id in record_ids
    }
    return record_ids, blocks


def _bootstrap_correlation(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    keys: Sequence[tuple[str, str]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None, int]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    record_ids, blocks = _block_indices(keys)
    boot: list[float] = []
    for _ in range(samples):
        selected = rng.choice(record_ids, size=len(record_ids), replace=True)
        indices = np.concatenate([blocks[str(record_id)] for record_id in selected])
        estimate = _correlation(method, x[indices], y[indices])[0]
        if estimate is not None:
            boot.append(estimate)
    if not boot:
        return None, None, 0
    low, high = np.percentile(np.asarray(boot), [2.5, 97.5])
    return float(low), float(high), len(boot)


def _permutation_correlation_p(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    keys: Sequence[tuple[str, str]],
    observed: float | None,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, int]:
    if samples <= 0:
        raise ValueError("permutation_samples must be positive")
    record_ids, blocks = _block_indices(keys)
    x_order = np.concatenate([blocks[record_id] for record_id in record_ids])
    valid: list[float] = []
    for _ in range(samples):
        permuted_records = rng.permutation(record_ids)
        y_order = np.concatenate([blocks[str(record_id)] for record_id in permuted_records])
        estimate = _correlation(method, x[x_order], y[y_order])[0]
        if estimate is not None:
            valid.append(estimate)
    if observed is None or not valid:
        return None, len(valid)
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in valid)
    return (extreme + 1) / (len(valid) + 1), len(valid)


def _holm_rows(rows: list[dict[str, Any]]) -> None:
    valid_indices = [index for index, row in enumerate(rows) if row.get("p_raw") is not None]
    adjusted = holm_adjust([float(rows[index]["p_raw"]) for index in valid_indices])
    for row in rows:
        row["p_holm"] = None
    for index, p_holm in zip(valid_indices, adjusted):
        rows[index]["p_holm"] = p_holm


def _correlation_rows(
    x: np.ndarray,
    y: np.ndarray,
    keys: Sequence[tuple[str, str]],
    *,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
    min_valid_fraction: float,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ["pearson", "spearman", "kendall"]:
        estimate, analytic_p = _correlation(method, x, y)
        ci_low, ci_high, bootstrap_valid = _bootstrap_correlation(
            method,
            x,
            y,
            keys,
            rng=rng,
            samples=bootstrap_samples,
        )
        permutation_p, permutation_valid = _permutation_correlation_p(
            method,
            x,
            y,
            keys,
            estimate,
            rng=rng,
            samples=permutation_samples,
        )
        bootstrap_fraction = bootstrap_valid / bootstrap_samples
        permutation_fraction = permutation_valid / permutation_samples
        meets_minimum = min(bootstrap_fraction, permutation_fraction) >= min_valid_fraction
        if bootstrap_fraction < min_valid_fraction:
            ci_low = ci_high = None
        if permutation_fraction < min_valid_fraction:
            permutation_p = None
        rows.append(
            {
                **(extra or {}),
                "method": method,
                "n": len(keys),
                "n_candidates": len(keys),
                "n_poems": len({key[0] for key in keys}),
                "unit": "candidate",
                "resampling_unit": "record_id",
                "seed": seed,
                "estimate": estimate,
                "effect": estimate,
                "effect_name": "correlation",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_raw": permutation_p,
                "p_block_permutation": permutation_p,
                "p_analytic_secondary": analytic_p,
                "bootstrap_requested": bootstrap_samples,
                "bootstrap_valid": bootstrap_valid,
                "bootstrap_valid_fraction": bootstrap_fraction,
                "permutation_requested": permutation_samples,
                "permutation_valid": permutation_valid,
                "permutation_valid_fraction": permutation_fraction,
                "minimum_valid_fraction": min_valid_fraction,
                "meets_min_valid_fraction": int(meets_minimum),
            }
        )
    _holm_rows(rows)
    return rows


def _bootstrap_correlation_difference(
    method: str,
    deepseek: np.ndarray,
    gpt: np.ndarray,
    human: np.ndarray,
    keys: Sequence[tuple[str, str]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None, float | None, int]:
    record_ids, blocks = _block_indices(keys)
    boot: list[float] = []
    for _ in range(samples):
        selected = rng.choice(record_ids, size=len(record_ids), replace=True)
        indices = np.concatenate([blocks[str(record_id)] for record_id in selected])
        gpt_corr = _correlation(method, gpt[indices], human[indices])[0]
        deepseek_corr = _correlation(method, deepseek[indices], human[indices])[0]
        if gpt_corr is not None and deepseek_corr is not None:
            boot.append(gpt_corr - deepseek_corr)
    if not boot:
        return None, None, None, 0
    boot_array = np.asarray(boot)
    low, high = np.percentile(boot_array, [2.5, 97.5])
    nonpositive = int(np.count_nonzero(boot_array <= 0))
    nonnegative = int(np.count_nonzero(boot_array >= 0))
    p_value = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (len(boot) + 1))
    return float(low), float(high), float(p_value), len(boot)


def _within_family_rows(
    deepseek_index: dict[tuple[str, str], dict[str, Any]],
    gpt_index: dict[tuple[str, str], dict[str, Any]],
    *,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judge_name, indexed in [("deepseek", deepseek_index), ("gpt", gpt_index)]:
        record_ids = sorted({record_id for record_id, _ in indexed})
        for family, (nt_system, t_system) in FAMILY_PAIRS.items():
            nt = np.asarray([_finite(indexed[(rid, nt_system)]["average_score"], "nt") for rid in record_ids])
            thinking = np.asarray([_finite(indexed[(rid, t_system)]["average_score"], "t") for rid in record_ids])
            summary = paired_summary(nt, thinking, rng=rng, bootstrap_samples=bootstrap_samples)
            rows.append(
                {
                    "judge": judge_name,
                    "family": family,
                    "metric": "average_score",
                    "left_system": nt_system,
                    "right_system": t_system,
                    "n": len(record_ids),
                    "unit": "record_id",
                    "seed": seed,
                    "mean_nt": summary["mean_left"],
                    "mean_t": summary["mean_right"],
                    "delta_t_minus_nt": summary["delta_right_minus_left"],
                    "direction": summary["direction"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "p_raw": summary["p_raw"],
                    "cohen_dz": summary["cohen_dz"],
                    "dz_status": summary["dz_status"],
                    "effect": summary["cohen_dz"],
                    "effect_name": "cohen_dz",
                }
            )
    adjusted = holm_adjust([float(row["p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
    for family in FAMILY_PAIRS:
        family_rows = [row for row in rows if row["family"] == family]
        signs = [np.sign(float(row["delta_t_minus_nt"])) for row in family_rows]
        agreement = "same" if signs[0] == signs[1] else "opposite"
        if 0 in signs:
            agreement = "both_tie" if signs[0] == signs[1] else "one_tie"
        for row in family_rows:
            row["direction_agreement"] = agreement
    return rows


def _fit_fe_interaction(
    metric: str,
    deepseek_index: dict[tuple[str, str], dict[str, Any]],
    gpt_index: dict[tuple[str, str], dict[str, Any]],
    selected_records: Sequence[str],
    *,
    excluded_system: str | None = None,
    cluster_p: bool,
) -> tuple[float, float | None]:
    systems = [system for system in ORIGINAL_SYSTEMS if system != excluded_system]
    observations: list[tuple[float, int, int, int, int]] = []
    for cluster_index, record_id in enumerate(selected_records):
        for system_index, system_id in enumerate(systems):
            is_deepseek_family = int(system_id.startswith("deepseek_"))
            for judge_deepseek, indexed in [(0, gpt_index), (1, deepseek_index)]:
                observations.append(
                    (
                        _finite(indexed[(record_id, system_id)][metric], metric),
                        judge_deepseek,
                        judge_deepseek * is_deepseek_family,
                        cluster_index,
                        system_index,
                    )
                )
    n_records = len(selected_records)
    n_systems = len(systems)
    x = np.zeros((len(observations), 3 + (n_records - 1) + (n_systems - 1)), dtype=float)
    y = np.asarray([observation[0] for observation in observations], dtype=float)
    clusters = np.asarray([observation[3] for observation in observations], dtype=int)
    x[:, 0] = 1.0
    for row_index, (_, judge, interaction, record_index, system_index) in enumerate(observations):
        x[row_index, 1] = judge
        x[row_index, 2] = interaction
        if record_index > 0:
            x[row_index, 2 + record_index] = 1.0
        if system_index > 0:
            x[row_index, 2 + (n_records - 1) + system_index] = 1.0
    beta = np.linalg.pinv(x) @ y
    coefficient = float(beta[2])
    if not cluster_p:
        return coefficient, None
    residuals = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    rank = int(np.linalg.matrix_rank(x))
    cluster_count = len(np.unique(clusters))
    correction = (cluster_count / (cluster_count - 1)) * ((len(y) - 1) / (len(y) - rank))
    variance = float((bread @ meat @ bread)[2, 2] * correction)
    standard_error = math.sqrt(max(0.0, variance))
    if standard_error <= 1e-12:
        p_value = 0.0 if abs(coefficient) > 1e-12 else 1.0
    else:
        statistic = coefficient / standard_error
        p_value = float(2 * stats.t.sf(abs(statistic), df=cluster_count - 1))
    return coefficient, p_value


def _fixed_effect_preference_row(
    metric: str,
    deepseek_index: dict[tuple[str, str], dict[str, Any]],
    gpt_index: dict[tuple[str, str], dict[str, Any]],
    *,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
    min_valid_fraction: float,
) -> dict[str, Any]:
    record_ids = sorted({record_id for record_id, _ in deepseek_index})
    coefficient, cluster_p = _fit_fe_interaction(
        metric,
        deepseek_index,
        gpt_index,
        record_ids,
        cluster_p=True,
    )
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        selected = [str(record_id) for record_id in rng.choice(record_ids, size=len(record_ids), replace=True)]
        value, _ = _fit_fe_interaction(
            metric,
            deepseek_index,
            gpt_index,
            selected,
            cluster_p=False,
        )
        if math.isfinite(value):
            bootstrap.append(value)
    valid_fraction = len(bootstrap) / bootstrap_samples
    if bootstrap and valid_fraction >= min_valid_fraction:
        ci_low, ci_high = np.percentile(np.asarray(bootstrap), [2.5, 97.5])
        ci_low_value: float | None = float(ci_low)
        ci_high_value: float | None = float(ci_high)
    else:
        ci_low_value = ci_high_value = None
    leave_one_out = [
        _fit_fe_interaction(
            metric,
            deepseek_index,
            gpt_index,
            record_ids,
            excluded_system=system_id,
            cluster_p=False,
        )[0]
        for system_id in ORIGINAL_SYSTEMS
    ]
    direction = (
        "deepseek_judge_extra_for_deepseek_family"
        if coefficient > 0
        else "deepseek_judge_lower_for_deepseek_family"
        if coefficient < 0
        else "no_mean_interaction"
    )
    return {
        "analysis": "fixed_effects_ols_clustered",
        "metric": metric,
        "formula": "score ~ judge_deepseek + judge_deepseek:is_deepseek_family + C(record_id) + C(system_id)",
        "covariance": "CR1 cluster-robust by record_id",
        "family_main_effect": "omitted_collinear_with_system_fixed_effects",
        "n": len(record_ids) * len(ORIGINAL_SYSTEMS),
        "n_observations": len(record_ids) * len(ORIGINAL_SYSTEMS) * 2,
        "n_candidates": len(record_ids) * len(ORIGINAL_SYSTEMS),
        "n_poems": len(record_ids),
        "n_systems": len(ORIGINAL_SYSTEMS),
        "unit": "candidate",
        "resampling_unit": "record_id",
        "seed": seed,
        "mean_interaction": coefficient,
        "interaction_coefficient": coefficient,
        "effect": coefficient,
        "effect_name": "judge_deepseek:is_deepseek_family",
        "direction": direction,
        "ci_low": ci_low_value,
        "ci_high": ci_high_value,
        "p_raw": cluster_p,
        "bootstrap_requested": bootstrap_samples,
        "bootstrap_valid": len(bootstrap),
        "bootstrap_valid_fraction": valid_fraction,
        "minimum_valid_fraction": min_valid_fraction,
        "meets_min_valid_fraction": int(valid_fraction >= min_valid_fraction),
        "leave_one_system_out_min": float(min(leave_one_out)),
        "leave_one_system_out_max": float(max(leave_one_out)),
    }


def _preference_row(
    metric: str,
    deepseek_index: dict[tuple[str, str], dict[str, Any]],
    gpt_index: dict[tuple[str, str], dict[str, Any]],
    *,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    record_ids = sorted({record_id for record_id, _ in deepseek_index})
    deepseek_systems = [system for system in ORIGINAL_SYSTEMS if system.startswith("deepseek_")]
    other_systems = [system for system in ORIGINAL_SYSTEMS if system not in deepseek_systems]
    ds_differences = np.asarray(
        [
            np.mean(
                [
                    _finite(deepseek_index[(rid, system)][metric], metric)
                    - _finite(gpt_index[(rid, system)][metric], metric)
                    for system in deepseek_systems
                ]
            )
            for rid in record_ids
        ]
    )
    other_differences = np.asarray(
        [
            np.mean(
                [
                    _finite(deepseek_index[(rid, system)][metric], metric)
                    - _finite(gpt_index[(rid, system)][metric], metric)
                    for system in other_systems
                ]
            )
            for rid in record_ids
        ]
    )
    summary = paired_summary(
        other_differences,
        ds_differences,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
    )
    interaction = float(summary["delta_right_minus_left"])
    direction = (
        "deepseek_judge_extra_for_deepseek_family"
        if interaction > 0
        else "deepseek_judge_lower_for_deepseek_family"
        if interaction < 0
        else "no_mean_interaction"
    )
    return {
        "analysis": "poem_contrast_descriptive",
        "metric": metric,
        "contrast": "mean(DeepSeekJudge-GPTJudge|DeepSeek-family)-mean(same|other)",
        "n": len(record_ids),
        "unit": "record_id",
        "seed": seed,
        "mean_deepseek_family_judge_difference": float(np.mean(ds_differences)),
        "mean_other_family_judge_difference": float(np.mean(other_differences)),
        "mean_interaction": interaction,
        "direction": direction,
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "p_raw": summary["p_raw"],
        "cohen_dz": summary["cohen_dz"],
        "dz_status": summary["dz_status"],
        "effect": summary["cohen_dz"],
        "effect_name": "cohen_dz",
    }


def _selection_weights(values: np.ndarray, *, highest: bool) -> np.ndarray:
    extrema = (np.max if highest else np.min)(values, axis=1, keepdims=True)
    selected = np.isclose(values, extrema, rtol=0.0, atol=1e-12)
    return selected / selected.sum(axis=1, keepdims=True)


def _identity(values: np.ndarray, entity_ids: Sequence[str], *, highest: bool) -> tuple[str, set[str]]:
    extreme = (np.max if highest else np.min)(values)
    selected = {
        entity_ids[index]
        for index, value in enumerate(values)
        if math.isclose(float(value), float(extreme), rel_tol=0.0, abs_tol=1e-12)
    }
    return "|".join(sorted(selected)), selected


def _rank_level_rows(
    deepseek_matrix: np.ndarray,
    gpt_matrix: np.ndarray,
    record_ids: Sequence[str],
    entity_ids: Sequence[str],
    *,
    level: str,
    id_field: str,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
    min_valid_fraction: float,
) -> list[dict[str, Any]]:
    indices = rng.integers(0, len(record_ids), size=(bootstrap_samples, len(record_ids)))
    deepseek_boot_means = deepseek_matrix[indices].mean(axis=1)
    gpt_boot_means = gpt_matrix[indices].mean(axis=1)
    deepseek_means = deepseek_matrix.mean(axis=0)
    gpt_means = gpt_matrix.mean(axis=0)
    deepseek_ranks = stats.rankdata(-deepseek_means, method="average")
    gpt_ranks = stats.rankdata(-gpt_means, method="average")
    deepseek_boot_ranks = np.vstack(
        [stats.rankdata(-values, method="average") for values in deepseek_boot_means]
    )
    gpt_boot_ranks = np.vstack(
        [stats.rankdata(-values, method="average") for values in gpt_boot_means]
    )
    deepseek_top = _selection_weights(deepseek_boot_means, highest=True)
    gpt_top = _selection_weights(gpt_boot_means, highest=True)
    deepseek_bottom = _selection_weights(deepseek_boot_means, highest=False)
    gpt_bottom = _selection_weights(gpt_boot_means, highest=False)
    differences = gpt_boot_means - deepseek_boot_means

    rows: list[dict[str, Any]] = []
    for entity_index, entity_id in enumerate(entity_ids):
        difference_low, difference_high = np.percentile(differences[:, entity_index], [2.5, 97.5])
        deepseek_rank_low, deepseek_rank_high = np.percentile(
            deepseek_boot_ranks[:, entity_index], [2.5, 97.5]
        )
        gpt_rank_low, gpt_rank_high = np.percentile(gpt_boot_ranks[:, entity_index], [2.5, 97.5])
        rows.append(
            {
                "row_type": level,
                "level": level,
                id_field: entity_id,
                "n": len(record_ids),
                "unit": "record_id",
                "seed": seed,
                "deepseek_mean": float(deepseek_means[entity_index]),
                "gpt_mean": float(gpt_means[entity_index]),
                "mean_difference_gpt_minus_deepseek": float(
                    gpt_means[entity_index] - deepseek_means[entity_index]
                ),
                "mean_difference_ci_low": float(difference_low),
                "mean_difference_ci_high": float(difference_high),
                "deepseek_rank": float(deepseek_ranks[entity_index]),
                "gpt_rank": float(gpt_ranks[entity_index]),
                "rank_delta_gpt_minus_deepseek": float(
                    gpt_ranks[entity_index] - deepseek_ranks[entity_index]
                ),
                "deepseek_bootstrap_mean_rank": float(deepseek_boot_ranks[:, entity_index].mean()),
                "deepseek_rank_ci_low": float(deepseek_rank_low),
                "deepseek_rank_ci_high": float(deepseek_rank_high),
                "gpt_bootstrap_mean_rank": float(gpt_boot_ranks[:, entity_index].mean()),
                "gpt_rank_ci_low": float(gpt_rank_low),
                "gpt_rank_ci_high": float(gpt_rank_high),
                "deepseek_top1_probability": float(deepseek_top[:, entity_index].mean()),
                "gpt_top1_probability": float(gpt_top[:, entity_index].mean()),
                "deepseek_bottom1_probability": float(deepseek_bottom[:, entity_index].mean()),
                "gpt_bottom1_probability": float(gpt_bottom[:, entity_index].mean()),
            }
        )

    observed_top_deepseek, top_deepseek_set = _identity(
        deepseek_means, entity_ids, highest=True
    )
    observed_top_gpt, top_gpt_set = _identity(gpt_means, entity_ids, highest=True)
    observed_bottom_deepseek, bottom_deepseek_set = _identity(
        deepseek_means, entity_ids, highest=False
    )
    observed_bottom_gpt, bottom_gpt_set = _identity(gpt_means, entity_ids, highest=False)
    same_top_probability = float(np.sum(deepseek_top * gpt_top, axis=1).mean())
    same_bottom_probability = float(np.sum(deepseek_bottom * gpt_bottom, axis=1).mean())
    for method in ["pearson", "spearman", "kendall"]:
        estimate, p_value = _correlation(method, deepseek_means, gpt_means)
        boot = [
            value
            for deepseek_values, gpt_values in zip(deepseek_boot_means, gpt_boot_means)
            if (
                value := _correlation(method, deepseek_values, gpt_values)[0]
            )
            is not None
        ]
        valid_fraction = len(boot) / bootstrap_samples
        if boot and valid_fraction >= min_valid_fraction:
            ci_low, ci_high = np.percentile(np.asarray(boot), [2.5, 97.5])
            ci_low_value: float | None = float(ci_low)
            ci_high_value: float | None = float(ci_high)
        else:
            ci_low_value = ci_high_value = None
        rows.append(
            {
                "row_type": "summary",
                "level": level,
                "scope": f"{level}_mean_rank_stability",
                "method": method,
                "n": len(record_ids),
                "n_entities": len(entity_ids),
                "unit": "record_id",
                "seed": seed,
                "estimate": estimate,
                "effect": estimate,
                "effect_name": "correlation",
                "ci_low": ci_low_value,
                "ci_high": ci_high_value,
                "p_raw": p_value,
                "bootstrap_requested": bootstrap_samples,
                "bootstrap_valid": len(boot),
                "bootstrap_valid_fraction": valid_fraction,
                "minimum_valid_fraction": min_valid_fraction,
                "meets_min_valid_fraction": int(valid_fraction >= min_valid_fraction),
                "observed_top_deepseek": observed_top_deepseek,
                "observed_top_gpt": observed_top_gpt,
                "observed_top_identity_agreement": int(top_deepseek_set == top_gpt_set),
                "observed_bottom_deepseek": observed_bottom_deepseek,
                "observed_bottom_gpt": observed_bottom_gpt,
                "observed_bottom_identity_agreement": int(
                    bottom_deepseek_set == bottom_gpt_set
                ),
                "bootstrap_same_top_probability": same_top_probability,
                "bootstrap_same_bottom_probability": same_bottom_probability,
            }
        )
    return rows


def _rank_stability_rows(
    deepseek_index: dict[tuple[str, str], dict[str, Any]],
    gpt_index: dict[tuple[str, str], dict[str, Any]],
    *,
    rng: np.random.Generator,
    seed: int,
    bootstrap_samples: int,
    min_valid_fraction: float,
) -> list[dict[str, Any]]:
    record_ids = sorted({record_id for record_id, _ in deepseek_index})
    deepseek_system_matrix = np.asarray(
        [
            [_finite(deepseek_index[(rid, system)]["average_score"], "score") for system in ORIGINAL_SYSTEMS]
            for rid in record_ids
        ]
    )
    gpt_system_matrix = np.asarray(
        [
            [_finite(gpt_index[(rid, system)]["average_score"], "score") for system in ORIGINAL_SYSTEMS]
            for rid in record_ids
        ]
    )
    system_columns = {system: index for index, system in enumerate(ORIGINAL_SYSTEMS)}
    family_ids = list(RANK_FAMILIES)
    deepseek_family_matrix = np.column_stack(
        [
            deepseek_system_matrix[:, [system_columns[system] for system in systems]].mean(axis=1)
            for systems in RANK_FAMILIES.values()
        ]
    )
    gpt_family_matrix = np.column_stack(
        [
            gpt_system_matrix[:, [system_columns[system] for system in systems]].mean(axis=1)
            for systems in RANK_FAMILIES.values()
        ]
    )
    rows = _rank_level_rows(
        deepseek_system_matrix,
        gpt_system_matrix,
        record_ids,
        ORIGINAL_SYSTEMS,
        level="system",
        id_field="system_id",
        rng=rng,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        min_valid_fraction=min_valid_fraction,
    )
    rows.extend(
        _rank_level_rows(
            deepseek_family_matrix,
            gpt_family_matrix,
            record_ids,
            family_ids,
            level="family",
            id_field="family_id",
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            min_valid_fraction=min_valid_fraction,
        )
    )
    summary_rows = [row for row in rows if row["row_type"] == "summary"]
    _holm_rows(summary_rows)
    return rows


def analyze_crossjudge(
    deepseek_rows: list[dict[str, Any]],
    gpt_rows: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    *,
    expected_record_count: int = 20,
    seed: int = SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    permutation_samples: int = PERMUTATION_SAMPLES,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, list[dict[str, Any]]]:
    if expected_record_count != 20:
        raise ValueError("Judge-A must contain exactly 20 poems")
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError("bootstrap_samples and permutation_samples must be positive")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must lie in (0, 1]")
    deepseek_index = _judge_index(deepseek_rows, label="DeepSeek judge", expected_record_count=20)
    gpt_index = _judge_index(gpt_rows, label="GPT judge", expected_record_count=20)
    if set(deepseek_index) != set(gpt_index):
        raise ValueError("DeepSeek and GPT judge candidate sets differ")
    keys = sorted(deepseek_index)
    human, human_metadata = _human_composite(human_rows, expected_keys=set(keys))
    rng = np.random.default_rng(seed)

    deepseek_average = np.asarray([_finite(deepseek_index[key]["average_score"], "DeepSeek score") for key in keys])
    gpt_average = np.asarray([_finite(gpt_index[key]["average_score"], "GPT score") for key in keys])
    human_average = np.asarray([human[key] for key in keys])

    agreement = _correlation_rows(
        deepseek_average,
        gpt_average,
        keys,
        rng=rng,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        min_valid_fraction=min_valid_fraction,
        extra={"scope": "candidate_level_deepseek_vs_gpt"},
    )
    alignment: list[dict[str, Any]] = []
    for judge_name, judge_values in [("deepseek", deepseek_average), ("gpt", gpt_average)]:
        alignment.extend(
            _correlation_rows(
                judge_values,
                human_average,
                keys,
                rng=rng,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                min_valid_fraction=min_valid_fraction,
                extra={
                    "judge": judge_name,
                    "scope": "candidate_level_judge_vs_human",
                    **human_metadata,
                },
            )
        )
    _holm_rows(alignment)

    differences: list[dict[str, Any]] = []
    for method in ["pearson", "spearman", "kendall"]:
        deepseek_correlation = _correlation(method, deepseek_average, human_average)[0]
        gpt_correlation = _correlation(method, gpt_average, human_average)[0]
        ci_low, ci_high, p_value, bootstrap_valid = _bootstrap_correlation_difference(
            method,
            deepseek_average,
            gpt_average,
            human_average,
            keys,
            rng=rng,
            samples=bootstrap_samples,
        )
        valid_fraction = bootstrap_valid / bootstrap_samples
        if valid_fraction < min_valid_fraction:
            ci_low = ci_high = None
            p_value = None
        difference = (
            gpt_correlation - deepseek_correlation
            if gpt_correlation is not None and deepseek_correlation is not None
            else None
        )
        differences.append(
            {
                "method": method,
                "n": len(keys),
                "n_candidates": len(keys),
                "n_poems": len({key[0] for key in keys}),
                "unit": "candidate",
                "resampling_unit": "record_id",
                "seed": seed,
                "deepseek_human_correlation": deepseek_correlation,
                "gpt_human_correlation": gpt_correlation,
                "difference_gpt_minus_deepseek": difference,
                "effect": difference,
                "effect_name": "correlation_difference",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_raw": p_value,
                "bootstrap_requested": bootstrap_samples,
                "bootstrap_valid": bootstrap_valid,
                "bootstrap_valid_fraction": valid_fraction,
                "minimum_valid_fraction": min_valid_fraction,
                "meets_min_valid_fraction": int(valid_fraction >= min_valid_fraction),
                **human_metadata,
            }
        )
    _holm_rows(differences)

    self_preference = [
        _fixed_effect_preference_row(
            "average_score",
            deepseek_index,
            gpt_index,
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            min_valid_fraction=min_valid_fraction,
        ),
        _preference_row(
            "average_score",
            deepseek_index,
            gpt_index,
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
    ]
    self_preference[0]["p_holm"] = self_preference[0]["p_raw"]
    self_preference[1]["p_holm"] = self_preference[1]["p_raw"]
    dimension_sensitivity = [
        _fixed_effect_preference_row(
            dimension,
            deepseek_index,
            gpt_index,
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            min_valid_fraction=min_valid_fraction,
        )
        for dimension in JUDGE_DIMENSIONS
    ]
    _holm_rows(dimension_sensitivity)

    return {
        "agreement": agreement,
        "self_preference": self_preference,
        "judge_human_alignment": alignment,
        "correlation_difference_bootstrap": differences,
        "within_family": _within_family_rows(
            deepseek_index,
            gpt_index,
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "rank_stability": _rank_stability_rows(
            deepseek_index,
            gpt_index,
            rng=rng,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            min_valid_fraction=min_valid_fraction,
        ),
        "dimension_sensitivity": dimension_sensitivity,
    }


def atomic_write_csvs(outputs: Sequence[tuple[Path, list[dict[str, Any]]]]) -> None:
    resolved = [destination.resolve(strict=False) for destination, _ in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate resolved output destination")
    temporary: list[tuple[Path, Path, int]] = []
    states: list[tuple[Path, bool, Path]] = []
    try:
        for destination, rows in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            temporary.append((temp, destination, len(rows)))
            write_csv(temp, rows)
            with temp.open(encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            if len(written) != len(rows):
                raise ValueError(f"temporary CSV validation failed for {destination}")
        for _, destination, _ in temporary:
            existed = destination.exists()
            backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
            states.append((destination, existed, backup))
            if existed:
                os.replace(destination, backup)
        for temp, destination, _ in temporary:
            os.replace(temp, destination)
    except Exception:
        for destination, existed, backup in states:
            if backup.exists():
                destination.unlink(missing_ok=True)
                os.replace(backup, destination)
            elif not existed:
                destination.unlink(missing_ok=True)
        for temp, _, _ in temporary:
            temp.unlink(missing_ok=True)
        for _, _, backup in states:
            backup.unlink(missing_ok=True)
        raise
    for _, _, backup in states:
        backup.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepseek-scores", type=Path, required=True)
    parser.add_argument("--gpt-scores", type=Path, required=True)
    parser.add_argument("--human-eval-long", type=Path, required=True)
    for name in OUTPUT_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutation-samples", type=int, default=PERMUTATION_SAMPLES)
    parser.add_argument("--minimum-valid-fraction", type=float, default=MIN_VALID_FRACTION)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_crossjudge(
        read_rows(args.deepseek_scores),
        read_rows(args.gpt_scores),
        read_rows(args.human_eval_long),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        min_valid_fraction=args.minimum_valid_fraction,
    )
    if args.verify_only:
        print("verified cross-judge inputs: 20 poems x 7 systems x 2 judges; human panel aligned")
        return 0
    atomic_write_csvs(
        [(getattr(args, f"{name}_output"), result[name]) for name in OUTPUT_NAMES]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
