#!/usr/bin/env python3
"""Paired statistical tests for the open Qwen3.5-9B mode comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats

try:
    from scripts.stats_intensity import (
        _cohen_dz,
        _index_unique,
        _paired_bootstrap_mean,
        holm_adjust,
        read_rows,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from stats_intensity import (  # type: ignore[no-redef]
        _cohen_dz,
        _index_unique,
        _paired_bootstrap_mean,
        holm_adjust,
        read_rows,
    )


NON_THINKING = "qwen35_9b_open_non_thinking"
THINKING = "qwen35_9b_open_thinking"
SEED = 20260716
BOOTSTRAP_SAMPLES = 10_000


def _identity(value: float) -> float:
    return value


def _length_deviation(value: float) -> float:
    return abs(value - 1.0)


AUTOMATIC_METRICS: dict[str, tuple[str, Callable[[float], float]]] = {
    "comet": ("comet", _identity),
    "bertscore_f1": ("bertscore_f1", _identity),
    "sacrebleu_sentence": ("sacrebleu_sentence", _identity),
    "chrfpp_sentence": ("chrfpp_sentence", _identity),
    "ter_sentence": ("ter_sentence", _identity),
    "line_count_diff": ("line_count_diff", _identity),
    "length_ratio_abs_deviation": ("length_ratio", _length_deviation),
}
JUDGE_METRICS = ("average_score", "rank_position")


def _paired_arrays(
    indexed: dict[str, dict[str, dict[str, Any]]],
    field: str,
    transform: Callable[[float], float] = _identity,
) -> tuple[np.ndarray, np.ndarray]:
    record_ids = sorted(indexed[NON_THINKING])
    left = np.asarray(
        [transform(float(indexed[NON_THINKING][record_id][field])) for record_id in record_ids],
        dtype=float,
    )
    right = np.asarray(
        [transform(float(indexed[THINKING][record_id][field])) for record_id in record_ids],
        dtype=float,
    )
    return left, right


def _paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | str | None]:
    differences = right - left
    ci_low, ci_high = _paired_bootstrap_mean(differences, rng, bootstrap_samples)
    delta = float(np.mean(differences))
    if np.allclose(differences, 0.0, rtol=0.0, atol=0.0):
        p_raw = 1.0
    else:
        p_raw = float(
            stats.wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
    cohen_dz, dz_status = _cohen_dz(differences)
    return {
        "mean_left": float(np.mean(left)),
        "mean_right": float(np.mean(right)),
        "delta_right_minus_left": delta,
        "direction": "right_higher" if delta > 0 else "right_lower" if delta < 0 else "tie",
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": p_raw,
        "cohen_dz": cohen_dz,
        "dz_status": dz_status,
    }


def analyze(
    automatic_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    *,
    seed: int = SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    expected_automatic_count: int = 397,
    expected_judge_count: int = 20,
) -> list[dict[str, Any]]:
    systems = [NON_THINKING, THINKING]
    automatic = _index_unique(
        automatic_rows,
        systems=systems,
        expected_count=expected_automatic_count,
        fields=sorted({field for field, _ in AUTOMATIC_METRICS.values()}),
        label="open-Qwen automatic scores",
    )
    judge = _index_unique(
        judge_rows,
        systems=systems,
        expected_count=expected_judge_count,
        fields=JUDGE_METRICS,
        label="open-Qwen judge scores",
    )
    automatic_rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    automatic_rows_out: list[dict[str, Any]] = []
    for metric, (field, transform) in AUTOMATIC_METRICS.items():
        left, right = _paired_arrays(automatic, field, transform)
        summary = _paired_summary(
            left,
            right,
            rng=automatic_rng,
            bootstrap_samples=bootstrap_samples,
        )
        automatic_rows_out.append(
            {
                "layer": "automatic",
                "metric": metric,
                "left_system": NON_THINKING,
                "right_system": THINKING,
                "n": expected_automatic_count,
                "unit": "record_id paired thinking-minus-non-thinking",
                "seed": seed,
                "bootstrap_samples": bootstrap_samples,
                "mean_nt": summary["mean_left"],
                "mean_t": summary["mean_right"],
                "delta_t_minus_nt": summary["delta_right_minus_left"],
                "direction": summary["direction"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "p_raw": summary["p_raw"],
                "cohen_dz": summary["cohen_dz"],
                "dz_status": summary["dz_status"],
            }
        )
    adjusted = holm_adjust([float(row["p_raw"]) for row in automatic_rows_out])
    for row, p_holm in zip(automatic_rows_out, adjusted, strict=True):
        row["p_holm"] = p_holm
    rows.extend(automatic_rows_out)

    judge_rng = np.random.default_rng(seed)
    for metric in JUDGE_METRICS:
        left, right = _paired_arrays(judge, metric)
        summary = _paired_summary(
            left,
            right,
            rng=judge_rng,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append(
            {
                "layer": "judge",
                "metric": metric,
                "left_system": NON_THINKING,
                "right_system": THINKING,
                "n": expected_judge_count,
                "unit": "record_id paired thinking-minus-non-thinking",
                "seed": seed,
                "bootstrap_samples": bootstrap_samples,
                "mean_nt": summary["mean_left"],
                "mean_t": summary["mean_right"],
                "delta_t_minus_nt": summary["delta_right_minus_left"],
                "direction": summary["direction"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "p_raw": summary["p_raw"],
                "p_holm": "",
                "cohen_dz": summary["cohen_dz"],
                "dz_status": summary["dz_status"],
            }
        )
    return rows


def exclusion_sensitivity(
    automatic_rows: list[dict[str, Any]],
    *,
    excluded_record_id: str,
    expected_automatic_count: int = 397,
) -> list[dict[str, Any]]:
    systems = [NON_THINKING, THINKING]
    automatic = _index_unique(
        automatic_rows,
        systems=systems,
        expected_count=expected_automatic_count,
        fields=("comet", "bertscore_f1"),
        label="open-Qwen automatic scores",
    )
    if excluded_record_id not in automatic[NON_THINKING]:
        raise ValueError("excluded record is absent from the paired automatic scores")
    rows: list[dict[str, Any]] = []
    for metric in ("comet", "bertscore_f1"):
        left, right = _paired_arrays(automatic, metric)
        kept_ids = [
            record_id
            for record_id in sorted(automatic[NON_THINKING])
            if record_id != excluded_record_id
        ]
        left_kept = np.asarray(
            [float(automatic[NON_THINKING][record_id][metric]) for record_id in kept_ids]
        )
        right_kept = np.asarray(
            [float(automatic[THINKING][record_id][metric]) for record_id in kept_ids]
        )
        left_shift = float(left_kept.mean() - left.mean())
        right_shift = float(right_kept.mean() - right.mean())
        rows.append(
            {
                "analysis": "single_record_exclusion",
                "metric": metric,
                "n_full": expected_automatic_count,
                "n_excluding": expected_automatic_count - 1,
                "excluded_n": 1,
                "nt_mean_shift": left_shift,
                "t_mean_shift": right_shift,
                "max_abs_system_mean_shift": max(abs(left_shift), abs(right_shift)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic-scores", required=True, type=Path)
    parser.add_argument("--judge-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--excluded-record-id")
    parser.add_argument("--sensitivity-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    automatic_rows = read_rows(args.automatic_scores)
    judge_rows = read_rows(args.judge_scores)
    rows = analyze(
        automatic_rows,
        judge_rows,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    _write_csv(args.output, rows)
    if bool(args.excluded_record_id) != bool(args.sensitivity_output):
        raise ValueError("--excluded-record-id and --sensitivity-output must be used together")
    if args.excluded_record_id:
        sensitivity = exclusion_sensitivity(
            automatic_rows,
            excluded_record_id=args.excluded_record_id,
        )
        _write_csv(args.sensitivity_output, sensitivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
