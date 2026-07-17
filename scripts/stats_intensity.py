#!/usr/bin/env python3
"""Paired dose-response statistics for the reasoning-intensity experiment.

All inference is paired by ``record_id``.  The script reads item-level files;
it never accepts manually entered aggregate results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats


SEED = 20260706
BOOTSTRAP_SAMPLES = 10_000
AUTO_METRICS = [
    "comet",
    "bertscore_f1",
    "sacrebleu_sentence",
    "chrfpp_sentence",
    "ter_sentence",
    "line_count_diff",
    "length_ratio",
]
JUDGE_DIMENSIONS = [
    "Semantic Fidelity",
    "Similarity to Reference",
    "Imagery and Rhetoric",
    "Thought and Emotion",
    "Lineation and Rhythm",
    "Modernity and Defamiliarization",
    "Cultural and Idiomatic Transfer",
    "Voice, Tone, and Style",
    "Poeticity",
    "English Naturalness",
    "Overall Impression",
]

DS_NON_THINKING = "deepseek_v4_flash_non_thinking"
DS_HIGH = "deepseek_v4_flash_thinking"
DS_MAX = "deepseek_v4_flash_thinking_max"
QWEN_NON_THINKING = "qwen36_plus_non_thinking"
QWEN_2048 = "qwen36_plus_thinking_b2048"
QWEN_4096 = "qwen36_plus_thinking_b4096"
QWEN_DEFAULT = "qwen36_plus_thinking"

COMPARISONS = [
    ("deepseek", "deepseek_nonthinking_to_high", DS_NON_THINKING, DS_HIGH),
    ("deepseek", "deepseek_high_to_max", DS_HIGH, DS_MAX),
    ("deepseek", "deepseek_nonthinking_to_max", DS_NON_THINKING, DS_MAX),
    ("qwen", "qwen_nonthinking_to_2048", QWEN_NON_THINKING, QWEN_2048),
    ("qwen", "qwen_nonthinking_to_4096", QWEN_NON_THINKING, QWEN_4096),
    ("qwen", "qwen_nonthinking_to_default", QWEN_NON_THINKING, QWEN_DEFAULT),
    ("qwen", "qwen_2048_to_4096", QWEN_2048, QWEN_4096),
    ("qwen", "qwen_4096_to_default", QWEN_4096, QWEN_DEFAULT),
    ("qwen", "qwen_2048_to_default", QWEN_2048, QWEN_DEFAULT),
]
REQUIRED_SYSTEMS = [DS_HIGH, DS_MAX, QWEN_2048, QWEN_4096, QWEN_DEFAULT]
QUALITY_SYSTEMS = [
    DS_NON_THINKING,
    DS_HIGH,
    DS_MAX,
    QWEN_NON_THINKING,
    QWEN_2048,
    QWEN_4096,
    QWEN_DEFAULT,
]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"input does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def read_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_rows(path)]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p values in the original order."""
    if not p_values:
        return []
    raw = np.asarray([_finite(value, "p-value") for value in p_values], dtype=float)
    if np.any((raw < 0) | (raw > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(raw, kind="stable")
    adjusted_sorted = np.empty(len(raw), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, (len(raw) - position) * raw[index])
        running = max(running, candidate)
        adjusted_sorted[position] = running
    adjusted = np.empty(len(raw), dtype=float)
    for position, index in enumerate(order):
        adjusted[index] = adjusted_sorted[position]
    return adjusted.tolist()


def _paired_bootstrap_mean(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    n = len(differences)
    indices = rng.integers(0, n, size=(samples, n))
    means = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _wilcoxon_p(differences: np.ndarray) -> float:
    if np.allclose(differences, 0.0, rtol=0.0, atol=0.0):
        return 1.0
    result = stats.wilcoxon(
        differences,
        zero_method="pratt",
        alternative="two-sided",
        method="auto",
    )
    p_value = float(result.pvalue)
    if not math.isfinite(p_value):
        raise ValueError("Wilcoxon returned a non-finite p-value")
    return p_value


def _cohen_dz(differences: np.ndarray) -> tuple[float | None, str]:
    if np.allclose(differences, 0.0, rtol=0.0, atol=0.0):
        return 0.0, "all_zero"
    sd = float(np.std(differences, ddof=1))
    if not math.isfinite(sd) or sd == 0:
        return None, "zero_variance_nonzero"
    return float(np.mean(differences) / sd), "ok"


def paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | str]:
    differences = right - left
    ci_low, ci_high = _paired_bootstrap_mean(differences, rng, bootstrap_samples)
    delta = float(np.mean(differences))
    direction = "right_higher" if delta > 0 else "right_lower" if delta < 0 else "tie"
    cohen_dz, dz_status = _cohen_dz(differences)
    return {
        "mean_left": float(np.mean(left)),
        "mean_right": float(np.mean(right)),
        "delta_right_minus_left": delta,
        "direction": direction,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": _wilcoxon_p(differences),
        "cohen_dz": cohen_dz,
        "dz_status": dz_status,
    }


def _index_unique(
    rows: Iterable[dict[str, Any]],
    *,
    systems: Sequence[str],
    expected_count: int,
    fields: Sequence[str],
    label: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    requested = set(systems)
    indexed: dict[str, dict[str, dict[str, Any]]] = {system: {} for system in systems}
    for row_number, row in enumerate(rows, start=1):
        system_id = str(row.get("system_id") or "")
        if system_id not in requested:
            continue
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"{label} row {row_number} has empty record_id")
        if record_id in indexed[system_id]:
            raise ValueError(f"{label} duplicate ({system_id}, {record_id})")
        status = row.get("status")
        if status not in (None, "", "ok"):
            raise ValueError(f"{label} non-ok status for ({system_id}, {record_id}): {status}")
        for field in fields:
            _finite(row.get(field), f"{label} {system_id}/{record_id}/{field}")
        indexed[system_id][record_id] = row
    for system_id in systems:
        actual = len(indexed[system_id])
        if actual != expected_count:
            raise ValueError(f"{label} {system_id}: expected {expected_count} records, found {actual}")
    reference_ids = set(indexed[systems[0]])
    for system_id in systems[1:]:
        if set(indexed[system_id]) != reference_ids:
            raise ValueError(f"{label} record_id set differs for {system_id}")
    return indexed


def _token_number(row: dict[str, Any], field: str) -> float:
    usage = row.get("token_usage")
    if not isinstance(usage, dict):
        raise ValueError(f"token_usage must be an object for {row.get('system_id')}/{row.get('record_id')}")
    if field == "reasoning_tokens":
        details = usage.get("completion_tokens_details")
        candidates = [
            details.get("reasoning_tokens") if isinstance(details, dict) else None,
            usage.get("reasoning_tokens"),
            row.get("reasoning_tokens"),
        ]
        value = next((item for item in candidates if item is not None), None)
    else:
        value = usage.get(field)
    return _finite(value, f"{row.get('system_id')}/{row.get('record_id')}/{field}")


def _resource_index(
    rows: Iterable[dict[str, Any]],
    *,
    expected_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {system: {} for system in REQUIRED_SYSTEMS}
    for row_number, row in enumerate(rows, start=1):
        system_id = str(row.get("system_id") or "")
        if system_id not in indexed:
            continue
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"outputs row {row_number} has empty record_id")
        if record_id in indexed[system_id]:
            raise ValueError(f"outputs duplicate ({system_id}, {record_id})")
        if row.get("status") != "ok":
            raise ValueError(f"outputs non-ok status for ({system_id}, {record_id})")
        _finite(row.get("latency_seconds"), f"outputs {system_id}/{record_id}/latency_seconds")
        for field in ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]:
            _token_number(row, field)
        indexed[system_id][record_id] = row
    for system_id in REQUIRED_SYSTEMS:
        if len(indexed[system_id]) != expected_count:
            raise ValueError(
                f"outputs {system_id}: expected {expected_count} records, found {len(indexed[system_id])}"
            )
    reference_ids = set(indexed[REQUIRED_SYSTEMS[0]])
    for system_id in REQUIRED_SYSTEMS[1:]:
        if set(indexed[system_id]) != reference_ids:
            raise ValueError(f"outputs record_id set differs for {system_id}")
    return indexed


def _resource_summary(
    final_indexed: dict[str, dict[str, dict[str, Any]]],
    attempt_rows: list[dict[str, Any]],
    *,
    attempt_scope: str,
) -> list[dict[str, Any]]:
    attempts_by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in REQUIRED_SYSTEMS}
    for row_number, row in enumerate(attempt_rows, start=1):
        system_id = str(row.get("system_id") or "")
        record_id = str(row.get("record_id") or "")
        status = str(row.get("status") or "")
        if not system_id:
            raise ValueError(f"attempt row {row_number} has empty system_id")
        if not record_id:
            raise ValueError(f"attempt row {row_number} has empty record_id")
        if not status:
            raise ValueError(f"attempt row {row_number} has empty status")
        if system_id not in attempts_by_system:
            continue
        if record_id not in final_indexed[system_id]:
            raise ValueError(f"attempt row {row_number} has unexpected record_id for {system_id}: {record_id}")
        if status == "ok":
            _finite(row.get("latency_seconds"), f"attempt {system_id}/{record_id}/latency_seconds")
            for field in ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]:
                _token_number(row, field)
        attempts_by_system[system_id].append(row)

    summaries: list[dict[str, Any]] = []
    for system_id in REQUIRED_SYSTEMS:
        attempts = attempts_by_system[system_id]
        if not attempts:
            raise ValueError(f"attempt data contains no rows for {system_id}")
        successful_rows = [row for row in attempts if row["status"] == "ok"]
        successful_ids = {str(row["record_id"]) for row in successful_rows}
        if successful_ids != set(final_indexed[system_id]):
            raise ValueError(f"successful attempt record set differs from final output for {system_id}")
        metrics = {
            "prompt_tokens": np.asarray([_token_number(row, "prompt_tokens") for row in successful_rows]),
            "completion_tokens": np.asarray([_token_number(row, "completion_tokens") for row in successful_rows]),
            "total_tokens": np.asarray([_token_number(row, "total_tokens") for row in successful_rows]),
            "reasoning_tokens": np.asarray([_token_number(row, "reasoning_tokens") for row in successful_rows]),
            "latency_seconds": np.asarray(
                [_finite(row["latency_seconds"], "latency") for row in successful_rows]
            ),
        }
        failure_count = len(attempts) - len(successful_rows)
        summary: dict[str, Any] = {
            "system_id": system_id,
            "n": len(successful_rows),
            "unit": "physical_attempt",
            "attempt_scope": attempt_scope,
            "resource_stat_scope": "successful_physical_attempts",
            "attempt_count": len(attempts),
            "successful_attempt_count": len(successful_rows),
            "success_count": len(successful_rows),
            "final_unique_success_count": len(final_indexed[system_id]),
            "failure_count": failure_count,
            "failure_rate": failure_count / len(attempts),
        }
        for name, values in metrics.items():
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_median"] = float(np.median(values))
            summary[f"{name}_total"] = float(np.sum(values))
        summaries.append(summary)
    return summaries


def analyze_intensity(
    original_score_rows: list[dict[str, Any]],
    supplemental_score_rows: list[dict[str, Any]],
    original_output_rows: list[dict[str, Any]],
    supplemental_output_rows: list[dict[str, Any]],
    judge_b_rows: list[dict[str, Any]],
    *,
    attempt_rows: list[dict[str, Any]] | None = None,
    expected_auto_count: int = 397,
    expected_judge_count: int = 20,
    seed: int = SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, list[dict[str, Any]]]:
    if expected_auto_count <= 0 or expected_judge_count <= 0:
        raise ValueError("expected counts must be positive")
    score_rows = original_score_rows + supplemental_score_rows
    auto = _index_unique(
        score_rows,
        systems=QUALITY_SYSTEMS,
        expected_count=expected_auto_count,
        fields=AUTO_METRICS,
        label="automatic scores",
    )
    judge_metrics = ["average_score", *JUDGE_DIMENSIONS]
    judge = _index_unique(
        judge_b_rows,
        systems=QUALITY_SYSTEMS,
        expected_count=expected_judge_count,
        fields=judge_metrics,
        label="Judge-B scores",
    )
    resources = _resource_index(
        original_output_rows + supplemental_output_rows,
        expected_count=expected_auto_count,
    )
    physical_attempts = (
        attempt_rows if attempt_rows is not None else original_output_rows + supplemental_output_rows
    )
    attempt_scope = "explicit_attempt_files" if attempt_rows is not None else "final_output_files"

    rng = np.random.default_rng(seed)
    paired: list[dict[str, Any]] = []
    for layer, indexed, metrics in [
        ("automatic", auto, AUTO_METRICS),
        ("judge_b", judge, judge_metrics),
    ]:
        for family, comparison, left_system, right_system in COMPARISONS:
            record_ids = sorted(indexed[left_system])
            if set(record_ids) != set(indexed[right_system]):
                raise ValueError(f"{layer} unpaired record sets for {comparison}")
            for metric in metrics:
                left = np.asarray(
                    [_finite(indexed[left_system][rid][metric], f"{comparison}/{metric}/left") for rid in record_ids]
                )
                right = np.asarray(
                    [_finite(indexed[right_system][rid][metric], f"{comparison}/{metric}/right") for rid in record_ids]
                )
                paired.append(
                    {
                        "layer": layer,
                        "family": family,
                        "comparison": comparison,
                        "metric": metric,
                        "left_system": left_system,
                        "right_system": right_system,
                        "n": len(record_ids),
                        "unit": "record_id",
                        "seed": seed,
                        **paired_summary(left, right, rng=rng, bootstrap_samples=bootstrap_samples),
                    }
                )
                paired[-1]["effect_name"] = "cohen_dz"
                paired[-1]["effect"] = paired[-1]["cohen_dz"]

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(paired):
        groups[(row["layer"], row["metric"])].append(index)
    for indices in groups.values():
        adjusted = holm_adjust([float(paired[index]["p_raw"]) for index in indices])
        for index, p_holm in zip(indices, adjusted):
            paired[index]["p_holm"] = p_holm

    return {
        "paired": paired,
        "resources": _resource_summary(resources, physical_attempts, attempt_scope=attempt_scope),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_csvs(outputs: Sequence[tuple[Path, list[dict[str, Any]]]]) -> None:
    """Transactionally replace a set of CSVs, restoring the complete old set on failure."""
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
    parser.add_argument("--original-scores", type=Path, required=True)
    parser.add_argument("--supplemental-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--original-outputs", type=Path, required=True)
    parser.add_argument("--supplemental-outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--attempts-path", type=Path, action="append", default=[])
    parser.add_argument("--judge-b-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--resources-output", type=Path, required=True)
    parser.add_argument("--expected-auto-count", type=int, default=397)
    parser.add_argument("--expected-judge-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_intensity(
        read_rows(args.original_scores),
        read_many(args.supplemental_scores),
        read_rows(args.original_outputs),
        read_many(args.supplemental_outputs),
        read_many(args.judge_b_scores),
        attempt_rows=read_many(args.attempts_path) if args.attempts_path else None,
        expected_auto_count=args.expected_auto_count,
        expected_judge_count=args.expected_judge_count,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.verify_only:
        print(
            f"verified intensity inputs: auto={args.expected_auto_count}/system, "
            f"judge={args.expected_judge_count}/system"
        )
        return 0
    atomic_write_csvs(
        [
            (args.paired_output, result["paired"]),
            (args.resources_output, result["resources"]),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
