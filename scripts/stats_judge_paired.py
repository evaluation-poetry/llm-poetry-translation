# -*- coding: utf-8 -*-
"""Paired LLM-as-judge tests and paired-design sensitivity MDEs (v2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:  # supports both ``python scripts/...`` and import as ``scripts...``
    from stats_human_paired import (
        atomic_write_csv,
        holm_adjust,
        mde_row,
        merge_mde_layers,
        paired_wilcoxon,
        reject_duplicate_destinations,
        write_mde_layer,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by module imports in tests
    from scripts.stats_human_paired import (
        atomic_write_csv,
        holm_adjust,
        mde_row,
        merge_mde_layers,
        paired_wilcoxon,
        reject_duplicate_destinations,
        write_mde_layer,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stats" / "rebuttal"
SEED = 20260706
FAMILIES = {
    "deepseek_v4_flash": (
        "deepseek_v4_flash_non_thinking",
        "deepseek_v4_flash_thinking",
    ),
    "qwen36_plus": ("qwen36_plus_non_thinking", "qwen36_plus_thinking"),
    "claude_sonnet46": (
        "claude_sonnet46_non_thinking",
        "claude_sonnet46_thinking",
    ),
}
FIELDS = [("average_score", True), ("rank_position", False)]
N_DECLARED_JUDGE_TESTS = len(FAMILIES) * len(FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge-mde-only",
        action="store_true",
        help="atomically merge current human and judge MDE layer files, then exit",
    )
    parser.add_argument(
        "--input", default=str(ROOT / "results" / "judge_scores.jsonl")
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT_DIR / "judge_paired_v2.csv")
    )
    parser.add_argument(
        "--mde-layer-output", default=str(DEFAULT_OUT_DIR / "mde_judge.csv")
    )
    parser.add_argument(
        "--human-mde-input", default=str(DEFAULT_OUT_DIR / "mde_human.csv")
    )
    parser.add_argument("--mde-output", default=str(DEFAULT_OUT_DIR / "mde.csv"))
    return parser


def _load_rows(path: str | Path) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {}
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            key = (str(row["system_id"]), str(row["record_id"]))
            if key in seen:
                raise ValueError(f"judge data contains duplicate pairing {key}")
            seen.add(key)
            values = np.asarray([row[field] for field, _ in FIELDS], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"judge data contains non-finite values at {key}")
            rows.setdefault(key[0], {})[key[1]] = row
    return rows


def compute_paired(rows: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SEED)
    output: list[dict] = []
    mde_rows: list[dict] = []
    for family, (nonthinking, thinking) in FAMILIES.items():
        if nonthinking not in rows or thinking not in rows:
            raise ValueError(f"judge data missing NT/T systems for {family}")
        record_ids = sorted(set(rows[nonthinking]) & set(rows[thinking]))
        if len(record_ids) != 20:
            raise ValueError(f"paired judge test requires 20 poems for {family}; got {len(record_ids)}")
        for field, higher_better in FIELDS:
            nonthinking_values = np.array(
                [float(rows[nonthinking][record_id][field]) for record_id in record_ids]
            )
            thinking_values = np.array(
                [float(rows[thinking][record_id][field]) for record_id in record_ids]
            )
            differences = thinking_values - nonthinking_values
            if not np.isfinite(differences).all():
                raise ValueError(f"non-finite judge differences for {family} {field}")
            wilcoxon_p, wilcoxon_method = paired_wilcoxon(differences)
            boot_means = np.array(
                [
                    differences[
                        rng.integers(0, len(differences), len(differences))
                    ].mean()
                    for _ in range(10000)
                ]
            )
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            paired_sd = differences.std(ddof=1)
            dz = differences.mean() / paired_sd if paired_sd > 0 else 0.0
            output.append(
                {
                    "family": family,
                    "field": field,
                    "higher_better": "yes" if higher_better else "no",
                    "analysis_unit": "record_id paired T−NT",
                    "n": len(differences),
                    "mean_nt": nonthinking_values.mean(),
                    "mean_t": thinking_values.mean(),
                    "delta_t_minus_nt": differences.mean(),
                    "ci_lo": float(ci_low),
                    "ci_hi": float(ci_high),
                    "wilcoxon_exact_p": float(wilcoxon_p),
                    "wilcoxon_method": wilcoxon_method,
                    "cohen_dz": float(dz),
                }
            )
            for alpha in [0.05, 0.05 / N_DECLARED_JUDGE_TESTS]:
                mde_rows.append(
                    mde_row(
                        layer="judge",
                        family=family,
                        measure=field,
                        differences=differences,
                        n_tests=N_DECLARED_JUDGE_TESTS,
                        alpha=alpha,
                    )
                )
    adjusted = holm_adjust(
        [row["wilcoxon_exact_p"] for row in output],
        family_size=N_DECLARED_JUDGE_TESTS,
    )
    for row, adjusted_p in zip(output, adjusted, strict=True):
        row["wilcoxon_p_holm"] = adjusted_p
        row["holm_family_n"] = N_DECLARED_JUDGE_TESTS
    return output, mde_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.merge_mde_only:
        reject_duplicate_destinations(
            args.human_mde_input, args.mde_layer_output, args.mde_output
        )
        merge_mde_layers(
            args.human_mde_input, args.mde_layer_output, args.mde_output
        )
        print(f"merged current human and judge MDE layers to {args.mde_output}")
        return
    reject_duplicate_destinations(args.output, args.mde_layer_output)
    rows = _load_rows(args.input)
    paired_rows, mde_rows = compute_paired(rows)
    atomic_write_csv(pd.DataFrame(paired_rows), args.output)
    write_mde_layer(args.mde_layer_output, mde_rows)
    print(f"wrote {len(paired_rows)} judge paired rows to {args.output}")
    print(f"wrote {len(mde_rows)} judge MDE rows to {args.mde_layer_output}")


if __name__ == "__main__":
    main()
