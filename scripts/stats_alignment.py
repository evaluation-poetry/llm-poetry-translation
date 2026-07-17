# -*- coding: utf-8 -*-
"""Recompute metric–judge–human alignment from raw analysis inputs.

The v2 output is intentionally isolated from the legacy ``results/stats/*.csv``
tables.  Each row states its analysis unit and sample size and reports Pearson,
Spearman, and Kendall correlations whenever a correlation is defined.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stats" / "rebuttal"
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
AUTO_ITEM = ["comet", "bertscore_f1", "chrfpp_sentence", "ter_sentence"]
DIMS = ["MF", "IR", "EV", "LR", "MD", "PT"]
HUMAN_COMPOSITE_POLICY = (
    "mean of complete six-dimension annotator composites; at least 6 complete "
    "annotators required per record_id×system_id; incomplete annotator ratings omitted"
)
AUTO_SYSTEM = {
    "comet_mean": True,
    "bertscore_f1_mean": True,
    "sacrebleu_corpus": True,
    "chrfpp_corpus": True,
    "ter_corpus": False,
}
CORRELATION_COLUMNS = [
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
    "kendall_tau",
    "kendall_p",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human",
        default=str(ROOT / "results" / "stats" / "human_eval_long.csv"),
    )
    parser.add_argument(
        "--judge", default=str(ROOT / "results" / "judge_scores.jsonl")
    )
    parser.add_argument(
        "--auto-scores", default=str(ROOT / "results" / "full397_scores.jsonl")
    )
    parser.add_argument(
        "--auto-summary", default=str(ROOT / "results" / "full397_summary.csv")
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT_DIR / "stats_alignment_v2.csv")
    )
    return parser


def validate_unique_finite(
    frame: pd.DataFrame,
    keys: Sequence[str],
    numeric_columns: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in [*keys, *numeric_columns] if column not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    if frame.duplicated(list(keys), keep=False).any():
        examples = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)].head(3)
        raise ValueError(f"{label} contains duplicate pairings: {examples.to_dict('records')}")
    numeric = frame.loc[:, list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} numeric pairings must be finite")


def correlation_stats(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values):
        raise ValueError("correlation inputs must be one-dimensional paired arrays")
    if len(x_values) < 2:
        raise ValueError("correlation requires at least two paired observations")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("correlation inputs must be finite")
    pearson_r, pearson_p = st.pearsonr(x_values, y_values)
    spearman_rho, spearman_p = st.spearmanr(x_values, y_values)
    kendall_tau, kendall_p = st.kendalltau(x_values, y_values)
    return {
        "n": len(x_values),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
    }


def build_human_composites(human: pd.DataFrame) -> pd.DataFrame:
    keys = ["annotator", "record_id", "system_id"]
    duplicate_keys = [*keys, "dim"]
    if human.duplicated(duplicate_keys, keep=False).any():
        raise ValueError("human data contains duplicate dimension ratings")
    unknown = set(human["dim"].unique()) - set(DIMS)
    if unknown:
        raise ValueError(f"human data contains unknown dimensions: {sorted(unknown)}")
    annotator_composites = (
        human.groupby(keys, as_index=False)
        .agg(
            annotator_composite=("score", "mean"),
            n_dims=("dim", "size"),
            n_unique_dims=("dim", "nunique"),
        )
    )
    annotator_composites = annotator_composites.loc[
        (annotator_composites["n_dims"] == len(DIMS))
        & (annotator_composites["n_unique_dims"] == len(DIMS))
    ]
    all_pairs = human[["record_id", "system_id"]].drop_duplicates()
    composites = (
        annotator_composites.groupby(["record_id", "system_id"], as_index=False)
        .agg(
            human_composite=("annotator_composite", "mean"),
            n_complete_annotators=("annotator", "nunique"),
        )
    )
    composites = all_pairs.merge(
        composites,
        on=["record_id", "system_id"],
        how="left",
        validate="one_to_one",
    )
    insufficient = composites["n_complete_annotators"].fillna(0) < 6
    if insufficient.any():
        examples = composites.loc[
            insufficient, ["record_id", "system_id", "n_complete_annotators"]
        ].head(3)
        raise ValueError(
            "human composite requires at least 6 complete annotators per pairing: "
            f"{examples.to_dict('records')}"
        )
    return composites


def orient_auto_delta(
    metric: str, raw_delta: Iterable[float]
) -> tuple[np.ndarray, str, str]:
    values = np.asarray(list(raw_delta), dtype=float)
    raw_definition = f"{metric} raw T-minus-NT"
    if metric == "ter_sentence":
        return (
            -values,
            "lower_is_better; negated raw T-minus-NT",
            raw_definition,
        )
    return values, "higher_is_better; raw T-minus-NT", raw_definition


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _atomic_write_csv(frame: pd.DataFrame, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _row(
    analysis_type: str,
    unit: str,
    scope: str,
    x_name: str,
    y_name: str,
    x: Iterable[float],
    y: Iterable[float],
    family: str = "",
    orientation: str = "not_applicable",
    raw_delta: str = "not_applicable",
) -> dict:
    return {
        "analysis_type": analysis_type,
        "analysis_unit": unit,
        "scope": scope,
        "family": family,
        "x_variable": x_name,
        "y_variable": y_name,
        "orientation": orientation,
        "raw_delta": raw_delta,
        "human_composite_policy": HUMAN_COMPOSITE_POLICY,
        **correlation_stats(x, y),
    }


def compute_alignment(
    human_path: str | Path,
    judge_path: str | Path,
    auto_scores_path: str | Path,
    auto_summary_path: str | Path,
) -> pd.DataFrame:
    human = pd.read_csv(human_path)
    required_human = ["annotator", "record_id", "system_id", "dim", "score"]
    missing = [column for column in required_human if column not in human]
    if missing:
        raise ValueError(f"human data missing required columns: {missing}")
    human = human.loc[human["score"].notna() & (human["score"] != "")].copy()
    human["score"] = pd.to_numeric(human["score"], errors="raise")
    human_keys = ["annotator", "record_id", "system_id", "dim"]
    validate_unique_finite(human, human_keys, ["score"], "human data")
    hcomp = build_human_composites(human)
    validate_unique_finite(
        hcomp,
        ["record_id", "system_id"],
        ["human_composite", "n_complete_annotators"],
        "human composite",
    )

    judge_rows = _read_jsonl(judge_path)
    judge = pd.DataFrame(
        [
            {
                "record_id": row["record_id"],
                "system_id": row["system_id"],
                "judge_avg": row["average_score"],
            }
            for row in judge_rows
        ]
    )
    validate_unique_finite(
        judge, ["record_id", "system_id"], ["judge_avg"], "judge data"
    )

    auto_rows = _read_jsonl(auto_scores_path)
    auto = pd.DataFrame(
        [
            {
                "record_id": row["record_id"],
                "system_id": row["system_id"],
                **{metric: row[metric] for metric in AUTO_ITEM},
            }
            for row in auto_rows
        ]
    )
    validate_unique_finite(
        auto, ["record_id", "system_id"], AUTO_ITEM, "automatic metric data"
    )

    summary = pd.read_csv(auto_summary_path)
    validate_unique_finite(
        summary, ["system_id"], list(AUTO_SYSTEM), "automatic metric summary"
    )

    output: list[dict] = []
    item = hcomp.merge(
        judge, on=["record_id", "system_id"], how="inner", validate="one_to_one"
    )
    if item.empty:
        raise ValueError("human and judge data have no shared item/system pairings")
    output.append(
        _row(
            "item_pooled",
            "record_id×system_id",
            "all_shared_items",
            "human_composite",
            "judge_average_score",
            item["human_composite"],
            item["judge_avg"],
        )
    )
    for system_id, group in item.groupby("system_id", sort=True):
        output.append(
            _row(
                "item_per_system",
                "record_id",
                str(system_id),
                "human_composite",
                "judge_average_score",
                group["human_composite"],
                group["judge_avg"],
            )
        )

    human_wide = hcomp.pivot(index="record_id", columns="system_id", values="human_composite")
    judge_wide = judge.pivot(index="record_id", columns="system_id", values="judge_avg")
    auto_wide = auto.pivot(index="record_id", columns="system_id", values=AUTO_ITEM)
    for family, (nonthinking, thinking) in FAMILIES.items():
        required_systems = {nonthinking, thinking}
        if not required_systems.issubset(human_wide.columns):
            raise ValueError(f"human data missing systems for family {family}")
        if not required_systems.issubset(judge_wide.columns):
            raise ValueError(f"judge data missing systems for family {family}")
        human_delta = (human_wide[thinking] - human_wide[nonthinking]).rename(
            "human_delta"
        )
        targets: dict[str, tuple[pd.Series, str, str]] = {
            "judge_delta": (
                judge_wide[thinking] - judge_wide[nonthinking],
                "higher_is_better; raw T-minus-NT",
                "judge_avg raw T-minus-NT",
            )
        }
        for metric in AUTO_ITEM:
            raw_delta = auto_wide[metric][thinking] - auto_wide[metric][nonthinking]
            oriented, orientation, raw_definition = orient_auto_delta(
                metric, raw_delta
            )
            name = (
                f"{metric}_improvement_delta"
                if metric == "ter_sentence"
                else f"{metric}_delta"
            )
            targets[name] = (
                pd.Series(oriented, index=raw_delta.index, name=name),
                orientation,
                raw_definition,
            )
        for target_name, (target, orientation, raw_definition) in targets.items():
            pairs = pd.concat([human_delta, target.rename(target_name)], axis=1).dropna()
            validate_unique_finite(
                pairs.reset_index(), ["record_id"], ["human_delta", target_name],
                f"paired delta {family} {target_name}",
            )
            output.append(
                _row(
                    "paired_delta",
                    "record_id paired T−NT delta",
                    target_name,
                    "human_composite_delta",
                    target_name,
                    pairs["human_delta"],
                    pairs[target_name],
                    family=family,
                    orientation=orientation,
                    raw_delta=raw_definition,
                )
            )

    human_system = hcomp.groupby("system_id")["human_composite"].mean()
    judge_system = judge.groupby("system_id")["judge_avg"].mean()
    summary_indexed = summary.set_index("system_id")
    for metric, higher_better in AUTO_SYSTEM.items():
        metric_values = pd.to_numeric(summary_indexed[metric], errors="raise")
        if not higher_better:
            metric_values = -metric_values
        for target_name, target in [
            ("human_system_mean", human_system),
            ("judge_system_mean", judge_system),
        ]:
            common = metric_values.index.intersection(target.index, sort=False)
            pairs = pd.DataFrame(
                {"metric": metric_values.loc[common], "target": target.loc[common]}
            )
            validate_unique_finite(
                pairs.reset_index(), ["system_id"], ["metric", "target"],
                f"system ranking {metric} {target_name}",
            )
            output.append(
                _row(
                    "system_ranking",
                    "system_id",
                    metric,
                    f"{metric}_higher_is_better",
                    target_name,
                    pairs["metric"],
                    pairs["target"],
                    orientation=(
                        "higher_is_better"
                        if higher_better
                        else "lower_is_better; metric negated before correlation"
                    ),
                    raw_delta="not_applicable",
                )
            )
    common = judge_system.index.intersection(human_system.index, sort=False)
    output.append(
        _row(
            "system_ranking",
            "system_id",
            "judge_vs_human",
            "judge_system_mean",
            "human_system_mean",
            judge_system.loc[common],
            human_system.loc[common],
        )
    )
    columns = [
        "analysis_type",
        "analysis_unit",
        "scope",
        "family",
        "x_variable",
        "y_variable",
        "orientation",
        "raw_delta",
        "human_composite_policy",
        "n",
        *CORRELATION_COLUMNS,
    ]
    return pd.DataFrame(output, columns=columns)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = compute_alignment(
        args.human, args.judge, args.auto_scores, args.auto_summary
    )
    _atomic_write_csv(output, args.output)
    print(f"wrote {len(output)} alignment rows to {args.output}")


if __name__ == "__main__":
    main()
