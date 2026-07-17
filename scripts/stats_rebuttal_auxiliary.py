# -*- coding: utf-8 -*-
"""Aggregate-only sensitivity analyses for the supplemental evaluation.

Inputs are the existing human, full-397 automatic-score, and original-judge
artifacts.  Outputs contain no poem text, translation text, record identifiers,
or candidate mappings.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "stats" / "rebuttal"
SEED = 20260706
N_RESAMPLES = 10000
WORD_RE = re.compile(r"[A-Za-z']+")
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
LARGE_HUMAN_SOURCES = {"21C", "PIC"}
LARGE_SOURCE_IDS = {
    "modern_chinese_poetry",
    "poetry_international_chinese",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human",
        default=str(ROOT / "results" / "stats" / "human_eval_long.csv"),
    )
    parser.add_argument(
        "--auto", default=str(ROOT / "results" / "full397_scores.jsonl")
    )
    parser.add_argument(
        "--judge", default=str(ROOT / "results" / "judge_scores.jsonl")
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    return parser


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if frame.duplicated(list(columns), keep=False).any():
        raise ValueError(f"{label} contains duplicate rows for {list(columns)}")


def _numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} column {column} must be finite")
    return values


def count_words(text: object) -> int:
    return len(WORD_RE.findall(text if isinstance(text, str) else ""))


def _human_candidates(human: pd.DataFrame) -> pd.DataFrame:
    required = ["annotator", "record_id", "source", "system_id", "dim", "score"]
    _require_columns(human, required, "human data")
    human = human.loc[human["score"].notna() & (human["score"] != "")].copy()
    _require_unique(
        human,
        ["annotator", "record_id", "system_id", "dim"],
        "human data",
    )
    human["score"] = _numeric(human, "score", "human data")
    if (human.groupby("record_id")["source"].nunique() != 1).any():
        raise ValueError("human data must have one source per record_id")
    return (
        human.groupby(["record_id", "source", "system_id"], as_index=False)
        .agg(score=("score", "mean"))
        .reset_index(drop=True)
    )


def _judge_candidates(judge: pd.DataFrame) -> pd.DataFrame:
    required = ["record_id", "source_id", "system_id", "average_score"]
    _require_columns(judge, required, "judge data")
    judge = judge.loc[:, required].copy()
    _require_unique(judge, ["record_id", "system_id"], "judge data")
    judge["score"] = _numeric(judge, "average_score", "judge data")
    if (judge.groupby("record_id")["source_id"].nunique() != 1).any():
        raise ValueError("judge data must have one source per record_id")
    return judge.drop(columns="average_score")


def _auto_candidates(auto: pd.DataFrame) -> pd.DataFrame:
    required = ["record_id", "source_id", "system_id", "comet", "hypothesis"]
    _require_columns(auto, required, "automatic-score data")
    auto = auto.loc[:, required].copy()
    _require_unique(auto, ["record_id", "system_id"], "automatic-score data")
    auto["comet"] = _numeric(auto, "comet", "automatic-score data")
    auto["word_count"] = auto["hypothesis"].map(count_words)
    if (auto["word_count"] <= 0).any():
        raise ValueError("automatic-score data contain an empty English candidate")
    if (auto.groupby("record_id")["source_id"].nunique() != 1).any():
        raise ValueError("automatic-score data must have one source per record_id")
    return auto.drop(columns="hypothesis")


def _meta(
    *, n: int, unit: str, seed: int | str, input_sha256: str, method: str
) -> dict[str, object]:
    del input_sha256
    return {
        "n": int(n),
        "unit": unit,
        "seed": seed,
        "method": method,
    }


def build_annotator_direction(
    human: pd.DataFrame, *, input_sha256: str
) -> pd.DataFrame:
    required = ["annotator", "record_id", "system_id", "dim", "score"]
    _require_columns(human, required, "human data")
    human = human.loc[human["score"].notna() & (human["score"] != "")].copy()
    _require_unique(
        human,
        ["annotator", "record_id", "system_id", "dim"],
        "human data",
    )
    human["score"] = _numeric(human, "score", "human data")
    candidate = (
        human.groupby(["annotator", "record_id", "system_id"], as_index=False)
        .agg(score=("score", "mean"))
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for family, (nonthinking, thinking) in FAMILIES.items():
        subset = candidate.loc[
            candidate["system_id"].isin([nonthinking, thinking])
        ]
        wide = subset.pivot(
            index=["annotator", "record_id"], columns="system_id", values="score"
        )
        if not {nonthinking, thinking}.issubset(wide.columns):
            raise ValueError(f"human data missing NT/T systems for {family}")
        if wide[[nonthinking, thinking]].isna().any().any():
            raise ValueError(f"human data have incomplete NT/T pairs for {family}")
        for annotator, paired in wide.groupby(level="annotator", sort=True):
            paired = paired.droplevel("annotator")
            delta = paired[nonthinking] - paired[thinking]
            rows.append(
                {
                    "analysis": "annotator_family_direction",
                    "annotator": str(annotator),
                    "family": family,
                    "mean_nt": float(paired[nonthinking].mean()),
                    "mean_t": float(paired[thinking].mean()),
                    "delta_nt_minus_t": float(delta.mean()),
                    "nt_gt_t": bool(delta.mean() > 0),
                    **_meta(
                        n=len(paired),
                        unit=(
                            "record_id paired NT-minus-T after per-candidate "
                            "dimension mean"
                        ),
                        seed="not_applicable",
                        input_sha256=input_sha256,
                        method=(
                            "Within each annotator and family, average dimensions per "
                            "candidate, pair NT/T by record_id, then equal-weight records."
                        ),
                    ),
                }
            )
    output = pd.DataFrame(rows)
    summary = (
        output.groupby("family", as_index=False)
        .agg(
            family_nt_gt_t_annotators=("nt_gt_t", "sum"),
            family_total_annotators=("annotator", "nunique"),
            family_delta_min=("delta_nt_minus_t", "min"),
            family_delta_max=("delta_nt_minus_t", "max"),
        )
        .reset_index(drop=True)
    )
    return output.merge(summary, on="family", how="left", validate="many_to_one")


def _paired_family(
    frame: pd.DataFrame,
    family: str,
    nonthinking: str,
    thinking: str,
    value: str,
    index: Sequence[str],
) -> pd.DataFrame:
    subset = frame.loc[frame["system_id"].isin([nonthinking, thinking])]
    wide = subset.pivot(index=list(index), columns="system_id", values=value)
    if not {nonthinking, thinking}.issubset(wide.columns):
        raise ValueError(f"data missing NT/T systems for {family}")
    if wide[[nonthinking, thinking]].isna().any().any():
        raise ValueError(f"data have incomplete NT/T pairs for {family}")
    return wide[[nonthinking, thinking]].rename(
        columns={nonthinking: "nt", thinking: "t"}
    )


def build_panel_representativeness(
    auto: pd.DataFrame,
    panel_ids: set[str],
    *,
    seed: int,
    n_resamples: int,
    input_sha256: str,
) -> pd.DataFrame:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    auto = _auto_candidates(auto)
    rows: list[dict[str, object]] = []
    child_seeds = np.random.SeedSequence(seed).spawn(len(FAMILIES))
    for (family, (nonthinking, thinking)), child_seed in zip(
        FAMILIES.items(), child_seeds, strict=True
    ):
        paired = _paired_family(
            auto,
            family,
            nonthinking,
            thinking,
            "comet",
            ["record_id", "source_id"],
        ).reset_index()
        paired["delta"] = paired["nt"] - paired["t"]
        panel = paired.loc[paired["record_id"].isin(panel_ids)].copy()
        if set(panel["record_id"]) != set(panel_ids):
            raise ValueError(f"panel record_ids missing from automatic scores for {family}")
        source_counts = panel.groupby("source_id").size().sort_index()
        if source_counts.empty:
            raise ValueError("panel must contain at least one record")
        rng = np.random.default_rng(child_seed)
        resampled = np.empty(n_resamples, dtype=float)
        by_source: dict[str, np.ndarray] = {}
        for source_id, sample_n in source_counts.items():
            values = paired.loc[paired["source_id"] == source_id, "delta"].to_numpy(
                dtype=float
            )
            if sample_n > len(values):
                raise ValueError(f"panel source {source_id} exceeds full-source size")
            by_source[str(source_id)] = values
        for iteration in range(n_resamples):
            draw = [
                rng.choice(by_source[source_id], size=int(sample_n), replace=False)
                for source_id, sample_n in source_counts.items()
            ]
            resampled[iteration] = float(np.concatenate(draw).mean())
        panel_delta = float(panel["delta"].mean())
        expected_delta = float(
            sum(
                paired.loc[paired["source_id"] == source_id, "delta"].mean()
                * int(sample_n)
                for source_id, sample_n in source_counts.items()
            )
            / len(panel)
        )
        observed_deviation = abs(panel_delta - expected_delta)
        p_two_sided = float(
            (1 + np.count_nonzero(np.abs(resampled - expected_delta) >= observed_deviation))
            / (n_resamples + 1)
        )
        ci_low, ci_high = np.percentile(resampled, [2.5, 97.5])
        rows.append(
            {
                "analysis": "panel_vs_full397_comet",
                "family": family,
                "mean_nt_panel": float(panel["nt"].mean()),
                "mean_t_panel": float(panel["t"].mean()),
                "panel_delta_nt_minus_t": panel_delta,
                "full_delta_nt_minus_t": float(paired["delta"].mean()),
                "panel_minus_full_delta": panel_delta - float(paired["delta"].mean()),
                "source_weighted_expected_delta": expected_delta,
                "resample_mean": float(resampled.mean()),
                "resample_ci_low": float(ci_low),
                "resample_ci_high": float(ci_high),
                "resample_p_two_sided": p_two_sided,
                "panel_inside_resample_95pct_interval": bool(
                    ci_low <= panel_delta <= ci_high
                ),
                "n_full": len(paired),
                "n_resamples": n_resamples,
                "panel_source_counts": ";".join(
                    f"{source_id}:{int(sample_n)}"
                    for source_id, sample_n in source_counts.items()
                ),
                **_meta(
                    n=len(panel),
                    unit="record_id paired NT-minus-T COMET",
                    seed=seed,
                    input_sha256=input_sha256,
                    method=(
                        "Compare the observed panel mean with the unweighted full-corpus "
                        "mean; assess panel deviation by fixed-seed sampling without "
                        "replacement within each source using the observed panel source "
                        "counts. The empirical two-sided p is descriptive, centered on "
                        "the source-weighted resampling expectation."
                    ),
                ),
            }
        )
    return pd.DataFrame(rows)


def _exclusion_rows(
    frame: pd.DataFrame,
    *,
    layer: str,
    source_column: str,
    sources: set[str],
    input_sha256: str,
    method_prefix: str,
) -> list[dict[str, object]]:
    frame = frame.loc[frame[source_column].isin(sources)]
    rows: list[dict[str, object]] = []
    for family, (nonthinking, thinking) in FAMILIES.items():
        paired = _paired_family(
            frame,
            family,
            nonthinking,
            thinking,
            "score",
            ["record_id"],
        )
        delta = paired["nt"] - paired["t"]
        rows.append(
            {
                "analysis": "exclude_rpr_brln",
                "layer": layer,
                "family": family,
                "sources_kept": "21C;PIC",
                "mean_nt": float(paired["nt"].mean()),
                "mean_t": float(paired["t"].mean()),
                "delta_nt_minus_t": float(delta.mean()),
                "nt_gt_t": bool(delta.mean() > 0),
                **_meta(
                    n=len(paired),
                    unit="record_id paired NT-minus-T",
                    seed="not_applicable",
                    input_sha256=input_sha256,
                    method=(
                        f"{method_prefix}; exclude RPR and BRLN, pair NT/T by "
                        "record_id, then equal-weight retained 21C and PIC records."
                    ),
                ),
            }
        )
    return rows


def build_small_source_exclusion(
    human: pd.DataFrame,
    judge: pd.DataFrame,
    *,
    human_input_sha256: str,
    judge_input_sha256: str,
) -> pd.DataFrame:
    human_candidates = _human_candidates(human)
    judge_candidates = _judge_candidates(judge)
    rows = _exclusion_rows(
        human_candidates,
        layer="human",
        source_column="source",
        sources=LARGE_HUMAN_SOURCES,
        input_sha256=human_input_sha256,
        method_prefix="Human candidate score is the mean over available annotators and dimensions",
    )
    rows.extend(
        _exclusion_rows(
            judge_candidates,
            layer="judge",
            source_column="source_id",
            sources=LARGE_SOURCE_IDS,
            input_sha256=judge_input_sha256,
            method_prefix="Judge candidate score is average_score",
        )
    )
    return pd.DataFrame(rows)


def _regression_values(model, term: str) -> dict[str, float]:
    interval = model.conf_int().loc[term]
    return {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "statistic": float(model.tvalues[term]),
        "p_value": float(model.pvalues[term]),
        "ci_low": float(interval.iloc[0]),
        "ci_high": float(interval.iloc[1]),
        "r_squared": float(model.rsquared),
    }


def _rank_tertiles(pair_mean_words: pd.Series) -> pd.Series:
    ordered = pd.DataFrame(
        {"pair_mean_words": pair_mean_words, "key": pair_mean_words.index.astype(str)}
    ).sort_values(["pair_mean_words", "key"], kind="stable")
    labels = np.asarray(["short", "middle", "long"])
    ordered["bucket"] = labels[
        np.minimum(2, np.floor(3 * np.arange(len(ordered)) / len(ordered)).astype(int))
    ]
    return ordered["bucket"].reindex(pair_mean_words.index)


def _length_layer_rows(
    candidate: pd.DataFrame,
    *,
    layer: str,
    input_sha256: str,
) -> list[dict[str, object]]:
    if candidate["record_id"].nunique() < 2:
        raise ValueError("length regression requires at least two poem clusters")
    fixed = smf.ols(
        "score ~ word_count + C(record_id) + C(system_id)", data=candidate
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": candidate["record_id"], "use_correction": True},
        use_t=True,
    )
    rows: list[dict[str, object]] = [
        {
            "analysis": "candidate_fixed_effect",
            "layer": layer,
            "family": "all_shared_systems",
            "bucket": "not_applicable",
            "estimate_name": "word_count_coefficient",
            "n_clusters": int(candidate["record_id"].nunique()),
            **_regression_values(fixed, "word_count"),
            **_meta(
                n=len(candidate),
                unit="record_id-by-system_id candidate",
                seed="not_applicable",
                input_sha256=input_sha256,
                method=(
                    "OLS score ~ candidate English word count + poem fixed effects + "
                    "system fixed effects; covariance clustered by record_id with "
                    "small-sample correction and t inference. Human scores are candidate "
                    "means over annotators and dimensions."
                    if layer == "human"
                    else "OLS average_score ~ candidate English word count + poem fixed "
                    "effects + system fixed effects; covariance clustered by record_id "
                    "with small-sample correction and t inference."
                ),
            ),
        }
    ]
    for family, (nonthinking, thinking) in FAMILIES.items():
        word_pairs = _paired_family(
            candidate,
            family,
            nonthinking,
            thinking,
            "word_count",
            ["record_id"],
        )
        score_pairs = _paired_family(
            candidate,
            family,
            nonthinking,
            thinking,
            "score",
            ["record_id"],
        )
        paired = pd.DataFrame(
            {
                "word_delta": word_pairs["nt"] - word_pairs["t"],
                "score_delta": score_pairs["nt"] - score_pairs["t"],
                "pair_mean_words": (word_pairs["nt"] + word_pairs["t"]) / 2,
            }
        )
        if paired["word_delta"].nunique() < 2:
            raise ValueError(f"word-count delta is constant for {layer} {family}")
        delta_model = smf.ols("score_delta ~ word_delta", data=paired).fit(
            cov_type="HC3", use_t=True
        )
        rows.append(
            {
                "analysis": "within_family_delta_regression",
                "layer": layer,
                "family": family,
                "bucket": "not_applicable",
                "estimate_name": "slope_score_delta_per_word_delta",
                "n_clusters": len(paired),
                "mean_score_delta_nt_minus_t": float(paired["score_delta"].mean()),
                "mean_word_delta_nt_minus_t": float(paired["word_delta"].mean()),
                **_regression_values(delta_model, "word_delta"),
                **_meta(
                    n=len(paired),
                    unit="record_id paired NT-minus-T delta",
                    seed="not_applicable",
                    input_sha256=input_sha256,
                    method=(
                        "OLS score_delta_NT-minus-T ~ word_delta_NT-minus-T within "
                        "family, with HC3 covariance; one paired record_id per row."
                    ),
                ),
            }
        )
        paired["bucket"] = _rank_tertiles(paired["pair_mean_words"])
        for bucket in ("short", "middle", "long"):
            subset = paired.loc[paired["bucket"] == bucket]
            score_delta = subset["score_delta"]
            rows.append(
                {
                    "analysis": "length_bucket_sensitivity",
                    "layer": layer,
                    "family": family,
                    "bucket": bucket,
                    "estimate_name": "mean_score_delta_nt_minus_t",
                    "n_clusters": len(subset),
                    "estimate": float(score_delta.mean()),
                    "mean_score_delta_nt_minus_t": float(score_delta.mean()),
                    "mean_word_delta_nt_minus_t": float(
                        subset["word_delta"].mean()
                    ),
                    "nt_gt_t_count": int((score_delta > 0).sum()),
                    "nt_gt_t_fraction": float((score_delta > 0).mean()),
                    "bucket_pair_mean_words_min": float(
                        subset["pair_mean_words"].min()
                    ),
                    "bucket_pair_mean_words_max": float(
                        subset["pair_mean_words"].max()
                    ),
                    **_meta(
                        n=len(subset),
                        unit="record_id paired NT-minus-T delta",
                        seed="not_applicable",
                        input_sha256=input_sha256,
                        method=(
                            "Descriptive sensitivity only: deterministic equal-count "
                            "tertiles of each family's NT/T pair-mean English word count; "
                            "report mean score delta and directional count. Small bucket "
                            "sizes are not treated as independent inferential evidence."
                        ),
                    ),
                }
            )
    return rows


def build_length_confound(
    auto: pd.DataFrame,
    human: pd.DataFrame,
    judge: pd.DataFrame,
    *,
    auto_input_sha256: str,
    human_input_sha256: str,
    judge_input_sha256: str,
) -> pd.DataFrame:
    lengths = _auto_candidates(auto).loc[
        :, ["record_id", "system_id", "word_count"]
    ]
    human_candidates = _human_candidates(human)
    judge_candidates = _judge_candidates(judge)
    human_merged = human_candidates.merge(
        lengths,
        on=["record_id", "system_id"],
        how="inner",
        validate="one_to_one",
    )
    judge_merged = judge_candidates.merge(
        lengths,
        on=["record_id", "system_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(human_merged) != len(human_candidates):
        raise ValueError("automatic scores missing human-evaluation candidates")
    if len(judge_merged) != len(judge_candidates):
        raise ValueError("automatic scores missing judge candidates")
    rows = _length_layer_rows(
        human_merged,
        layer="human",
        input_sha256=f"{auto_input_sha256};{human_input_sha256}",
    )
    rows.extend(
        _length_layer_rows(
            judge_merged,
            layer="judge",
            input_sha256=f"{auto_input_sha256};{judge_input_sha256}",
        )
    )
    return pd.DataFrame(rows)


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return pd.DataFrame(rows)


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


def run_analysis(
    human_path: str | Path,
    auto_path: str | Path,
    judge_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = SEED,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, Path]:
    human = pd.read_csv(human_path)
    auto = _read_jsonl(auto_path)
    judge = _read_jsonl(judge_path)
    panel_ids = set(human["record_id"].astype(str))
    judge_ids = set(judge["record_id"].astype(str))
    if panel_ids != judge_ids:
        raise ValueError("human and original-judge artifacts must contain the same panel")

    frames = {
        "annotator": build_annotator_direction(human, input_sha256=""),
        "panel": build_panel_representativeness(
            auto,
            panel_ids,
            seed=seed,
            n_resamples=n_resamples,
            input_sha256="",
        ),
        "exclusion": build_small_source_exclusion(
            human,
            judge,
            human_input_sha256="",
            judge_input_sha256="",
        ),
        "length": build_length_confound(
            auto,
            human,
            judge,
            auto_input_sha256="",
            human_input_sha256="",
            judge_input_sha256="",
        ),
    }
    output_dir = Path(output_dir)
    outputs = {
        "annotator": output_dir / "rebuttal_aux_annotator_direction.csv",
        "panel": output_dir / "rebuttal_aux_panel_comet.csv",
        "exclusion": output_dir / "rebuttal_aux_exclude_small_sources.csv",
        "length": output_dir / "rebuttal_aux_length_confound.csv",
    }
    for name, frame in frames.items():
        _atomic_write_csv(frame, outputs[name])
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run_analysis(
        args.human,
        args.auto,
        args.judge,
        args.output_dir,
        seed=args.seed,
        n_resamples=args.n_resamples,
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
