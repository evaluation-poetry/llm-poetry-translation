# -*- coding: utf-8 -*-
"""Human-evaluation paired tests, agreement, and design-sensitivity MDEs.

This v2 script preserves the original mixed-model, Wilcoxon, bootstrap, Cohen
``d_z``, ordinal Krippendorff alpha, and Kendall W analyses.  It adds separate
Holm corrections, nominal Fleiss kappa for the six integer dimensions, and
paired-design MDE rows.  All defaults write below ``results/stats/rebuttal``.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stats" / "rebuttal"
SEED = 20260706
DIMS = ["MF", "IR", "EV", "LR", "MD", "PT"]
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
N_DECLARED_HUMAN_TESTS = len(FAMILIES) * (len(DIMS) + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(ROOT / "results" / "stats" / "human_eval_long.csv")
    )
    parser.add_argument(
        "--paired-output", default=str(DEFAULT_OUT_DIR / "human_paired_v2.csv")
    )
    parser.add_argument(
        "--iaa-output", default=str(DEFAULT_OUT_DIR / "human_iaa_v2.csv")
    )
    parser.add_argument(
        "--mde-layer-output", default=str(DEFAULT_OUT_DIR / "mde_human.csv")
    )
    return parser


def reject_duplicate_destinations(*destinations: str | Path) -> None:
    resolved = [str(Path(path).resolve()) for path in destinations]
    if len(resolved) != len(set(resolved)):
        raise ValueError("output destination paths must be distinct")


def atomic_write_csv(frame: pd.DataFrame, destination: str | Path) -> None:
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


def holm_adjust(
    pvalues: Iterable[float], family_size: int | None = None
) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if family_size is None:
        family_size = len(finite_values)
    if family_size < len(finite_values):
        raise ValueError("Holm family size cannot be smaller than finite p-value count")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if not len(finite_values):
        return adjusted
    padded = np.concatenate(
        [finite_values, np.ones(family_size - len(finite_values), dtype=float)]
    )
    padded_adjusted = multipletests(padded, alpha=0.05, method="holm")[1]
    adjusted[finite_mask] = padded_adjusted[: len(finite_values)]
    return adjusted


def validate_human_scores(frame: pd.DataFrame) -> None:
    required = [
        "annotator",
        "poem_id",
        "record_id",
        "system_id",
        "eval_id",
        "dim",
        "score",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"human data missing required columns: {missing}")
    keys = ["annotator", "poem_id", "system_id", "eval_id", "dim"]
    if frame.duplicated(keys, keep=False).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head(3)
        raise ValueError(f"human data contains duplicate ratings: {examples.to_dict('records')}")
    scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("human scores must be finite")
    if not np.equal(scores, np.floor(scores)).all() or not (
        (scores >= 1) & (scores <= 6)
    ).all():
        raise ValueError("human scores must be integer values in 1..6")
    unknown_dims = set(frame["dim"].unique()) - set(DIMS)
    if unknown_dims:
        raise ValueError(f"human data contains unknown dimensions: {sorted(unknown_dims)}")


def mixed_model(sub: pd.DataFrame) -> dict[str, float | str]:
    """Fit score ~ mode with crossed annotator and poem random intercepts."""
    model_data = sub.copy()
    model_data["mode_t"] = (model_data["variant"] == "T").astype(float)
    model_data["const_group"] = 1
    model = smf.mixedlm(
        "score ~ mode_t",
        data=model_data,
        groups="const_group",
        vc_formula={"annotator": "0 + C(annotator)", "poem": "0 + C(poem_id)"},
        re_formula="0",
    )
    attempt_messages: list[str] = []
    for optimizer in ["lbfgs", "powell", "cg"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = model.fit(
                    reml=True, method=optimizer, maxiter=500, disp=False
                )
            except Exception as exc:
                attempt_messages.append(
                    f"{optimizer}: {type(exc).__name__}: {exc}"
                )
                continue
        warning_messages = [str(item.message) for item in caught]
        attempt_messages.extend(
            f"{optimizer}: {message}" for message in warning_messages
        )
        try:
            coefficient = float(fit.params["mode_t"])
            standard_error = float(fit.bse["mode_t"])
            p_value = float(fit.pvalues["mode_t"])
        except (KeyError, TypeError, ValueError) as exc:
            attempt_messages.append(f"{optimizer}: invalid fit values: {exc}")
            continue
        finite_valid = np.isfinite(
            [coefficient, standard_error, p_value]
        ).all() and standard_error > 0
        hessian_valid = not any(
            "not positive definite" in message.lower()
            and "hessian" in message.lower()
            for message in warning_messages
        )
        if bool(getattr(fit, "converged", False)) and finite_valid and hessian_valid:
            return {
                "coef": coefficient,
                "se": standard_error,
                "p": p_value,
                "optimizer": optimizer,
                "status": "ok",
                "warnings": " | ".join(attempt_messages),
            }
        reasons = []
        if not bool(getattr(fit, "converged", False)):
            reasons.append("not converged")
        if not finite_valid:
            reasons.append("non-finite coefficient/SE/p or SE<=0")
        if not hessian_valid:
            reasons.append("non-positive-definite Hessian")
        attempt_messages.append(f"{optimizer}: rejected ({'; '.join(reasons)})")
    return {
        "coef": np.nan,
        "se": np.nan,
        "p": np.nan,
        "optimizer": "",
        "status": "failed_all_optimizers",
        "warnings": " | ".join(attempt_messages),
    }


def build_complete_composites(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["annotator", "poem_id", "record_id", "system_id", "eval_id"]
    required = [*keys, "dim", "score"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"human data missing composite columns: {missing}")
    duplicate_keys = [*keys, "dim"]
    if df.duplicated(duplicate_keys, keep=False).any():
        raise ValueError("duplicate dimensions prevent a valid six-dimension composite")
    grouped = (
        df.groupby(keys, as_index=False)
        .agg(
            score=("score", "mean"),
            n_dims=("dim", "size"),
            n_unique_dims=("dim", "nunique"),
        )
    )
    complete = grouped.loc[
        (grouped["n_dims"] == len(DIMS))
        & (grouped["n_unique_dims"] == len(DIMS))
    ].copy()
    complete["dim"] = "COMPOSITE"
    return complete.drop(columns="n_unique_dims")


def kendall_w(matrix: np.ndarray) -> float:
    """Tie-corrected Kendall's W for an items-by-raters complete matrix."""
    n_items, n_raters = matrix.shape
    ranks = np.apply_along_axis(st.rankdata, 0, matrix)
    rank_sums = ranks.sum(axis=1)
    squared_deviation = ((rank_sums - rank_sums.mean()) ** 2).sum()
    tie_term = 0.0
    for rater in range(n_raters):
        _, counts = np.unique(matrix[:, rater], return_counts=True)
        tie_term += (counts**3 - counts).sum()
    denominator = (
        n_raters**2 * (n_items**3 - n_items) - n_raters * tie_term
    )
    return float(12 * squared_deviation / denominator) if denominator > 0 else np.nan


def _fleiss_counts(matrix: np.ndarray) -> np.ndarray:
    counts = np.zeros((matrix.shape[0], 6), dtype=int)
    for row_index, row in enumerate(matrix.astype(int)):
        for category in range(1, 7):
            counts[row_index, category - 1] = int(np.sum(row == category))
    return counts


def compute_iaa(df: pd.DataFrame, comp: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    item_index = sorted(df["eval_id"].unique())
    rater_index = sorted(df["annotator"].unique())
    for dim in DIMS + ["COMPOSITE"]:
        base = comp if dim == "COMPOSITE" else df.loc[df["dim"] == dim]
        pivot = base.pivot(
            index="eval_id", columns="annotator", values="score"
        ).reindex(index=item_index, columns=rater_index)
        alpha = krippendorff.alpha(
            reliability_data=pivot.T.to_numpy(dtype=float),
            level_of_measurement="ordinal",
        )
        complete = pivot.dropna(axis=0, how="any")
        matrix = complete.to_numpy(dtype=float)
        agreement_w = kendall_w(matrix) if len(complete) else np.nan
        if dim == "COMPOSITE":
            kappa = np.nan
            n_fleiss = 0
            n_raters_fleiss = 0
            assumption = (
                "Fleiss kappa not computed for COMPOSITE because averaged scores are "
                "not nominal integer categories."
            )
        else:
            if len(complete) == 0:
                kappa = np.nan
                n_fleiss = 0
                n_raters_fleiss = 0
            else:
                kappa = float(fleiss_kappa(_fleiss_counts(matrix), method="fleiss"))
                n_fleiss = len(complete)
                n_raters_fleiss = matrix.shape[1]
            assumption = (
                "Nominal Fleiss kappa on complete items with the same rater set; "
                "integer categories 1..6."
            )
        rows.append(
            {
                "dim": dim,
                "n_items": len(pivot),
                "n_complete": len(complete),
                "krippendorff_alpha_ordinal": float(alpha),
                "kendall_w": agreement_w,
                "fleiss_kappa_nominal": kappa,
                "n_fleiss_complete": n_fleiss,
                "n_raters_fleiss": n_raters_fleiss,
                "fleiss_assumption": assumption,
            }
        )
    return rows


def mde_row(
    *,
    layer: str,
    family: str,
    measure: str,
    differences: Iterable[float],
    n_tests: int,
    alpha: float,
    power: float = 0.80,
) -> dict:
    values = np.asarray(list(differences), dtype=float)
    if len(values) != 20:
        raise ValueError(f"paired MDE requires n=20; got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("paired differences for MDE must be finite")
    standardized = float(
        TTestPower().solve_power(
            effect_size=None,
            nobs=len(values),
            alpha=alpha,
            power=power,
            alternative="two-sided",
        )
    )
    paired_sd = float(values.std(ddof=1))
    raw_mde = standardized * paired_sd if paired_sd > 0 else np.nan
    alpha_scope = "nominal" if np.isclose(alpha, 0.05) else "holm_conservative"
    return {
        "layer": layer,
        "family": family,
        "measure": measure,
        "n": len(values),
        "power": power,
        "alpha": alpha,
        "alpha_scope": alpha_scope,
        "n_declared_tests": n_tests,
        "standardized_dz_mde": standardized,
        "observed_paired_sd": paired_sd,
        "raw_scale_mde": raw_mde,
        "assumptions": (
            "Design sensitivity, not post-hoc proof: two-sided paired-t approximation "
            "with n=20 and power=0.80; raw MDE equals dz MDE times observed paired SD."
        ),
    }


def write_mde_layer(destination: str | Path, rows: Sequence[dict]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("MDE layer cannot be empty")
    if "layer" not in frame or frame["layer"].nunique() != 1:
        raise ValueError("MDE layer file must contain exactly one declared layer")
    sort_columns = [
        column
        for column in ["layer", "family", "measure", "alpha"]
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    atomic_write_csv(frame, destination)


def merge_mde_layers(
    human_path: str | Path, judge_path: str | Path, destination: str | Path
) -> None:
    reject_duplicate_destinations(human_path, judge_path, destination)
    if not Path(human_path).exists() or not Path(judge_path).exists():
        raise ValueError("both human and judge MDE layer files are required")
    human_frame = pd.read_csv(human_path)
    judge_frame = pd.read_csv(judge_path)
    identity = ["layer", "family", "measure", "alpha"]
    for expected_layer, frame in [
        ("human", human_frame),
        ("judge", judge_frame),
    ]:
        missing = [column for column in identity if column not in frame]
        if missing:
            raise ValueError(f"{expected_layer} MDE file missing columns: {missing}")
        if set(frame["layer"]) != {expected_layer}:
            raise ValueError(f"{expected_layer} MDE file contains wrong layer values")
        if frame.duplicated(identity, keep=False).any():
            raise ValueError(f"duplicate MDE rows in {expected_layer} layer file")
    combined = pd.concat([human_frame, judge_frame], ignore_index=True, sort=False)
    if combined.duplicated(identity, keep=False).any():
        raise ValueError("duplicate MDE rows across layer files")
    combined = combined.sort_values(identity, kind="stable").reset_index(drop=True)
    atomic_write_csv(combined, destination)


def _paired_differences(sub: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    poem_means = (
        sub.groupby(["poem_id", "variant"], as_index=False)["score"]
        .mean()
        .pivot(index="poem_id", columns="variant", values="score")
    )
    if not {"NT", "T"}.issubset(poem_means.columns):
        raise ValueError("paired human test is missing an NT or T column")
    paired = poem_means[["NT", "T"]].dropna()
    differences = (paired["T"] - paired["NT"]).to_numpy(dtype=float)
    if len(differences) != 20:
        raise ValueError(f"paired human test requires 20 poems; got {len(differences)}")
    if not np.isfinite(differences).all():
        raise ValueError("paired human differences must be finite")
    return paired, differences


def paired_wilcoxon(differences: np.ndarray) -> tuple[float, str]:
    if np.allclose(differences, 0):
        return 1.0, "all_zero"
    has_zeros = bool(np.any(np.isclose(differences, 0)))
    with warnings.catch_warnings():
        if has_zeros:
            warnings.filterwarnings(
                "ignore",
                message="Exact p-value calculation does not work if there are zeros.*",
                category=UserWarning,
            )
        _, p_value = st.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
    method = "exact_requested_scipy_fallback" if has_zeros else "exact"
    return float(p_value), method


def compute_paired(
    df: pd.DataFrame, comp: pd.DataFrame
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    mde_rows: list[dict] = []
    for family, (nonthinking, thinking) in FAMILIES.items():
        for dim in DIMS + ["COMPOSITE"]:
            source = comp if dim == "COMPOSITE" else df.loc[df["dim"] == dim]
            sub = source.loc[source["system_id"].isin([nonthinking, thinking])].copy()
            if sub.empty:
                raise ValueError(f"no human rows for {family} {dim}")
            sub["variant"] = np.where(sub["system_id"] == thinking, "T", "NT")
            mixed = mixed_model(sub)
            paired, differences = _paired_differences(sub)
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
            rows.append(
                {
                    "family": family,
                    "dim": dim,
                    "analysis_unit": (
                        "mixed model: rating row; descriptives/Wilcoxon/bootstrap/dz: "
                        "equal-weight poem_id after annotator mean"
                    ),
                    "n_rating_rows_mixedlm": len(sub),
                    "n_poems_paired": len(paired),
                    "mean_nt": paired["NT"].mean(),
                    "mean_t": paired["T"].mean(),
                    "delta_t_minus_nt": differences.mean(),
                    "mixedlm_coef": mixed["coef"],
                    "mixedlm_se": mixed["se"],
                    "mixedlm_p": mixed["p"],
                    "mixedlm_optimizer": mixed["optimizer"],
                    "mixedlm_status": mixed["status"],
                    "mixedlm_warnings": mixed["warnings"],
                    "wilcoxon_exact_p_poemmeans": float(wilcoxon_p),
                    "wilcoxon_method": wilcoxon_method,
                    "boot_ci_lo": float(ci_low),
                    "boot_ci_hi": float(ci_high),
                    "cohen_dz_poemmeans": float(dz),
                }
            )
            for alpha in [0.05, 0.05 / N_DECLARED_HUMAN_TESTS]:
                mde_rows.append(
                    mde_row(
                        layer="human",
                        family=family,
                        measure=dim,
                        differences=differences,
                        n_tests=N_DECLARED_HUMAN_TESTS,
                        alpha=alpha,
                    )
                )
    mixed_adjusted = holm_adjust(
        [row["mixedlm_p"] for row in rows], family_size=N_DECLARED_HUMAN_TESTS
    )
    wilcoxon_adjusted = holm_adjust(
        [row["wilcoxon_exact_p_poemmeans"] for row in rows],
        family_size=N_DECLARED_HUMAN_TESTS,
    )
    for row, mixed_holm, wilcoxon_holm in zip(
        rows, mixed_adjusted, wilcoxon_adjusted, strict=True
    ):
        row["mixedlm_p_holm"] = mixed_holm
        row["wilcoxon_p_holm"] = wilcoxon_holm
        row["holm_family_n"] = N_DECLARED_HUMAN_TESTS
    return rows, mde_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reject_duplicate_destinations(
        args.paired_output, args.iaa_output, args.mde_layer_output
    )
    frame = pd.read_csv(args.input)
    frame = frame.loc[frame["score"].notna() & (frame["score"] != "")].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    validate_human_scores(frame)
    comp = build_complete_composites(frame)
    paired_rows, mde_rows = compute_paired(frame, comp)
    iaa_rows = compute_iaa(frame, comp)
    atomic_write_csv(pd.DataFrame(paired_rows), args.paired_output)
    atomic_write_csv(pd.DataFrame(iaa_rows), args.iaa_output)
    write_mde_layer(args.mde_layer_output, mde_rows)
    print(f"wrote {len(paired_rows)} human paired rows to {args.paired_output}")
    print(f"wrote {len(iaa_rows)} human IAA rows to {args.iaa_output}")
    print(f"wrote {len(mde_rows)} human MDE rows to {args.mde_layer_output}")


if __name__ == "__main__":
    main()
