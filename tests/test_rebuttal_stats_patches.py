from __future__ import annotations

import math
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import stats_alignment as alignment
from scripts import stats_human_paired as human
from scripts import stats_judge_paired as judge


def test_correlation_stats_reports_pearson_separately_from_rank_correlations():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 2, 3, 5, 100], dtype=float)

    result = alignment.correlation_stats(x, y)

    assert result["n"] == 5
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["pearson_r"] < 0.8
    assert 0 <= result["pearson_p"] <= 1


def test_alignment_rejects_duplicate_or_nonfinite_pairs():
    duplicate = pd.DataFrame(
        {"record_id": ["p1", "p1"], "system_id": ["s1", "s1"], "x": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        alignment.validate_unique_finite(
            duplicate, ["record_id", "system_id"], ["x"], "synthetic"
        )

    nonfinite = pd.DataFrame(
        {"record_id": ["p1"], "system_id": ["s1"], "x": [np.inf]}
    )
    with pytest.raises(ValueError, match="finite"):
        alignment.validate_unique_finite(
            nonfinite, ["record_id", "system_id"], ["x"], "synthetic"
        )


def test_alignment_ter_delta_is_oriented_as_improvement():
    oriented, orientation, raw_definition = alignment.orient_auto_delta(
        "ter_sentence", np.array([2.0, -3.0])
    )
    assert oriented.tolist() == [-2.0, 3.0]
    assert orientation == "lower_is_better; negated raw T-minus-NT"
    assert raw_definition == "ter_sentence raw T-minus-NT"


def test_holm_adjustment_matches_known_values_and_is_monotone_by_raw_p():
    raw = np.array([0.01, 0.02, 0.04, np.nan])
    adjusted = human.holm_adjust(raw)

    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert math.isnan(adjusted[3])
    order = np.argsort(raw[:3])
    assert np.all(np.diff(adjusted[:3][order]) >= 0)


def test_judge_holm_uses_same_declared_family_of_tests():
    assert judge.holm_adjust([0.01, 0.02, 0.04]) == pytest.approx([0.03, 0.04, 0.04])


def test_wilcoxon_preserves_exact_request_and_labels_scipy_zero_fallback():
    p_value, method = human.paired_wilcoxon(
        np.array([0.0, -1.0, -0.5, 0.25, -0.75] * 4)
    )
    assert 0 <= p_value <= 1
    assert method == "exact_requested_scipy_fallback"


def _iaa_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scores = {
        "e1": [1, 1, 2],
        "e2": [3, 3, 3],
        "e3": [5, 6, 5],
        "e4": [2, 2, 2],
    }
    for dim in human.DIMS:
        for eval_id, values in scores.items():
            for idx, score in enumerate(values, start=1):
                rows.append(
                    {
                        "annotator": f"A{idx}",
                        "poem_id": eval_id,
                        "record_id": eval_id,
                        "system_id": "s1",
                        "eval_id": eval_id,
                        "dim": dim,
                        "score": score,
                    }
                )
    df = pd.DataFrame(rows)
    comp = (
        df.groupby(["annotator", "poem_id", "system_id", "eval_id"], as_index=False)
        .agg(score=("score", "mean"), n_dims=("score", "size"))
        .assign(dim="COMPOSITE")
    )
    return df, comp


def test_iaa_adds_fleiss_for_integer_dimensions_but_omits_composite():
    df, comp = _iaa_fixture()

    rows = human.compute_iaa(df, comp)
    by_dim = {row["dim"]: row for row in rows}

    assert 0 <= by_dim["MF"]["fleiss_kappa_nominal"] <= 1
    assert by_dim["MF"]["n_fleiss_complete"] == 4
    assert math.isnan(by_dim["COMPOSITE"]["fleiss_kappa_nominal"])
    assert "not computed" in by_dim["COMPOSITE"]["fleiss_assumption"].lower()


def test_human_validation_rejects_out_of_range_and_duplicate_scores():
    valid = pd.DataFrame(
        [
            {
                "annotator": "A1",
                "poem_id": "P1",
                "record_id": "r1",
                "system_id": "s1",
                "eval_id": "e1",
                "dim": "MF",
                "score": 1,
            },
            {
                "annotator": "A2",
                "poem_id": "P1",
                "record_id": "r1",
                "system_id": "s1",
                "eval_id": "e1",
                "dim": "MF",
                "score": 6,
            },
        ]
    )
    human.validate_human_scores(valid)

    out_of_range = valid.copy()
    out_of_range.loc[0, "score"] = 7
    with pytest.raises(ValueError, match="1..6"):
        human.validate_human_scores(out_of_range)

    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        human.validate_human_scores(duplicate)


class _FakeFit:
    def __init__(self, *, converged=True, coef=0.2, se=0.1, p=0.04):
        self.converged = converged
        self.params = {"mode_t": coef}
        self.bse = {"mode_t": se}
        self.pvalues = {"mode_t": p}


class _FakeMixedModel:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def fit(self, *, method, **kwargs):
        self.calls.append(method)
        outcome = self.outcomes[method]
        if isinstance(outcome, BaseException):
            raise outcome
        fit, warning_messages = outcome
        for message in warning_messages:
            warnings.warn(message, UserWarning)
        return fit


def _minimal_mixed_data():
    return pd.DataFrame(
        {
            "score": [1.0, 2.0, 2.0, 3.0],
            "variant": ["NT", "T", "NT", "T"],
            "annotator": ["A1", "A1", "A2", "A2"],
            "poem_id": ["P1", "P1", "P2", "P2"],
        }
    )


def test_mixed_model_retries_nonconverged_fit_with_deterministic_optimizer(monkeypatch):
    model = _FakeMixedModel(
        {
            "lbfgs": (_FakeFit(converged=False), ["failed to converge"]),
            "powell": (_FakeFit(converged=True, coef=0.3, se=0.2, p=0.02), []),
            "cg": AssertionError("cg should not run after valid powell fit"),
        }
    )
    monkeypatch.setattr(human.smf, "mixedlm", lambda *args, **kwargs: model)

    result = human.mixed_model(_minimal_mixed_data())

    assert model.calls == ["lbfgs", "powell"]
    assert result["status"] == "ok"
    assert result["optimizer"] == "powell"
    assert result["p"] == pytest.approx(0.02)
    assert "failed to converge" in result["warnings"]


def test_mixed_model_rejects_nonpositive_definite_hessian_and_uses_next(monkeypatch):
    model = _FakeMixedModel(
        {
            "lbfgs": (
                _FakeFit(converged=True, p=0.001),
                ["The Hessian matrix at the estimated parameter values is not positive definite."],
            ),
            "powell": (_FakeFit(converged=True, coef=0.4, se=0.2, p=0.03), []),
            "cg": AssertionError("cg should not run"),
        }
    )
    monkeypatch.setattr(human.smf, "mixedlm", lambda *args, **kwargs: model)

    result = human.mixed_model(_minimal_mixed_data())

    assert result["optimizer"] == "powell"
    assert result["p"] == pytest.approx(0.03)
    assert "not positive definite" in result["warnings"]


def test_mixed_model_all_invalid_fits_return_na_with_explicit_status(monkeypatch):
    model = _FakeMixedModel(
        {
            "lbfgs": (_FakeFit(converged=False), []),
            "powell": RuntimeError("powell failed"),
            "cg": (_FakeFit(converged=True, se=0.0), []),
        }
    )
    monkeypatch.setattr(human.smf, "mixedlm", lambda *args, **kwargs: model)

    result = human.mixed_model(_minimal_mixed_data())

    assert model.calls == ["lbfgs", "powell", "cg"]
    assert result["status"] == "failed_all_optimizers"
    assert result["optimizer"] == ""
    assert math.isnan(result["coef"])
    assert math.isnan(result["se"])
    assert math.isnan(result["p"])
    assert "powell failed" in result["warnings"]


def test_complete_composite_omits_five_dimension_ratings_and_iaa_counts_missing():
    df, _ = _iaa_fixture()
    sparse = df.drop(
        df[(df["eval_id"] == "e1") & (df["annotator"] == "A1") & (df["dim"] == "PT")].index
    ).drop(
        df[(df["eval_id"] == "e2") & (df["annotator"] == "A2") & (df["dim"] == "MD")].index
    )

    comp = human.build_complete_composites(sparse)
    rows = human.compute_iaa(sparse, comp)
    composite = next(row for row in rows if row["dim"] == "COMPOSITE")

    assert not ((comp["eval_id"] == "e1") & (comp["annotator"] == "A1")).any()
    assert not ((comp["eval_id"] == "e2") & (comp["annotator"] == "A2")).any()
    assert len(comp) == 10
    assert composite["n_items"] == 4
    assert composite["n_complete"] == 2


def test_alignment_human_composite_requires_six_complete_annotators():
    rows = []
    for annotator_index in range(7):
        for dim in human.DIMS:
            if annotator_index == 6 and dim == "PT":
                continue
            rows.append(
                {
                    "annotator": f"A{annotator_index + 1}",
                    "record_id": "r1",
                    "system_id": "s1",
                    "dim": dim,
                    "score": 1 + annotator_index % 6,
                }
            )
    frame = pd.DataFrame(rows)
    composite = alignment.build_human_composites(frame)
    assert composite.loc[0, "n_complete_annotators"] == 6
    assert composite.loc[0, "human_composite"] == pytest.approx(3.5)

    too_sparse = frame.loc[~frame["annotator"].isin(["A5", "A6"])].copy()
    with pytest.raises(ValueError, match="at least 6 complete annotators"):
        alignment.build_human_composites(too_sparse)


def test_standardized_mde_is_positive_and_holm_conservative_alpha_raises_it():
    differences = np.linspace(-1.0, 1.0, 20)

    nominal = human.mde_row(
        layer="human",
        family="family",
        measure="MF",
        differences=differences,
        n_tests=21,
        alpha=0.05,
    )
    conservative = human.mde_row(
        layer="human",
        family="family",
        measure="MF",
        differences=differences,
        n_tests=21,
        alpha=0.05 / 21,
    )

    assert nominal["n"] == 20
    assert nominal["standardized_dz_mde"] > 0
    assert nominal["raw_scale_mde"] > 0
    assert conservative["standardized_dz_mde"] > nominal["standardized_dz_mde"]
    assert conservative["raw_scale_mde"] > nominal["raw_scale_mde"]
    assert "design sensitivity" in nominal["assumptions"].lower()


def test_default_outputs_are_v2_and_do_not_point_at_legacy_files():
    alignment_args = alignment.build_parser().parse_args([])
    human_args = human.build_parser().parse_args([])
    judge_args = judge.build_parser().parse_args([])

    assert Path(alignment_args.output).as_posix().endswith(
        "results/stats/rebuttal/stats_alignment_v2.csv"
    )
    assert Path(human_args.paired_output).as_posix().endswith(
        "results/stats/rebuttal/human_paired_v2.csv"
    )
    assert Path(human_args.iaa_output).as_posix().endswith(
        "results/stats/rebuttal/human_iaa_v2.csv"
    )
    assert Path(judge_args.output).as_posix().endswith(
        "results/stats/rebuttal/judge_paired_v2.csv"
    )
    for path in [
        alignment_args.output,
        human_args.paired_output,
        human_args.iaa_output,
        judge_args.output,
    ]:
        assert "_v2.csv" in Path(path).name


def test_layer_mde_writes_merge_deterministically_and_survive_repeated_runs(tmp_path):
    human_path = tmp_path / "mde_human.csv"
    judge_path = tmp_path / "mde_judge.csv"
    combined_path = tmp_path / "mde.csv"
    human_rows = [
        {
            "layer": "human",
            "family": "f1",
            "measure": "MF",
            "alpha": 0.05,
            "n": 20,
        }
    ]
    judge_rows = [
        {
            "layer": "judge",
            "family": "f1",
            "measure": "average_score",
            "alpha": 0.05,
            "n": 20,
        }
    ]

    human.write_mde_layer(human_path, human_rows)
    with pytest.raises(ValueError, match="both human and judge"):
        human.merge_mde_layers(human_path, judge_path, combined_path)
    human.write_mde_layer(judge_path, judge_rows)
    human.merge_mde_layers(human_path, judge_path, combined_path)
    # Repeat both layers in the opposite order; no rows may disappear or duplicate.
    human.write_mde_layer(judge_path, judge_rows)
    human.write_mde_layer(human_path, human_rows)
    human.merge_mde_layers(human_path, judge_path, combined_path)

    combined = pd.read_csv(combined_path)
    assert combined["layer"].tolist() == ["human", "judge"]
    assert len(combined) == 2


def test_atomic_write_rolls_back_existing_destination_on_replace_failure(tmp_path, monkeypatch):
    destination = tmp_path / "result.csv"
    destination.write_text("old\n", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(human.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        human.atomic_write_csv(pd.DataFrame({"value": [1]}), destination)

    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".result.csv.*.tmp")) == []


def test_duplicate_destination_paths_are_rejected():
    with pytest.raises(ValueError, match="destination"):
        human.reject_duplicate_destinations("same.csv", Path("same.csv"))


def test_human_descriptives_use_equal_weight_paired_poem_means(monkeypatch):
    monkeypatch.setattr(human, "FAMILIES", {"family": ("nt", "t")})
    monkeypatch.setattr(human, "DIMS", ["MF"])
    monkeypatch.setattr(human, "N_DECLARED_HUMAN_TESTS", 2)
    monkeypatch.setattr(
        human,
        "mixed_model",
        lambda sub: {
            "coef": 0.1,
            "se": 0.1,
            "p": 0.2,
            "optimizer": "lbfgs",
            "status": "ok",
            "warnings": "",
        },
    )
    rows = []
    for poem_index in range(20):
        rows.append(
            {
                "annotator": "A1",
                "poem_id": f"P{poem_index}",
                "system_id": "nt",
                "score": 1,
                "dim": "MF",
            }
        )
        thinking_scores = [6] * 7 if poem_index == 0 else [2]
        for annotator_index, score in enumerate(thinking_scores):
            rows.append(
                {
                    "annotator": f"A{annotator_index + 1}",
                    "poem_id": f"P{poem_index}",
                    "system_id": "t",
                    "score": score,
                    "dim": "MF",
                }
            )
    frame = pd.DataFrame(rows)
    comp = frame.assign(dim="COMPOSITE")

    paired_rows, _ = human.compute_paired(frame, comp)

    assert len(paired_rows) == 2
    for row in paired_rows:
        assert row["mean_nt"] == pytest.approx(1.0)
        assert row["mean_t"] == pytest.approx(2.2)
        assert row["delta_t_minus_nt"] == pytest.approx(1.2)
        assert row["n_poems_paired"] == 20
        assert row["n_rating_rows_mixedlm"] == len(frame)
        assert "rating row" in row["analysis_unit"]
        assert "equal-weight poem" in row["analysis_unit"]


def test_alignment_main_writes_all_v2_analysis_types_from_synthetic_raw_inputs(tmp_path):
    systems = [
        "baidu_translate",
        *[system for pair in alignment.FAMILIES.values() for system in pair],
    ]
    record_ids = [f"r{i}" for i in range(5)]
    human_rows = []
    judge_rows = []
    auto_rows = []
    for system_index, system_id in enumerate(systems):
        is_thinking = system_id.endswith("_thinking") and not system_id.endswith(
            "_non_thinking"
        )
        for record_index, record_id in enumerate(record_ids):
            family_offset = system_index * 0.13
            mode_offset = (record_index - 2) * 0.07 if is_thinking else 0.0
            for annotator_index in range(7):
                for dim_index, dim in enumerate(human.DIMS):
                    human_rows.append(
                        {
                            "annotator": f"A{annotator_index + 1}",
                            "record_id": record_id,
                            "system_id": system_id,
                            "dim": dim,
                            "score": 2.0
                            + record_index * 0.4
                            + family_offset
                            + mode_offset
                            + annotator_index * 0.01
                            + dim_index * 0.005,
                        }
                    )
            judge_rows.append(
                {
                    "record_id": record_id,
                    "system_id": system_id,
                    "average_score": 4.0
                    + record_index**2 * 0.09
                    + family_offset
                    + mode_offset * 0.5,
                }
            )
            auto_rows.append(
                {
                    "record_id": record_id,
                    "system_id": system_id,
                    "comet": 0.5 + system_index * 0.01 + record_index * 0.003 + mode_offset,
                    "bertscore_f1": 0.6
                    + system_index * 0.009
                    + record_index * 0.004
                    + mode_offset * 0.7,
                    "chrfpp_sentence": 30
                    + system_index
                    + record_index * 0.5
                    + mode_offset * 4,
                    "ter_sentence": 100
                    - system_index
                    + record_index * 0.3
                    - mode_offset * 3,
                }
            )
    summary_rows = [
        {
            "system_id": system_id,
            "comet_mean": 0.5 + index * 0.01,
            "bertscore_f1_mean": 0.6 + index * 0.01,
            "sacrebleu_corpus": 10 + index,
            "chrfpp_corpus": 30 + index,
            "ter_corpus": 100 - index,
        }
        for index, system_id in enumerate(systems)
    ]
    human_path = tmp_path / "human.csv"
    judge_path = tmp_path / "judge.jsonl"
    auto_path = tmp_path / "auto.jsonl"
    summary_path = tmp_path / "summary.csv"
    output_path = tmp_path / "stats_alignment_v2.csv"
    pd.DataFrame(human_rows).to_csv(human_path, index=False)
    judge_path.write_text(
        "".join(json.dumps(row) + "\n" for row in judge_rows), encoding="utf-8"
    )
    auto_path.write_text(
        "".join(json.dumps(row) + "\n" for row in auto_rows), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    alignment.main(
        [
            "--human",
            str(human_path),
            "--judge",
            str(judge_path),
            "--auto-scores",
            str(auto_path),
            "--auto-summary",
            str(summary_path),
            "--output",
            str(output_path),
        ]
    )

    output = pd.read_csv(output_path)
    assert set(output["analysis_type"]) == {
        "item_pooled",
        "item_per_system",
        "paired_delta",
        "system_ranking",
    }
    assert len(output) == 34
    assert (output["n"] >= 5).all()
    assert {"pearson_r", "spearman_rho", "kendall_tau"}.issubset(output.columns)
    ter_delta = output.loc[
        (output["analysis_type"] == "paired_delta")
        & (output["scope"] == "ter_sentence_improvement_delta")
    ]
    assert len(ter_delta) == 3
    assert ter_delta["orientation"].eq(
        "lower_is_better; negated raw T-minus-NT"
    ).all()
    assert ter_delta["raw_delta"].eq("ter_sentence raw T-minus-NT").all()
    assert output["human_composite_policy"].str.contains(
        "at least 6 complete annotators"
    ).all()


def _write_synthetic_human(path: Path) -> None:
    rows = []
    for family_index, (nonthinking, thinking) in enumerate(human.FAMILIES.values()):
        for system_id, is_thinking in [(nonthinking, False), (thinking, True)]:
            for poem_index in range(20):
                for annotator_index in range(3):
                    for dim_index, dim in enumerate(human.DIMS):
                        score = 2 + (
                            poem_index + annotator_index + dim_index + family_index
                        ) % 4
                        if is_thinking and (poem_index + dim_index) % 2:
                            score -= 1
                        rows.append(
                            {
                                "annotator": f"A{annotator_index + 1}",
                                "eval_id": f"r{poem_index}_{system_id}",
                                "poem_id": f"P{poem_index:02d}",
                                "record_id": f"r{poem_index}",
                                "source": "synthetic",
                                "system_id": system_id,
                                "dim": dim,
                                "score": score,
                            }
                        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synthetic_judge(path: Path) -> None:
    rows = []
    for family_index, (nonthinking, thinking) in enumerate(judge.FAMILIES.values()):
        for system_id, is_thinking in [(nonthinking, False), (thinking, True)]:
            for poem_index in range(20):
                rows.append(
                    {
                        "record_id": f"r{poem_index}",
                        "system_id": system_id,
                        "average_score": 7
                        + family_index * 0.1
                        + poem_index * 0.02
                        - (0.1 + 0.1 * (poem_index % 3) if is_thinking else 0),
                        "rank_position": 2
                        + family_index
                        + (1 if is_thinking and poem_index % 2 else 0),
                    }
                )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_repeated_human_and_judge_main_execution_keeps_both_mde_layers(
    tmp_path, monkeypatch
):
    human_input = tmp_path / "human.csv"
    judge_input = tmp_path / "judge.jsonl"
    _write_synthetic_human(human_input)
    _write_synthetic_judge(judge_input)
    paired_human = tmp_path / "human_paired_v2.csv"
    iaa_human = tmp_path / "human_iaa_v2.csv"
    paired_judge = tmp_path / "judge_paired_v2.csv"
    mde_human = tmp_path / "mde_human.csv"
    mde_judge = tmp_path / "mde_judge.csv"
    mde_combined = tmp_path / "mde.csv"
    monkeypatch.setattr(
        human,
        "mixed_model",
        lambda sub: {
            "coef": 0.1,
            "se": 0.02,
            "p": 0.01,
            "optimizer": "lbfgs",
            "status": "ok",
            "warnings": "",
        },
    )
    human_args = [
        "--input",
        str(human_input),
        "--paired-output",
        str(paired_human),
        "--iaa-output",
        str(iaa_human),
        "--mde-layer-output",
        str(mde_human),
    ]
    judge_args = [
        "--input",
        str(judge_input),
        "--output",
        str(paired_judge),
        "--mde-layer-output",
        str(mde_judge),
    ]

    mde_combined.write_text("sentinel\n", encoding="utf-8")
    human.main(human_args)
    judge.main(judge_args)
    assert mde_combined.read_text(encoding="utf-8") == "sentinel\n"
    merge_args = [
        "--merge-mde-only",
        "--human-mde-input",
        str(mde_human),
        "--mde-layer-output",
        str(mde_judge),
        "--mde-output",
        str(mde_combined),
    ]
    judge.main(merge_args)
    merged_bytes = mde_combined.read_bytes()
    human.main(human_args)
    judge.main(judge_args)
    assert mde_combined.read_bytes() == merged_bytes
    judge.main(merge_args)

    combined = pd.read_csv(mde_combined)
    assert set(combined["layer"]) == {"human", "judge"}
    assert len(combined.loc[combined["layer"] == "human"]) == 42
    assert len(combined.loc[combined["layer"] == "judge"]) == 12
    assert len(combined) == 54
    human_paired = pd.read_csv(paired_human)
    judge_paired = pd.read_csv(paired_judge)
    assert len(human_paired) == 21
    assert len(judge_paired) == 6
    allowed_methods = {"exact", "exact_requested_scipy_fallback", "all_zero"}
    assert set(human_paired["wilcoxon_method"]) <= allowed_methods
    assert set(judge_paired["wilcoxon_method"]) <= allowed_methods
