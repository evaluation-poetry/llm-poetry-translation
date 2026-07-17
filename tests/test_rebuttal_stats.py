from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stats_crossjudge as cross  # noqa: E402
import stats_intensity as intensity  # noqa: E402


METRICS = [
    "comet",
    "bertscore_f1",
    "sacrebleu_sentence",
    "chrfpp_sentence",
    "ter_sentence",
    "line_count_diff",
    "length_ratio",
]

DS_HIGH = "deepseek_v4_flash_thinking"
DS_MAX = "deepseek_v4_flash_thinking_max"
DS_NT = "deepseek_v4_flash_non_thinking"
Q2048 = "qwen36_plus_thinking_b2048"
Q4096 = "qwen36_plus_thinking_b4096"
QDEFAULT = "qwen36_plus_thinking"
QNT = "qwen36_plus_non_thinking"

ORIGINAL_SYSTEMS = [
    "baidu_translate",
    "deepseek_v4_flash_non_thinking",
    DS_HIGH,
    "qwen36_plus_non_thinking",
    QDEFAULT,
    "claude_sonnet46_non_thinking",
    "claude_sonnet46_thinking",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_row(record_id: str, system_id: str, value: float) -> dict:
    return {
        "record_id": record_id,
        "system_id": system_id,
        "status": "ok",
        **{metric: value for metric in METRICS},
    }


def output_row(record_id: str, system_id: str, scale: float) -> dict:
    reasoning = 10.0 * scale
    completion = 20.0 * scale
    prompt = 5.0 * scale
    return {
        "record_id": record_id,
        "system_id": system_id,
        "status": "ok",
        "latency_seconds": 2.0 * scale,
        "token_usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def make_intensity_data(n_auto: int = 4, n_judge: int = 3) -> tuple[list[dict], ...]:
    record_ids = [f"a{i}" for i in range(n_auto)]
    original_scores: list[dict] = []
    supplemental_scores: list[dict] = []
    original_outputs: list[dict] = []
    supplemental_outputs: list[dict] = []
    values = {
        DS_NT: 0.5,
        DS_HIGH: 1.0,
        DS_MAX: 2.0,
        QNT: 1.5,
        Q2048: 2.0,
        Q4096: 4.0,
        QDEFAULT: 5.0,
    }
    for system_id in [DS_NT, DS_HIGH, QNT, QDEFAULT]:
        original_scores.extend(score_row(rid, system_id, values[system_id]) for rid in record_ids)
        original_outputs.extend(output_row(rid, system_id, values[system_id]) for rid in record_ids)
    for system_id in [DS_MAX, Q2048, Q4096]:
        supplemental_scores.extend(score_row(rid, system_id, values[system_id]) for rid in record_ids)
        supplemental_outputs.extend(output_row(rid, system_id, values[system_id]) for rid in record_ids)

    judge_rows: list[dict] = []
    for rid in record_ids[:n_judge]:
        for system_id, value in values.items():
            judge_rows.append(
                {
                    "record_id": rid,
                    "system_id": system_id,
                    "status": "ok",
                    "average_score": value,
                    **{dimension: value for dimension in cross.JUDGE_DIMENSIONS},
                }
            )
    return original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows


def test_holm_is_monotone_in_sorted_raw_p_values() -> None:
    raw = [0.01, 0.04, 0.03, 0.20]
    adjusted = intensity.holm_adjust(raw)
    order = np.argsort(raw)
    ordered_adjusted = [adjusted[index] for index in order]
    assert all(a <= b for a, b in zip(ordered_adjusted, ordered_adjusted[1:]))
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_cohen_dz_distinguishes_all_zero_from_constant_nonzero() -> None:
    assert intensity._cohen_dz(np.zeros(4)) == (0.0, "all_zero")
    assert intensity._cohen_dz(np.ones(4)) == (None, "zero_variance_nonzero")
    value, status = intensity._cohen_dz(np.asarray([1.0, 2.0, 3.0]))
    assert value == pytest.approx(2.0)
    assert status == "ok"


def test_wilcoxon_propagates_non_all_zero_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise ValueError("synthetic scipy failure")

    monkeypatch.setattr(intensity.stats, "wilcoxon", fail)
    with pytest.raises(ValueError, match="synthetic scipy failure"):
        intensity._wilcoxon_p(np.asarray([0.0, 1.0]))


def test_intensity_analysis_has_exact_directions_reproducible_bootstrap_and_resources() -> None:
    data = make_intensity_data()
    first = intensity.analyze_intensity(
        *data,
        expected_auto_count=4,
        expected_judge_count=3,
        seed=71,
        bootstrap_samples=300,
    )
    second = intensity.analyze_intensity(
        *data,
        expected_auto_count=4,
        expected_judge_count=3,
        seed=71,
        bootstrap_samples=300,
    )
    assert first == second

    auto = {
        (row["comparison"], row["metric"]): row
        for row in first["paired"]
        if row["layer"] == "automatic"
    }
    assert auto[("deepseek_high_to_max", "comet")]["delta_right_minus_left"] == pytest.approx(1.0)
    assert auto[("deepseek_nonthinking_to_high", "comet")]["delta_right_minus_left"] == pytest.approx(0.5)
    assert auto[("deepseek_nonthinking_to_max", "comet")]["delta_right_minus_left"] == pytest.approx(1.5)
    assert auto[("qwen_nonthinking_to_2048", "comet")]["delta_right_minus_left"] == pytest.approx(0.5)
    assert auto[("qwen_nonthinking_to_4096", "comet")]["delta_right_minus_left"] == pytest.approx(2.5)
    assert auto[("qwen_nonthinking_to_default", "comet")]["delta_right_minus_left"] == pytest.approx(3.5)
    assert auto[("qwen_2048_to_4096", "comet")]["delta_right_minus_left"] == pytest.approx(2.0)
    assert auto[("qwen_4096_to_default", "comet")]["delta_right_minus_left"] == pytest.approx(1.0)
    assert auto[("qwen_2048_to_default", "comet")]["delta_right_minus_left"] == pytest.approx(3.0)
    assert auto[("qwen_2048_to_default", "comet")]["direction"] == "right_higher"
    assert auto[("deepseek_high_to_max", "comet")]["n"] == 4
    assert auto[("deepseek_high_to_max", "comet")]["unit"] == "record_id"
    assert "p_raw" in auto[("deepseek_high_to_max", "comet")]
    assert "p_holm" in auto[("deepseek_high_to_max", "comet")]
    assert "cohen_dz" in auto[("deepseek_high_to_max", "comet")]
    assert auto[("deepseek_high_to_max", "comet")]["cohen_dz"] is None
    assert auto[("deepseek_high_to_max", "comet")]["dz_status"] == "zero_variance_nonzero"
    assert auto[("deepseek_high_to_max", "comet")]["effect_name"] == "cohen_dz"
    assert auto[("deepseek_high_to_max", "comet")]["effect"] == auto[("deepseek_high_to_max", "comet")]["cohen_dz"]

    judge = [row for row in first["paired"] if row["layer"] == "judge_b"]
    assert {row["metric"] for row in judge} == {"average_score", *cross.JUDGE_DIMENSIONS}
    assert all(row["n"] == 3 for row in judge)

    resources = {row["system_id"]: row for row in first["resources"]}
    assert resources[DS_MAX]["reasoning_tokens_mean"] == pytest.approx(20.0)
    assert resources[Q4096]["latency_seconds_mean"] == pytest.approx(8.0)
    assert resources[DS_HIGH]["attempt_scope"] == "final_output_files"
    assert resources[DS_HIGH]["resource_stat_scope"] == "successful_physical_attempts"
    assert resources[DS_HIGH]["attempt_count"] == 4
    assert resources[DS_HIGH]["successful_attempt_count"] == 4
    assert resources[DS_HIGH]["final_unique_success_count"] == 4
    assert resources[DS_HIGH]["failure_count"] == 0
    assert resources[DS_HIGH]["failure_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "missing", "status"])
def test_intensity_rejects_invalid_or_unpaired_inputs(bad: str) -> None:
    original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows = make_intensity_data()
    if bad == "duplicate":
        supplemental_scores.append(dict(supplemental_scores[0]))
    elif bad == "nonfinite":
        supplemental_scores[0]["comet"] = math.nan
    elif bad == "missing":
        supplemental_scores.pop(0)
    else:
        supplemental_outputs[0]["status"] = "error"
    with pytest.raises(ValueError):
        intensity.analyze_intensity(
            original_scores,
            supplemental_scores,
            original_outputs,
            supplemental_outputs,
            judge_rows,
            expected_auto_count=4,
            expected_judge_count=3,
            seed=7,
            bootstrap_samples=50,
        )


def test_resource_summary_counts_failed_then_successful_physical_retry() -> None:
    original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows = make_intensity_data()
    failed_retry = {
        "record_id": "a0",
        "system_id": DS_MAX,
        "status": "error",
        "error": "transient",
    }
    attempts = [*original_outputs, failed_retry, *supplemental_outputs]
    result = intensity.analyze_intensity(
        original_scores,
        supplemental_scores,
        original_outputs,
        supplemental_outputs,
        judge_rows,
        attempt_rows=attempts,
        expected_auto_count=4,
        expected_judge_count=3,
        seed=7,
        bootstrap_samples=20,
    )
    resources = {row["system_id"]: row for row in result["resources"]}
    row = resources[DS_MAX]
    assert row["attempt_scope"] == "explicit_attempt_files"
    assert row["attempt_count"] == 5
    assert row["successful_attempt_count"] == 4
    assert row["final_unique_success_count"] == 4
    assert row["failure_count"] == 1
    assert row["failure_rate"] == pytest.approx(0.2)
    assert row["reasoning_tokens_mean"] == pytest.approx(20.0)


@pytest.mark.parametrize("missing_field", ["system_id", "record_id", "status"])
def test_resource_attempts_reject_missing_identity_or_status(missing_field: str) -> None:
    original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows = make_intensity_data()
    attempts = [*original_outputs, *supplemental_outputs]
    attempts[0] = {key: value for key, value in attempts[0].items() if key != missing_field}
    with pytest.raises(ValueError, match=missing_field):
        intensity.analyze_intensity(
            original_scores,
            supplemental_scores,
            original_outputs,
            supplemental_outputs,
            judge_rows,
            attempt_rows=attempts,
            expected_auto_count=4,
            expected_judge_count=3,
            seed=7,
            bootstrap_samples=20,
        )


SYSTEM_OFFSETS = {
    "baidu_translate": 0,
    "deepseek_v4_flash_non_thinking": 1,
    DS_HIGH: 2,
    "qwen36_plus_non_thinking": 1,
    QDEFAULT: 2,
    "claude_sonnet46_non_thinking": 2,
    "claude_sonnet46_thinking": 3,
}
HUMAN_DIMS = ["MF", "IR", "EV", "LR", "MD", "PT"]
ANNOTATORS = [f"A{index:02d}" for index in range(1, 8)]


def make_crossjudge_data() -> tuple[list[dict], list[dict], list[dict]]:
    deepseek_rows: list[dict] = []
    gpt_rows: list[dict] = []
    human_rows: list[dict] = []
    for poem_index in range(20):
        record_id = f"p{poem_index:02d}"
        poem_effect = poem_index % 3 + 1
        for system_id in ORIGINAL_SYSTEMS:
            human = poem_effect + SYSTEM_OFFSETS[system_id]
            is_deepseek = system_id.startswith("deepseek_")
            gpt = human
            deepseek = human + (1.5 if is_deepseek else 0.0)
            base = {"record_id": record_id, "system_id": system_id, "status": "ok"}
            deepseek_rows.append(
                {
                    **base,
                    "average_score": deepseek,
                    **{dimension: deepseek + dim_index * 0.001 for dim_index, dimension in enumerate(cross.JUDGE_DIMENSIONS)},
                }
            )
            gpt_rows.append(
                {
                    **base,
                    "average_score": gpt,
                    **{dimension: gpt + dim_index * 0.001 for dim_index, dimension in enumerate(cross.JUDGE_DIMENSIONS)},
                }
            )
            for annotator in ANNOTATORS:
                for dim in HUMAN_DIMS:
                    human_rows.append(
                        {
                            "record_id": record_id,
                            "system_id": system_id,
                            "annotator": annotator,
                            "dim": dim,
                            "score": human,
                        }
                    )
    return deepseek_rows, gpt_rows, human_rows


def rows_by(output: dict[str, list[dict]], name: str, **keys: str) -> list[dict]:
    return [row for row in output[name] if all(row.get(key) == value for key, value in keys.items())]


def test_crossjudge_correlations_bias_interaction_and_bootstrap_are_reproducible() -> None:
    data = make_crossjudge_data()
    first = cross.analyze_crossjudge(
        *data, seed=113, bootstrap_samples=60, permutation_samples=60
    )
    second = cross.analyze_crossjudge(
        *data, seed=113, bootstrap_samples=60, permutation_samples=60
    )
    assert first == second
    assert set(first) == {
        "agreement",
        "self_preference",
        "judge_human_alignment",
        "correlation_difference_bootstrap",
        "within_family",
        "rank_stability",
        "dimension_sensitivity",
    }

    agreement = {row["method"]: row for row in first["agreement"]}
    assert agreement["pearson"]["estimate"] < 1.0
    assert agreement["spearman"]["estimate"] < 1.0
    assert agreement["kendall"]["estimate"] < 1.0
    assert all(row["n"] == 140 and row["unit"] == "candidate" for row in agreement.values())
    assert all(row["n_candidates"] == 140 and row["n_poems"] == 20 for row in agreement.values())
    assert all(row["resampling_unit"] == "record_id" for row in agreement.values())
    assert all(row["p_raw"] == row["p_block_permutation"] for row in agreement.values())
    assert all(row["permutation_requested"] == 60 for row in agreement.values())
    assert all(0 <= row["permutation_valid_fraction"] <= 1 for row in agreement.values())

    gpt_human = {
        row["method"]: row
        for row in first["judge_human_alignment"]
        if row["judge"] == "gpt"
    }
    assert gpt_human["pearson"]["estimate"] == pytest.approx(1.0)
    assert gpt_human["spearman"]["estimate"] == pytest.approx(1.0)
    assert gpt_human["kendall"]["estimate"] == pytest.approx(1.0)
    assert all(
        row["n"] == 140
        and row["unit"] == "candidate"
        and row["n_candidates"] == 140
        and row["n_poems"] == 20
        and row["resampling_unit"] == "record_id"
        for row in first["judge_human_alignment"]
    )

    differences = {row["method"]: row for row in first["correlation_difference_bootstrap"]}
    assert differences["pearson"]["difference_gpt_minus_deepseek"] > 0
    assert differences["spearman"]["difference_gpt_minus_deepseek"] > 0
    assert differences["kendall"]["difference_gpt_minus_deepseek"] > 0
    assert all(
        row["n"] == 140
        and row["unit"] == "candidate"
        and row["n_candidates"] == 140
        and row["n_poems"] == 20
        and row["resampling_unit"] == "record_id"
        for row in differences.values()
    )

    primary = first["self_preference"][0]
    assert primary["metric"] == "average_score"
    assert primary["analysis"] == "fixed_effects_ols_clustered"
    assert primary["mean_interaction"] == pytest.approx(1.5)
    assert primary["direction"] == "deepseek_judge_extra_for_deepseek_family"
    assert "C(record_id)" in primary["formula"] and "C(system_id)" in primary["formula"]
    assert primary["covariance"] == "CR1 cluster-robust by record_id"
    assert primary["n_poems"] == 20 and primary["n_systems"] == 7
    assert primary["unit"] == "candidate"
    assert primary["n"] == 140 and primary["resampling_unit"] == "record_id"
    assert primary["leave_one_system_out_min"] == pytest.approx(1.5)
    assert primary["leave_one_system_out_max"] == pytest.approx(1.5)
    assert any(row["analysis"] == "poem_contrast_descriptive" for row in first["self_preference"])
    assert len(first["dimension_sensitivity"]) == len(cross.JUDGE_DIMENSIONS)
    assert all(row["analysis"] == "fixed_effects_ols_clustered" for row in first["dimension_sensitivity"])
    assert all(row["mean_interaction"] == pytest.approx(1.5) for row in first["dimension_sensitivity"])
    assert all(
        row["n"] == 140
        and row["unit"] == "candidate"
        and row["n_candidates"] == 140
        and row["n_poems"] == 20
        and row["resampling_unit"] == "record_id"
        for row in first["dimension_sensitivity"]
    )

    within = rows_by(first, "within_family", judge="gpt", family="deepseek")
    assert len(within) == 1
    assert within[0]["delta_t_minus_nt"] == pytest.approx(1.0)
    assert within[0]["direction_agreement"] == "same"
    assert "p_raw" in within[0] and "p_holm" in within[0]

    rank_types = {row["row_type"] for row in first["rank_stability"]}
    assert rank_types == {"system", "family", "summary"}
    system_rows = rows_by(first, "rank_stability", row_type="system")
    family_rows = rows_by(first, "rank_stability", row_type="family")
    assert len(system_rows) == 7
    assert len(family_rows) == 4
    deepseek_system = next(row for row in system_rows if row["system_id"] == DS_HIGH)
    assert deepseek_system["mean_difference_gpt_minus_deepseek"] == pytest.approx(-1.5)
    assert deepseek_system["mean_difference_ci_low"] == pytest.approx(-1.5)
    assert deepseek_system["mean_difference_ci_high"] == pytest.approx(-1.5)
    assert deepseek_system["deepseek_top1_probability"] == pytest.approx(1.0)
    assert deepseek_system["gpt_top1_probability"] == pytest.approx(0.0)
    assert deepseek_system["deepseek_bootstrap_mean_rank"] == pytest.approx(deepseek_system["deepseek_rank"])
    assert deepseek_system["gpt_bootstrap_mean_rank"] == pytest.approx(deepseek_system["gpt_rank"])
    assert 0 <= deepseek_system["deepseek_bottom1_probability"] <= 1
    assert 0 <= deepseek_system["gpt_bottom1_probability"] <= 1

    deepseek_family = next(row for row in family_rows if row["family_id"] == "DeepSeek")
    assert deepseek_family["mean_difference_gpt_minus_deepseek"] == pytest.approx(-1.5)
    assert deepseek_family["deepseek_top1_probability"] == pytest.approx(1.0)
    assert deepseek_family["gpt_top1_probability"] == pytest.approx(0.0)
    baidu_family = next(row for row in family_rows if row["family_id"] == "Baidu")
    assert baidu_family["deepseek_bottom1_probability"] == pytest.approx(1.0)
    assert baidu_family["gpt_bottom1_probability"] == pytest.approx(1.0)

    rank_summaries = rows_by(first, "rank_stability", row_type="summary")
    assert len(rank_summaries) == 6
    assert {row["level"] for row in rank_summaries} == {"system", "family"}
    assert {row["method"] for row in rank_summaries} == {
        "pearson",
        "spearman",
        "kendall",
    }
    assert all(row["n"] == 20 for row in rank_summaries)
    assert all(row["unit"] == "record_id" for row in rank_summaries)
    system_summary = next(
        row for row in rank_summaries if row["level"] == "system" and row["method"] == "pearson"
    )
    assert system_summary["n_entities"] == 7
    assert system_summary["observed_top_identity_agreement"] == 0
    assert system_summary["observed_bottom_identity_agreement"] == 1
    assert system_summary["bootstrap_same_top_probability"] == pytest.approx(0.0)
    assert system_summary["bootstrap_same_bottom_probability"] == pytest.approx(1.0)
    family_summary = next(
        row for row in rank_summaries if row["level"] == "family" and row["method"] == "pearson"
    )
    assert family_summary["n_entities"] == 4
    assert family_summary["observed_top_deepseek"] == "DeepSeek"
    assert family_summary["observed_top_gpt"] == "Claude"


@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "missing", "system_set", "human_missing"])
def test_crossjudge_requires_exact_20_by_7_pairing_and_finite_values(bad: str) -> None:
    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    if bad == "duplicate":
        gpt_rows.append(dict(gpt_rows[0]))
    elif bad == "nonfinite":
        deepseek_rows[0]["average_score"] = math.inf
    elif bad == "missing":
        gpt_rows.pop()
    elif bad == "system_set":
        gpt_rows[0]["system_id"] = "unexpected"
    else:
        human_rows = [row for row in human_rows if not (row["record_id"] == "p00" and row["system_id"] == ORIGINAL_SYSTEMS[0])]
    with pytest.raises(ValueError):
        cross.analyze_crossjudge(
            deepseek_rows,
            gpt_rows,
            human_rows,
            seed=1,
            bootstrap_samples=20,
            permutation_samples=20,
        )


def test_human_composite_allows_known_sparse_cells_and_reports_policy() -> None:
    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    human_rows = human_rows[2:]
    result = cross.analyze_crossjudge(
        deepseek_rows,
        gpt_rows,
        human_rows,
        seed=5,
        bootstrap_samples=20,
        permutation_samples=20,
    )
    assert all(row["n"] == 140 for row in result["judge_human_alignment"])
    assert all(row["human_missing_cells"] == 2 for row in result["judge_human_alignment"])
    assert all(">=6_of_7" in row["human_missing_policy"] for row in result["judge_human_alignment"])


@pytest.mark.parametrize("case", ["duplicate", "missing_too_many", "range", "noninteger"])
def test_human_panel_rejects_invalid_grid(case: str) -> None:
    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    if case == "duplicate":
        human_rows.append(dict(human_rows[0]))
    elif case == "missing_too_many":
        target = human_rows[0]
        human_rows = [
            row
            for row in human_rows
            if not (
                row["record_id"] == target["record_id"]
                and row["system_id"] == target["system_id"]
                and row["dim"] == target["dim"]
                and row["annotator"] in {"A01", "A02"}
            )
        ]
    elif case == "range":
        human_rows[0]["score"] = 7
    else:
        human_rows[0]["score"] = 3.5
    with pytest.raises(ValueError):
        cross.analyze_crossjudge(
            deepseek_rows,
            gpt_rows,
            human_rows,
            seed=2,
            bootstrap_samples=10,
            permutation_samples=10,
        )


def test_fixed_effect_interaction_controls_shared_non_deepseek_system_offset() -> None:
    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    for rows in [deepseek_rows, gpt_rows]:
        for row in rows:
            if row["system_id"] == "baidu_translate":
                row["average_score"] += 4
                for dimension in cross.JUDGE_DIMENSIONS:
                    row[dimension] += 4
    result = cross.analyze_crossjudge(
        deepseek_rows,
        gpt_rows,
        human_rows,
        seed=8,
        bootstrap_samples=20,
        permutation_samples=20,
    )
    assert result["self_preference"][0]["mean_interaction"] == pytest.approx(1.5)


def test_constant_correlations_are_reported_not_fatal() -> None:
    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    for rows in [deepseek_rows, gpt_rows]:
        for row in rows:
            row["average_score"] = 4.0
            for dimension in cross.JUDGE_DIMENSIONS:
                row[dimension] = 4.0
    result = cross.analyze_crossjudge(
        deepseek_rows,
        gpt_rows,
        human_rows,
        seed=4,
        bootstrap_samples=10,
        permutation_samples=10,
    )
    assert all(row["estimate"] is None for row in result["agreement"])
    assert all(row["bootstrap_valid"] == 0 for row in result["agreement"])
    assert all(row["permutation_valid"] == 0 for row in result["agreement"])
    assert all(row["meets_min_valid_fraction"] == 0 for row in result["agreement"])


def test_verify_only_validates_but_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows = make_intensity_data()
    paths = {
        "original_scores": tmp_path / "original_scores.jsonl",
        "supplemental_scores": tmp_path / "supplemental_scores.jsonl",
        "original_outputs": tmp_path / "original_outputs.jsonl",
        "supplemental_outputs": tmp_path / "supplemental_outputs.jsonl",
        "judge": tmp_path / "judge.jsonl",
    }
    for key, rows in zip(paths, [original_scores, supplemental_scores, original_outputs, supplemental_outputs, judge_rows]):
        write_jsonl(paths[key], rows)
    attempts_a = tmp_path / "attempts_a.jsonl"
    attempts_b = tmp_path / "attempts_b.jsonl"
    write_jsonl(attempts_a, original_outputs)
    write_jsonl(attempts_b, supplemental_outputs)
    paired_out = tmp_path / "paired.csv"
    resources_out = tmp_path / "resources.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stats_intensity.py",
            "--original-scores", str(paths["original_scores"]),
            "--supplemental-scores", str(paths["supplemental_scores"]),
            "--original-outputs", str(paths["original_outputs"]),
            "--supplemental-outputs", str(paths["supplemental_outputs"]),
            "--attempts-path", str(attempts_a),
            "--attempts-path", str(attempts_b),
            "--judge-b-scores", str(paths["judge"]),
            "--paired-output", str(paired_out),
            "--resources-output", str(resources_out),
            "--expected-auto-count", "4",
            "--expected-judge-count", "3",
            "--bootstrap-samples", "20",
            "--verify-only",
        ],
    )
    assert intensity.main() == 0
    assert not paired_out.exists()
    assert not resources_out.exists()

    deepseek_rows, gpt_rows, human_rows = make_crossjudge_data()
    deepseek_path = tmp_path / "deepseek.jsonl"
    gpt_path = tmp_path / "gpt.jsonl"
    human_path = tmp_path / "human.csv"
    write_jsonl(deepseek_path, deepseek_rows)
    write_jsonl(gpt_path, gpt_rows)
    write_csv(human_path, human_rows)
    outputs = {name: tmp_path / f"{name}.csv" for name in cross.OUTPUT_NAMES}
    argv = [
        "stats_crossjudge.py",
        "--deepseek-scores", str(deepseek_path),
        "--gpt-scores", str(gpt_path),
        "--human-eval-long", str(human_path),
        "--bootstrap-samples", "20",
        "--permutation-samples", "20",
        "--verify-only",
    ]
    for name, path in outputs.items():
        argv.extend([f"--{name.replace('_', '-')}-output", str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    assert cross.main() == 0
    assert not any(path.exists() for path in outputs.values())


@pytest.mark.parametrize("module, output_count", [(intensity, 2), (cross, 7)])
def test_atomic_csv_batch_preserves_all_existing_outputs_on_write_failure(
    module, output_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [(tmp_path / f"out_{index}.csv", [{"value": index}]) for index in range(output_count)]
    for path, _ in outputs:
        path.write_text("OLD\n", encoding="utf-8")
    original_write = module.write_csv
    calls = 0

    def flaky_write(path: Path, rows: list[dict]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original_write(path, rows)

    monkeypatch.setattr(module, "write_csv", flaky_write)
    with pytest.raises(OSError, match="injected write failure"):
        module.atomic_write_csvs(outputs)
    assert all(path.read_text(encoding="utf-8") == "OLD\n" for path, _ in outputs)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("module, output_count", [(intensity, 2), (cross, 7)])
def test_atomic_csv_batch_rolls_back_second_replacement_failure_byte_for_byte(
    module, output_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [(tmp_path / f"out_{index}.csv", [{"value": index}]) for index in range(output_count)]
    originals = {}
    for index, (path, _) in enumerate(outputs):
        payload = b"OLD\x00" + bytes([index]) + b"\r\n"
        path.write_bytes(payload)
        originals[path] = payload
    real_replace = module.os.replace
    destination_replacements = 0

    def fail_second_temp_replace(source, destination):
        nonlocal destination_replacements
        if str(source).endswith(".tmp"):
            destination_replacements += 1
            if destination_replacements == 2:
                raise OSError("injected replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_second_temp_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        module.atomic_write_csvs(outputs)
    assert {path: path.read_bytes() for path, _ in outputs} == originals
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize("module", [intensity, cross])
def test_atomic_csv_batch_rejects_duplicate_resolved_destinations(
    module, tmp_path: Path
) -> None:
    destination = tmp_path / "same.csv"
    destination.write_bytes(b"ORIGINAL")
    duplicate_spelling = tmp_path / "subdir" / ".." / "same.csv"
    with pytest.raises(ValueError, match="duplicate"):
        module.atomic_write_csvs(
            [(destination, [{"value": 1}]), (duplicate_spelling, [{"value": 2}])]
        )
    assert destination.read_bytes() == b"ORIGINAL"
