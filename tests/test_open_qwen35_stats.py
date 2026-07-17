from __future__ import annotations

import math

import numpy as np
from scipy import stats

from scripts import stats_open_qwen35 as oq


def _automatic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(6):
        record_id = f"r{index}"
        for system_id, offset in ((oq.NON_THINKING, 0.0), (oq.THINKING, 0.1)):
            rows.append(
                {
                    "record_id": record_id,
                    "system_id": system_id,
                    "status": "ok",
                    "comet": 0.5 + index * 0.01 - offset,
                    "bertscore_f1": 0.6 + index * 0.01 + offset,
                    "sacrebleu_sentence": 10 + index + offset,
                    "chrfpp_sentence": 30 + index + offset,
                    "ter_sentence": 50 - index + offset,
                    "line_count_diff": index + offset,
                    "length_ratio": 1.0 + (offset if index % 2 else -offset),
                }
            )
    return rows


def _judge_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    differences = [0.0, 0.0, 0.5, -0.2, 0.1, 0.8, 0.3, -0.4]
    for index, thinking_offset in enumerate(differences):
        for system_id, offset in (
            (oq.NON_THINKING, 0.0),
            (oq.THINKING, thinking_offset),
        ):
            rows.append(
                {
                    "record_id": f"j{index}",
                    "system_id": system_id,
                    "average_score": 7.0 + index * 0.1 + offset,
                    "rank_position": 5.0 - offset,
                }
            )
    return rows


def test_analyze_reports_paired_metrics_holm_and_no_record_ids():
    rows = oq.analyze(
        _automatic_rows(),
        _judge_rows(),
        seed=7,
        bootstrap_samples=200,
        expected_automatic_count=6,
        expected_judge_count=8,
    )

    automatic = [row for row in rows if row["layer"] == "automatic"]
    judge = [row for row in rows if row["layer"] == "judge"]
    assert len(automatic) == 7
    assert len(judge) == 2
    assert {row["metric"] for row in automatic} == set(oq.AUTOMATIC_METRICS)
    assert {row["metric"] for row in judge} == {"average_score", "rank_position"}
    assert all(math.isfinite(float(row["p_holm"])) for row in automatic)
    assert all(row["n"] == 6 for row in automatic)
    assert all(row["n"] == 8 for row in judge)
    assert all("record_id" not in row for row in rows)

    comet = next(row for row in automatic if row["metric"] == "comet")
    bertscore = next(row for row in automatic if row["metric"] == "bertscore_f1")
    assert comet["delta_t_minus_nt"] < 0
    assert bertscore["delta_t_minus_nt"] > 0

    average = next(row for row in judge if row["metric"] == "average_score")
    differences = np.asarray([0.0, 0.0, 0.5, -0.2, 0.1, 0.8, 0.3, -0.4])
    expected_p = stats.wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    ).pvalue
    rng = np.random.default_rng(7)
    bootstrap_means = differences[
        rng.integers(0, len(differences), size=(200, len(differences)))
    ].mean(axis=1)
    expected_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    assert average["p_raw"] == expected_p
    np.testing.assert_allclose(
        [average["ci_low"], average["ci_high"]],
        expected_ci,
        rtol=0,
        atol=1e-12,
    )


def test_exclusion_sensitivity_reports_only_aggregate_shifts():
    rows = oq.exclusion_sensitivity(
        _automatic_rows(),
        excluded_record_id="r0",
        expected_automatic_count=6,
    )

    assert {row["metric"] for row in rows} == {"comet", "bertscore_f1"}
    assert all(row["excluded_n"] == 1 for row in rows)
    assert all("record_id" not in row for row in rows)
    assert all(float(row["max_abs_system_mean_shift"]) >= 0 for row in rows)
