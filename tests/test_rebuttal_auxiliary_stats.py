from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import stats_rebuttal_auxiliary as aux


SOURCE_LABELS = {
    "modern_chinese_poetry": "21C",
    "poetry_international_chinese": "PIC",
    "paper_republic_read": "RPR",
    "belt_road_literary_network": "BRLN",
}


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auto_rows: list[dict] = []
    human_rows: list[dict] = []
    judge_rows: list[dict] = []
    record_number = 0
    for source_id, source_label in SOURCE_LABELS.items():
        for source_index in range(3):
            record_id = f"r{record_number:02d}"
            for family_index, (_, (nonthinking, thinking)) in enumerate(
                aux.FAMILIES.items()
            ):
                base_words = 12 + record_number + family_index
                word_counts = {
                    nonthinking: base_words + (record_number % 3),
                    thinking: base_words - 1 + (record_number % 2),
                }
                for variant_index, system_id in enumerate((nonthinking, thinking)):
                    word_count = word_counts[system_id]
                    is_nonthinking = system_id == nonthinking
                    comet = (
                        0.60
                        + 0.001 * record_number
                        + 0.004 * family_index
                        + (0.02 + 0.001 * source_index if is_nonthinking else 0.0)
                    )
                    score = (
                        3.0
                        + 0.02 * word_count
                        + 0.01 * record_number
                        + 0.08 * family_index
                        + (0.18 if is_nonthinking else 0.0)
                        + 0.005 * ((record_number + variant_index) % 3)
                    )
                    auto_rows.append(
                        {
                            "record_id": record_id,
                            "source_id": source_id,
                            "system_id": system_id,
                            "comet": comet,
                            "hypothesis": " ".join(["word"] * word_count),
                        }
                    )
                    judge_rows.append(
                        {
                            "record_id": record_id,
                            "source_id": source_id,
                            "system_id": system_id,
                            "average_score": score,
                        }
                    )
                    for annotator_index, annotator in enumerate(("A01", "A02")):
                        for dim_index, dim in enumerate(("MF", "PT")):
                            human_rows.append(
                                {
                                    "annotator": annotator,
                                    "record_id": record_id,
                                    "source": source_label,
                                    "system_id": system_id,
                                    "dim": dim,
                                    "score": score
                                    + 0.02 * annotator_index
                                    + 0.01 * dim_index,
                                }
                            )
            record_number += 1
    return (
        pd.DataFrame(human_rows),
        pd.DataFrame(auto_rows),
        pd.DataFrame(judge_rows),
    )


def test_annotator_direction_reports_family_counts_and_ranges():
    rows: list[dict] = []
    advantages = {
        "A01": [1.0, 1.0, 1.0],
        "A02": [-1.0, 0.0, 1.0],
    }
    for annotator, family_advantages in advantages.items():
        for family_index, (_, (nonthinking, thinking)) in enumerate(
            aux.FAMILIES.items()
        ):
            for record_id in ("r1", "r2"):
                for dim in ("MF", "PT"):
                    rows.append(
                        {
                            "annotator": annotator,
                            "record_id": record_id,
                            "source": "21C",
                            "system_id": nonthinking,
                            "dim": dim,
                            "score": 3.0 + family_advantages[family_index],
                        }
                    )
                    rows.append(
                        {
                            "annotator": annotator,
                            "record_id": record_id,
                            "source": "21C",
                            "system_id": thinking,
                            "dim": dim,
                            "score": 3.0,
                        }
                    )

    result = aux.build_annotator_direction(
        pd.DataFrame(rows), input_sha256="human.csv=abc"
    )

    assert len(result) == 6
    deepseek = result.loc[result["family"] == "deepseek_v4_flash"]
    assert set(deepseek["family_nt_gt_t_annotators"]) == {1}
    assert set(deepseek["family_total_annotators"]) == {2}
    assert set(deepseek["family_delta_min"]) == {-1.0}
    assert set(deepseek["family_delta_max"]) == {1.0}
    assert set(result["n"]) == {2}
    assert set(result["unit"]) == {
        "record_id paired NT-minus-T after per-candidate dimension mean"
    }


def test_panel_comet_resampling_is_fixed_seed_and_source_stratified():
    human, auto, _ = _synthetic_frames()
    panel_ids = set(
        human.loc[
            human["record_id"].str[-2:].astype(int) % 3 != 2, "record_id"
        ]
    )

    first = aux.build_panel_representativeness(
        auto,
        panel_ids,
        seed=123,
        n_resamples=200,
        input_sha256="auto.jsonl=abc;human.csv=def",
    )
    second = aux.build_panel_representativeness(
        auto,
        panel_ids,
        seed=123,
        n_resamples=200,
        input_sha256="auto.jsonl=abc;human.csv=def",
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["n"]) == {8}
    assert set(first["n_full"]) == {12}
    assert set(first["panel_source_counts"]) == {
        "belt_road_literary_network:2;modern_chinese_poetry:2;"
        "paper_republic_read:2;poetry_international_chinese:2"
    }
    assert first["resample_p_two_sided"].between(0, 1).all()


def test_excluding_small_sources_uses_only_21c_and_pic_record_pairs():
    human, _, judge = _synthetic_frames()

    result = aux.build_small_source_exclusion(
        human,
        judge,
        human_input_sha256="human.csv=abc",
        judge_input_sha256="judge.jsonl=def",
    )

    assert len(result) == 6
    assert set(result["layer"]) == {"human", "judge"}
    assert set(result["n"]) == {6}
    assert (result["delta_nt_minus_t"] > 0).all()
    assert set(result["sources_kept"]) == {"21C;PIC"}


def test_length_confound_covers_clustered_fe_delta_regressions_and_buckets():
    human, auto, judge = _synthetic_frames()

    result = aux.build_length_confound(
        auto,
        human,
        judge,
        auto_input_sha256="auto.jsonl=abc",
        human_input_sha256="human.csv=def",
        judge_input_sha256="judge.jsonl=ghi",
    )

    fixed = result.loc[result["analysis"] == "candidate_fixed_effect"]
    delta = result.loc[result["analysis"] == "within_family_delta_regression"]
    buckets = result.loc[result["analysis"] == "length_bucket_sensitivity"]
    assert len(fixed) == 2
    assert set(fixed["n"]) == {72}
    assert set(fixed["n_clusters"]) == {12}
    assert np.isfinite(fixed["estimate"]).all()
    assert len(delta) == 6
    assert set(delta["n"]) == {12}
    assert np.isfinite(delta["estimate"]).all()
    assert len(buckets) == 18
    assert set(buckets["bucket"]) == {"short", "middle", "long"}
    assert set(buckets.groupby(["layer", "family"])["n"].sum()) == {12}


def test_run_analysis_writes_aggregate_only_outputs_without_hashes(tmp_path: Path):
    human, auto, judge = _synthetic_frames()
    human_path = tmp_path / "human.csv"
    auto_path = tmp_path / "auto.jsonl"
    judge_path = tmp_path / "judge.jsonl"
    output_dir = tmp_path / "out"
    human.to_csv(human_path, index=False)
    for path, frame in ((auto_path, auto), (judge_path, judge)):
        with path.open("w", encoding="utf-8") as handle:
            for row in frame.to_dict("records"):
                handle.write(json.dumps(row) + "\n")

    outputs = aux.run_analysis(
        human_path,
        auto_path,
        judge_path,
        output_dir,
        seed=123,
        n_resamples=100,
    )
    first_panel_bytes = outputs["panel"].read_bytes()
    outputs_second = aux.run_analysis(
        human_path,
        auto_path,
        judge_path,
        output_dir,
        seed=123,
        n_resamples=100,
    )

    assert first_panel_bytes == outputs_second["panel"].read_bytes()
    assert set(outputs) == {"annotator", "panel", "exclusion", "length"}
    banned = {
        "record_id",
        "poem_id",
        "hypothesis",
        "translation",
        "source_text",
        "candidate_id",
    }
    for name, path in outputs.items():
        assert path.exists(), name
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            assert {"n", "unit", "seed", "method"}.issubset(frame.columns)
            assert not (banned & set(frame.columns))
            assert "input_sha256" not in frame.columns
