from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.audit_statistical_release import EXPECTED_FILES, audit_directory, audit_result_file


def _write_csv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_audit_accepts_aggregate_inferential_fields(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    _write_csv(
        path,
        ["analysis", "system_id", "n", "effect", "ci_low", "ci_high", "p_holm"],
        {
            "analysis": "paired_wilcoxon",
            "system_id": "example_system",
            "n": 20,
            "effect": -0.5,
            "ci_low": -0.8,
            "ci_high": -0.2,
            "p_holm": 0.01,
        },
    )
    assert audit_result_file(path) == 1


@pytest.mark.parametrize(
    ("fieldnames", "row", "message"),
    [
        (["analysis", "effect"], {"analysis": "test", "effect": "nan"}, "non-finite"),
        (["analysis", "p_holm"], {"analysis": "test", "p_holm": 1.1}, "probability"),
        (
            ["analysis", "ci_low", "ci_high"],
            {"analysis": "test", "ci_low": 2, "ci_high": 1},
            "reversed interval",
        ),
    ],
)
def test_audit_rejects_invalid_statistical_values(
    tmp_path: Path,
    fieldnames: list[str],
    row: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "test.csv"
    _write_csv(path, fieldnames, row)
    with pytest.raises(ValueError, match=message):
        audit_result_file(path)


@pytest.mark.parametrize(
    "column",
    [
        "record_id",
        "candidate_id",
        "source_zh",
        "reference_en",
        "translation",
        "hypothesis",
        "reasoning_content",
        "raw_response",
        "input_sha256",
        "artifact_path",
        "private_mapping",
    ],
)
def test_audit_rejects_private_or_per_record_columns(tmp_path: Path, column: str) -> None:
    path = tmp_path / "test.csv"
    _write_csv(path, ["analysis", column], {"analysis": "test", column: "private"})
    with pytest.raises(ValueError, match="forbidden column"):
        audit_result_file(path)


@pytest.mark.parametrize(
    "filename",
    ["full397_summary.csv", "intensity_resources.csv", "run_manifest.csv", "raw_scores.csv"],
)
def test_audit_rejects_non_inferential_filenames(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    _write_csv(path, ["analysis", "n", "p_raw"], {"analysis": "test", "n": 2, "p_raw": 1})
    with pytest.raises(ValueError, match="forbidden filename"):
        audit_result_file(path)


def test_directory_audit_requires_exact_release_allowlist(tmp_path: Path) -> None:
    for filename in EXPECTED_FILES:
        _write_csv(
            tmp_path / filename,
            ["analysis", "n", "p_raw"],
            {"analysis": "test", "n": 2, "p_raw": 1},
        )
    assert audit_directory(tmp_path) == len(EXPECTED_FILES)

    (tmp_path / next(iter(EXPECTED_FILES))).unlink()
    with pytest.raises(ValueError, match="missing"):
        audit_directory(tmp_path)


def test_repository_statistical_results_pass_release_audit() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results" / "statistical_tests"
    assert audit_directory(result_dir) == len(EXPECTED_FILES)
