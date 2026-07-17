#!/usr/bin/env python3
"""Check that the published statistical-result directory contains inferential outputs only."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


EXPECTED_FILES = frozenset(
    {
        "alignment_correlation_difference_bootstrap.csv",
        "crossjudge_agreement.csv",
        "crossjudge_dimension_sensitivity.csv",
        "crossjudge_rank_stability.csv",
        "crossjudge_self_preference.csv",
        "crossjudge_within_family.csv",
        "human_iaa_v2.csv",
        "human_paired_v2.csv",
        "intensity_paired.csv",
        "judge_human_alignment.csv",
        "judge_paired_v2.csv",
        "mde.csv",
        "open_qwen35_exclusion_sensitivity.csv",
        "open_qwen35_paired.csv",
        "rebuttal_aux_annotator_direction.csv",
        "rebuttal_aux_exclude_small_sources.csv",
        "rebuttal_aux_length_confound.csv",
        "rebuttal_aux_panel_comet.csv",
        "stats_alignment_v2.csv",
    }
)

FORBIDDEN_FILENAME = re.compile(r"(?:summary|resources?|manifest|raw|mapping|hash|scores?)", re.IGNORECASE)
FORBIDDEN_COLUMNS = {
    "record_id",
    "candidate_id",
    "source_zh",
    "reference_en",
    "translation",
    "hypothesis",
    "reasoning_content",
    "raw_response",
    "input_sha256",
}
FORBIDDEN_COLUMN_FRAGMENT = re.compile(
    r"(?:^|_)(?:sha\d*|hash|path|url|endpoint|mapping)(?:$|_)",
    re.IGNORECASE,
)
ABSOLUTE_PATH = re.compile(r"(?:^|\s)(?:[A-Za-z]:[\\/]|/home/|/Users/|/mnt/)")
HEX_DIGEST = re.compile(r"^[0-9a-f]{40,128}$", re.IGNORECASE)
PROBABILITY_COLUMN = re.compile(
    r"(?:^p(?:_|$)|_p(?:_|$)|probability$|fraction$|^(?:alpha|power)$)",
    re.IGNORECASE,
)
NONFINITE_LITERAL = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def _as_finite_number(value: str, *, path: Path, row_number: int, column: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"non-numeric value in {path.name}, row {row_number}, column {column}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in {path.name}, row {row_number}, column {column}")
    return number


def _audit_statistical_values(path: Path, row_number: int, row: dict[str, str | None]) -> None:
    for column, value in row.items():
        text = (value or "").strip()
        if text.lower() in NONFINITE_LITERAL:
            raise ValueError(f"non-finite value in {path.name}, row {row_number}, column {column}")
        if text and PROBABILITY_COLUMN.search(column):
            probability = _as_finite_number(text, path=path, row_number=row_number, column=column)
            if not 0 <= probability <= 1:
                raise ValueError(f"probability outside [0, 1] in {path.name}, row {row_number}, column {column}")

    for lower_column in row:
        if lower_column.endswith("_low"):
            upper_column = lower_column[:-4] + "_high"
        elif lower_column.endswith("_lo"):
            upper_column = lower_column[:-3] + "_hi"
        else:
            continue
        lower_text = (row.get(lower_column) or "").strip()
        upper_text = (row.get(upper_column) or "").strip()
        if not lower_text or not upper_text:
            continue
        lower = _as_finite_number(lower_text, path=path, row_number=row_number, column=lower_column)
        upper = _as_finite_number(upper_text, path=path, row_number=row_number, column=upper_column)
        if lower > upper:
            raise ValueError(f"reversed interval in {path.name}, row {row_number}: {lower_column}/{upper_column}")


def audit_result_file(path: Path) -> int:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"result is not CSV: {path.name}")
    if FORBIDDEN_FILENAME.search(path.name):
        raise ValueError(f"forbidden filename: {path.name}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path.name}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"duplicate CSV column in {path.name}")
        for column in reader.fieldnames:
            if column != column.strip():
                raise ValueError(f"whitespace in CSV column name in {path.name}: {column!r}")
            normalized = column.strip().lower()
            if normalized in FORBIDDEN_COLUMNS or FORBIDDEN_COLUMN_FRAGMENT.search(normalized):
                raise ValueError(f"forbidden column in {path.name}: {column}")

        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            _audit_statistical_values(path, row_count, row)
            for value in row.values():
                text = (value or "").strip()
                if text.startswith(("http://", "https://")) or ABSOLUTE_PATH.search(text):
                    raise ValueError(f"private location in {path.name}, row {row_count}")
                if HEX_DIGEST.fullmatch(text):
                    raise ValueError(f"digest-like value in {path.name}, row {row_count}")
                lowered = text.lower()
                if "codex" in lowered or "fable5" in lowered:
                    raise ValueError(f"agent attribution in {path.name}, row {row_count}")

    if row_count == 0:
        raise ValueError(f"empty result file: {path.name}")
    return row_count


def audit_directory(directory: Path) -> int:
    if not directory.is_dir():
        raise ValueError(f"statistical result directory missing: {directory}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_FILES - actual)
    extra = sorted(actual - EXPECTED_FILES)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if extra:
            parts.append("unexpected: " + ", ".join(extra))
        raise ValueError("; ".join(parts))
    for filename in sorted(EXPECTED_FILES):
        audit_result_file(directory / filename)
    return len(EXPECTED_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing published inferential CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = audit_directory(args.directory)
    print(f"approved statistical result files: {count}")


if __name__ == "__main__":
    main()
