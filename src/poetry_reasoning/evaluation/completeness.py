from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from poetry_reasoning.baselines.common import is_zh_en_record_with_reference


class CompletenessError(RuntimeError):
    """Raised when an experiment artifact does not satisfy its completeness gate."""


def check_translation_completeness(
    output_rows: Iterable[Mapping[str, Any]],
    *,
    systems: Sequence[str],
    expected_count: int,
    dataset_rows: Iterable[Mapping[str, Any]] | None = None,
    expected_prompt_version: str | None = None,
) -> dict[str, Any]:
    """Return a strict, side-effect-free completeness report for translation outputs."""
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")

    requested = list(dict.fromkeys(str(system_id) for system_id in systems))
    if not requested:
        raise ValueError("at least one system is required")
    outputs = list(output_rows)

    expected_sources: dict[str, str] | None = None
    eligible_dataset_count: int | None = None
    if dataset_rows is not None:
        eligible = [
            row
            for row in dataset_rows
            if is_zh_en_record_with_reference(dict(row))
        ]
        expected_sources = {
            str(row.get("record_id") or ""): str(row.get("source_id") or "")
            for row in eligible
            if str(row.get("record_id") or "")
        }
        eligible_dataset_count = len(eligible)

    system_reports: dict[str, dict[str, Any]] = {}
    for system_id in requested:
        rows = [row for row in outputs if str(row.get("system_id") or "") == system_id]
        record_ids = [str(row.get("record_id") or "") for row in rows]
        key_counts = Counter((system_id, record_id) for record_id in record_ids)
        duplicate_key_count = sum(count - 1 for count in key_counts.values() if count > 1)

        actual_ids = {record_id for record_id in record_ids if record_id}
        missing_record_count = 0
        unexpected_record_count = 0
        source_mismatch_count = 0
        if expected_sources is not None:
            expected_ids = set(expected_sources)
            missing_record_count = len(expected_ids - actual_ids)
            unexpected_record_count = len(actual_ids - expected_ids)
            source_mismatch_count = sum(
                1
                for row in rows
                if (record_id := str(row.get("record_id") or "")) in expected_sources
                and str(row.get("source_id") or "") != expected_sources[record_id]
            )

        diagnostics = {
            "row_count": len(rows),
            "unique_key_count": len(key_counts),
            "expected_count": expected_count,
            "duplicate_key_count": duplicate_key_count,
            "invalid_status_count": sum(row.get("status") != "ok" for row in rows),
            "empty_output_count": sum(not str(row.get("output_text") or "").strip() for row in rows),
            "missing_source_id_count": sum(not str(row.get("source_id") or "").strip() for row in rows),
            "missing_record_id_count": sum(not record_id for record_id in record_ids),
            "prompt_mismatch_count": (
                sum(str(row.get("prompt_version") or "") != expected_prompt_version for row in rows)
                if expected_prompt_version is not None
                else 0
            ),
            "missing_record_count": missing_record_count,
            "unexpected_record_count": unexpected_record_count,
            "source_mismatch_count": source_mismatch_count,
        }
        count_fields = (
            "duplicate_key_count",
            "invalid_status_count",
            "empty_output_count",
            "missing_source_id_count",
            "missing_record_id_count",
            "prompt_mismatch_count",
            "missing_record_count",
            "unexpected_record_count",
            "source_mismatch_count",
        )
        diagnostics["complete"] = (
            diagnostics["row_count"] == expected_count
            and diagnostics["unique_key_count"] == expected_count
            and all(diagnostics[field] == 0 for field in count_fields)
            and (eligible_dataset_count is None or eligible_dataset_count == expected_count)
        )
        system_reports[system_id] = diagnostics

    return {
        "complete": all(report["complete"] for report in system_reports.values()),
        "expected_count": expected_count,
        "eligible_dataset_count": eligible_dataset_count,
        "systems": system_reports,
    }


def assert_complete(report: Mapping[str, Any]) -> None:
    """Raise if a report does not pass the strict completeness gate."""
    if not report.get("complete"):
        raise CompletenessError("Experiment completeness gate failed")
