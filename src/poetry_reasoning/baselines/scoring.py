from __future__ import annotations

from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

from .common import diagnostic_scores, nonempty_lines, read_jsonl, write_jsonl


METRIC_FIELDS = [
    "comet",
    "bertscore_f1",
    "sacrebleu_sentence",
    "chrfpp_sentence",
    "ter_sentence",
]

SOURCE_ORDER = {
    "modern_chinese_poetry": 0,
    "poetry_international_chinese": 1,
    "paper_republic_read": 2,
    "belt_road_literary_network": 3,
}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _ci95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _paired_line_segments(hypothesis: str, reference: str) -> list[tuple[str, str]]:
    """Pair non-empty poem lines for line-level TER.

    Whole-poem TER is prohibitively slow for long poems because each poem becomes
    a single very long segment. Pairing lines keeps the metric tractable and
    matches the project's form-preservation diagnostics.
    """
    hyp_lines = nonempty_lines(hypothesis)
    ref_lines = nonempty_lines(reference)
    count = max(len(hyp_lines), len(ref_lines))
    return [
        (
            hyp_lines[index] if index < len(hyp_lines) else "",
            ref_lines[index] if index < len(ref_lines) else "",
        )
        for index in range(count)
    ]


def _compute_sacrebleu(
    rows: list[dict[str, Any]],
    metric_errors: list[str],
    compute_ter: bool = True,
) -> dict[str, float | None]:
    try:
        import sacrebleu
        from sacrebleu.metrics import TER
    except Exception as exc:
        metric_errors.append(f"sacrebleu unavailable: {exc}")
        return {"sacrebleu_corpus": None, "chrfpp_corpus": None, "ter_corpus": None}

    hypotheses = [row["hypothesis"] for row in rows]
    references = [row["reference_en"] for row in rows]
    try:
        corpus_bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        corpus_chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
        corpus_ter = None
        ter_metric = TER()
        if compute_ter:
            line_hypotheses: list[str] = []
            line_references: list[str] = []
            for row in rows:
                for hypothesis, reference in _paired_line_segments(row["hypothesis"], row["reference_en"]):
                    line_hypotheses.append(hypothesis)
                    line_references.append(reference)
            corpus_ter = ter_metric.corpus_score(line_hypotheses, [line_references]).score if line_hypotheses else None
        for row in rows:
            row["sacrebleu_sentence"] = sacrebleu.sentence_bleu(row["hypothesis"], [row["reference_en"]]).score
            row["chrfpp_sentence"] = sacrebleu.sentence_chrf(row["hypothesis"], [row["reference_en"]], word_order=2).score
            if compute_ter:
                pairs = _paired_line_segments(row["hypothesis"], row["reference_en"])
                line_hypotheses = [hypothesis for hypothesis, _reference in pairs]
                line_references = [reference for _hypothesis, reference in pairs]
                row["ter_sentence"] = (
                    ter_metric.corpus_score(line_hypotheses, [line_references]).score if line_hypotheses else None
                )
            else:
                row["ter_sentence"] = None
        return {"sacrebleu_corpus": corpus_bleu, "chrfpp_corpus": corpus_chrf, "ter_corpus": corpus_ter}
    except Exception as exc:
        metric_errors.append(f"sacrebleu failed: {exc}")
        return {"sacrebleu_corpus": None, "chrfpp_corpus": None, "ter_corpus": None}


def _compute_bertscore(rows: list[dict[str, Any]], metric_errors: list[str]) -> None:
    try:
        from bert_score import score as bert_score
    except Exception as exc:
        metric_errors.append(f"BERTScore unavailable: {exc}")
        return
    try:
        model_type = os.getenv("BERTSCORE_MODEL", "microsoft/deberta-xlarge-mnli")
        device = os.getenv("BERTSCORE_DEVICE") or None
        batch_size = int(os.getenv("BERTSCORE_BATCH_SIZE", "64"))
        if os.getenv("SCORE_PROGRESS"):
            system_name = rows[0].get("system_name", "unknown") if rows else "unknown"
            print(
                f"[BERTScore] start system={system_name} rows={len(rows)} "
                f"model={model_type} device={device or 'default'} batch_size={batch_size}",
                flush=True,
            )
        hypotheses = [row["hypothesis"] for row in rows]
        references = [row["reference_en"] for row in rows]
        kwargs: dict[str, Any] = {
            "lang": "en",
            "model_type": model_type,
            "batch_size": batch_size,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        _, _, f1 = bert_score(hypotheses, references, **kwargs)
        for row, value in zip(rows, f1.tolist(), strict=True):
            row["bertscore_f1"] = float(value)
        if os.getenv("SCORE_PROGRESS"):
            system_name = rows[0].get("system_name", "unknown") if rows else "unknown"
            print(f"[BERTScore] done system={system_name}", flush=True)
    except Exception as exc:
        metric_errors.append(f"BERTScore failed: {exc}")


def _compute_comet(rows: list[dict[str, Any]], metric_errors: list[str]) -> None:
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as exc:
        metric_errors.append(f"COMET unavailable: {exc}")
        return
    try:
        model_name = os.getenv("COMET_MODEL", "Unbabel/wmt22-comet-da")
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        data = [
            {"src": row["source_zh"], "mt": row["hypothesis"], "ref": row["reference_en"]}
            for row in rows
        ]
        batch_size = int(os.getenv("COMET_BATCH_SIZE", "4"))
        gpus = int(os.getenv("COMET_GPUS", "0"))
        prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
        scores = getattr(prediction, "scores", None) or prediction.get("scores", [])
        for row, value in zip(rows, scores, strict=True):
            row["comet"] = float(value)
    except Exception as exc:
        metric_errors.append(f"COMET failed: {exc}")


def score_outputs(
    manifest_path: Path,
    outputs_path: Path,
    scores_path: Path,
    summary_path: Path,
    metric_profile: str = "all",
    summary_by_source_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scoring_started = time.perf_counter()
    manifest = {row["record_id"]: row for row in read_jsonl(manifest_path)}
    raw_outputs = read_jsonl(outputs_path)
    if not manifest:
        raise ValueError(f"No manifest records found in {manifest_path}.")

    latest_outputs: dict[tuple[str, str], dict[str, Any]] = {}
    for output in raw_outputs:
        record_id = str(output.get("record_id", ""))
        if record_id not in manifest:
            continue
        key = (str(output.get("system_id", "unknown")), record_id)
        latest_outputs[key] = output
    outputs = list(latest_outputs.values())

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    system_names: dict[str, str] = {}
    for output in outputs:
        system_id = output.get("system_id", "unknown")
        by_system[system_id].append(output)
        system_names[system_id] = output.get("system_name", system_id)

    all_scores: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    ok_rows_by_system: dict[str, list[dict[str, Any]]] = {}
    metric_errors_by_system: dict[str, list[str]] = {}
    for system_id in sorted(by_system):
        outputs_for_system = by_system[system_id]
        ok_rows: list[dict[str, Any]] = []
        not_ok = [row for row in outputs_for_system if row.get("status") != "ok"]
        for output in outputs_for_system:
            record = manifest.get(output.get("record_id"))
            if not record:
                continue
            hypothesis = (output.get("output_text") or "").strip()
            score_row = {
                "record_id": record["record_id"],
                "system_id": system_id,
                "system_name": system_names[system_id],
                "status": output.get("status", "unknown"),
                "error": output.get("error", ""),
                "source_id": record.get("source_id", ""),
                "source_name": record.get("source_name", ""),
                "quality_flags": record.get("quality_flags", []),
                "source_zh": record.get("source_zh", ""),
                "reference_en": record.get("reference_en", ""),
                "hypothesis": hypothesis,
                "reasoning_content": output.get("reasoning_content", ""),
                "prompt_version": output.get("prompt_version", ""),
                "latency_seconds": output.get("latency_seconds", 0.0),
            }
            score_row.update(diagnostic_scores(hypothesis, record.get("reference_en", "")))
            if output.get("status") == "ok" and hypothesis:
                ok_rows.append(score_row)
            all_scores.append(score_row)

        metric_errors: list[str] = []
        corpus_metrics = {"sacrebleu_corpus": None, "chrfpp_corpus": None, "ter_corpus": None}
        if ok_rows:
            corpus_metrics = _compute_sacrebleu(
                ok_rows,
                metric_errors,
                compute_ter=metric_profile in {"light", "bertscore", "comet", "all"},
            )
            if metric_profile in {"all", "bertscore"}:
                _compute_bertscore(ok_rows, metric_errors)
            else:
                metric_errors.append(f"BERTScore skipped by metric profile: {metric_profile}")
            if metric_profile not in {"all", "comet"}:
                metric_errors.append(f"COMET skipped by metric profile: {metric_profile}")
        ok_rows_by_system[system_id] = ok_rows
        metric_errors_by_system[system_id] = metric_errors

        per_record_by_key = {(row["system_id"], row["record_id"]): row for row in all_scores}
        for row in ok_rows:
            per_record_by_key[(row["system_id"], row["record_id"])].update(row)

        summary = {
            "system_id": system_id,
            "system_name": system_names[system_id],
            "n_total": len(outputs_for_system),
            "n_ok": len(ok_rows),
            "n_failed": len(not_ok),
            "status": "ok" if ok_rows else "not_available_or_failed",
            "prompt_version": ";".join(
                sorted({str(row.get("prompt_version", "")) for row in outputs_for_system if row.get("prompt_version")})
            ),
            "metric_errors": "; ".join(metric_errors),
            "runtime_seconds_total": sum(
                value for value in (_safe_float(row.get("latency_seconds")) for row in outputs_for_system) if value is not None
            ),
            **corpus_metrics,
        }
        for field in METRIC_FIELDS + ["line_count_diff", "length_ratio", "english_ratio", "empty_output"]:
            values = [_safe_float(row.get(field)) for row in ok_rows]
            clean_values = [value for value in values if value is not None]
            summary[f"{field}_mean"] = _mean(clean_values)
            summary[f"{field}_sd"] = _sd(clean_values)
            summary[f"{field}_ci95"] = _ci95(clean_values)
        if not ok_rows and not_ok:
            first_error = not_ok[0].get("error", "")
            summary["metric_errors"] = summary["metric_errors"] or first_error
        summaries.append(summary)

    if metric_profile in {"all", "comet"}:
        comet_rows = [row for system_rows in ok_rows_by_system.values() for row in system_rows]
        comet_errors: list[str] = []
        if comet_rows:
            _compute_comet(comet_rows, comet_errors)
        for summary in summaries:
            system_id = summary["system_id"]
            if comet_errors:
                metric_errors_by_system[system_id].extend(comet_errors)
            values = [_safe_float(row.get("comet")) for row in ok_rows_by_system.get(system_id, [])]
            clean_values = [value for value in values if value is not None]
            summary["comet_mean"] = _mean(clean_values)
            summary["comet_sd"] = _sd(clean_values)
            summary["comet_ci95"] = _ci95(clean_values)
            summary["metric_errors"] = "; ".join(metric_errors_by_system[system_id])

    scoring_elapsed_total = time.perf_counter() - scoring_started
    rows_with_ok = sum(1 for summary in summaries if summary.get("n_ok"))
    scoring_elapsed_share = scoring_elapsed_total / rows_with_ok if rows_with_ok else 0.0
    for summary in summaries:
        if summary.get("n_ok"):
            summary["scoring_seconds_total"] = scoring_elapsed_share
        else:
            summary["scoring_seconds_total"] = 0.0
        api_seconds = _safe_float(summary.get("runtime_seconds_total")) or 0.0
        scoring_seconds = _safe_float(summary.get("scoring_seconds_total")) or 0.0
        summary["total_seconds_including_scoring"] = api_seconds + scoring_seconds

    write_jsonl(scores_path, all_scores)
    _write_summary_csv(summary_path, summaries)
    if summary_by_source_path:
        source_summaries = _build_source_summaries(
            all_scores,
            summaries,
            metric_profile=metric_profile,
        )
        _write_summary_csv(summary_by_source_path, source_summaries)
    return all_scores, summaries


def _build_source_summaries(
    all_scores: list[dict[str, Any]],
    system_summaries: list[dict[str, Any]] | None = None,
    metric_profile: str = "all",
) -> list[dict[str, Any]]:
    scoring_by_system = {
        str(row.get("system_id")): (_safe_float(row.get("scoring_seconds_total")) or 0.0)
        for row in (system_summaries or [])
    }
    n_ok_by_system = {
        str(row.get("system_id")): int(row.get("n_ok") or 0)
        for row in (system_summaries or [])
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_scores:
        grouped[(str(row.get("source_id") or "unknown"), str(row.get("system_id") or "unknown"))].append(row)

    summaries: list[dict[str, Any]] = []
    for (source_id, system_id), rows in sorted(
        grouped.items(),
        key=lambda item: (SOURCE_ORDER.get(item[0][0], 999), item[0][0], item[0][1]),
    ):
        system_name = rows[0].get("system_name", system_id)
        source_name = rows[0].get("source_name", "")
        ok_rows = [row for row in rows if row.get("status") == "ok" and (row.get("hypothesis") or "").strip()]
        not_ok = [row for row in rows if row.get("status") != "ok"]
        metric_errors: list[str] = []
        corpus_metrics = {"sacrebleu_corpus": None, "chrfpp_corpus": None, "ter_corpus": None}
        if ok_rows:
            corpus_metrics = _compute_sacrebleu(
                ok_rows,
                metric_errors,
                compute_ter=metric_profile in {"light", "bertscore", "comet", "all"},
            )
            if metric_profile not in {"all", "bertscore"}:
                metric_errors.append(f"BERTScore skipped by metric profile: {metric_profile}")
            if metric_profile not in {"all", "comet"}:
                metric_errors.append(f"COMET skipped by metric profile: {metric_profile}")
        summary: dict[str, Any] = {
            "source_id": source_id,
            "source_name": source_name,
            "system_id": system_id,
            "system_name": system_name,
            "n_total": len(rows),
            "n_ok": len(ok_rows),
            "n_failed": len(not_ok),
            "status": "ok" if ok_rows else "not_available_or_failed",
            "prompt_version": ";".join(
                sorted({str(row.get("prompt_version", "")) for row in rows if row.get("prompt_version")})
            ),
            "metric_errors": "; ".join(metric_errors),
            "runtime_seconds_total": sum(
                value for value in (_safe_float(row.get("latency_seconds")) for row in rows) if value is not None
            ),
            **corpus_metrics,
        }
        scoring_total = scoring_by_system.get(system_id, 0.0)
        system_n_ok = n_ok_by_system.get(system_id, 0)
        source_n_ok = len(ok_rows)
        summary["scoring_seconds_total"] = scoring_total * source_n_ok / system_n_ok if system_n_ok else 0.0
        api_seconds = _safe_float(summary.get("runtime_seconds_total")) or 0.0
        summary["total_seconds_including_scoring"] = api_seconds + summary["scoring_seconds_total"]
        for field in METRIC_FIELDS + ["line_count_diff", "length_ratio", "english_ratio", "empty_output"]:
            values = [_safe_float(row.get(field)) for row in ok_rows]
            clean_values = [value for value in values if value is not None]
            summary[f"{field}_mean"] = _mean(clean_values)
            summary[f"{field}_sd"] = _sd(clean_values)
            summary[f"{field}_ci95"] = _ci95(clean_values)
        if not ok_rows and not_ok:
            first_error = not_ok[0].get("error", "")
            summary["metric_errors"] = summary["metric_errors"] or first_error
        summaries.append(summary)
    return summaries


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_id",
        "system_name",
        "source_id",
        "source_name",
        "n_total",
        "n_ok",
        "n_failed",
        "status",
        "prompt_version",
        "runtime_seconds_total",
        "scoring_seconds_total",
        "total_seconds_including_scoring",
        "comet_mean",
        "comet_sd",
        "comet_ci95",
        "bertscore_f1_mean",
        "bertscore_f1_sd",
        "bertscore_f1_ci95",
        "sacrebleu_corpus",
        "chrfpp_corpus",
        "ter_corpus",
        "line_count_diff_mean",
        "length_ratio_mean",
        "english_ratio_mean",
        "empty_output_mean",
        "metric_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
