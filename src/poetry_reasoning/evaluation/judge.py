from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import requests

from poetry_reasoning.baselines.common import (
    append_jsonl,
    is_zh_en_record_with_reference,
    now_iso,
    read_jsonl,
    write_jsonl,
)


SOURCE_ORDER = {
    "modern_chinese_poetry": 0,
    "poetry_international_chinese": 1,
    "paper_republic_read": 2,
    "belt_road_literary_network": 3,
}

SOURCE_LABELS = {
    "modern_chinese_poetry": "21st Century Chinese Poetry",
    "poetry_international_chinese": "Poetry International Chinese",
    "paper_republic_read": "Read Paper Republic",
    "belt_road_literary_network": "Belt and Road Literary Network",
}

OVERALL_SOURCE_ID = "overall_modern_poetry"
OVERALL_SOURCE_LABEL = "Overall Modern Poetry"
JUDGE_ID = "deepseek_v4_pro_non_thinking"
JUDGE_PROMPT_VERSION = "agents_md_judge_v1_20260514"

DEFAULT_CANDIDATE_SYSTEM_IDS = [
    "baidu_translate",
    "deepseek_v4_flash_non_thinking",
    "deepseek_v4_flash_thinking",
    "qwen36_plus_non_thinking",
    "qwen36_plus_thinking",
    "claude_sonnet46_non_thinking",
    "claude_sonnet46_thinking",
]

JUDGE_DIMENSIONS = [
    "Semantic Fidelity",
    "Similarity to Reference",
    "Imagery and Rhetoric",
    "Thought and Emotion",
    "Lineation and Rhythm",
    "Modernity and Defamiliarization",
    "Cultural and Idiomatic Transfer",
    "Voice, Tone, and Style",
    "Poeticity",
    "English Naturalness",
    "Overall Impression",
]

JUDGE_PROMPT_TEMPLATE = """You are a professional bilingual evaluator of Chinese-to-English modern poetry translation.
Your task is to evaluate anonymous candidate English translations of a Chinese modern poem. You will receive: 1. the Chinese source poem, 2. one authoritative human English reference translation, 3. one or more anonymous candidate translations.
Important rules: Do not infer or mention the model, vendor, system identity, data source, or generation order. Use the authoritative human reference as an important benchmark, but not as the only acceptable translation. A candidate may differ from the reference and still score highly if it faithfully and poetically renders the Chinese source. Evaluate the final translated poem only. Do not reward or penalize hidden reasoning. Give scores from 0 to 10 in increments of 0.5. Higher is better. For each candidate, score every dimension and provide a concise reason for each dimension. Output strict JSON only. Do not use Markdown.
Evaluation dimensions: 1. Semantic Fidelity: preservation of the source poem's core meaning, events, relations, images, and implied content. 2. Similarity to Reference: thematic, stylistic, and interpretive similarity to the authoritative human reference, without requiring word-for-word matching. 3. Imagery and Rhetoric: preservation or creative transformation of images, metaphors, personification, parallelism, ambiguity, and rhetorical effects. 4. Thought and Emotion: conveyance of the poem's intellectual movement, emotional pressure, mood, and deeper thought. 5. Lineation and Rhythm: effectiveness of line breaks, pauses, pacing, rhythm, sound patterning, and free-verse musicality in English. 6. Modernity and Defamiliarization: retention of modern poetic texture, experimental quality, estrangement, unusual collocations, and non-conventional perception. 7. Cultural and Idiomatic Transfer: handling of culture-specific terms, historical references, idioms, proper names, local images, and Chinese-specific expressions. 8. Voice, Tone, and Style: preservation of the source poem's voice and stylistic posture, such as quietness, irony, fragmentation, solemnity, colloquiality, lyricism, or experimental density. 9. Poeticity: poetic density, suggestiveness, aesthetic tension, image resonance, and memorable language. 10. English Naturalness: whether the translation reads as strong English poetry rather than rigid translationese, while allowing poetic deviation from ordinary grammar. 11. Overall Impression: the overall strength of the candidate translation.
Return strict JSON with this schema: {{"candidate_scores":[{{"candidate_id":"A","scores":{{"Semantic Fidelity":0.0,"Similarity to Reference":0.0,"Imagery and Rhetoric":0.0,"Thought and Emotion":0.0,"Lineation and Rhythm":0.0,"Modernity and Defamiliarization":0.0,"Cultural and Idiomatic Transfer":0.0,"Voice, Tone, and Style":0.0,"Poeticity":0.0,"English Naturalness":0.0,"Overall Impression":0.0}},"reasons":{{"Semantic Fidelity":"brief reason","Similarity to Reference":"brief reason","Imagery and Rhetoric":"brief reason","Thought and Emotion":"brief reason","Lineation and Rhythm":"brief reason","Modernity and Defamiliarization":"brief reason","Cultural and Idiomatic Transfer":"brief reason","Voice, Tone, and Style":"brief reason","Poeticity":"brief reason","English Naturalness":"brief reason","Overall Impression":"brief reason"}},"average_score":0.0,"major_errors":["brief error label if any"]}}],"ranking":["A","B"],"ranking_reason":"brief comparative reason"}}
Chinese source poem:
{source_zh}
Authoritative human English reference translation:
{reference_en}
Anonymous candidate translations:
{candidate_translations}
"""


def _redact_secret(text: str) -> str:
    redacted = text or ""
    for key, value in os.environ.items():
        if ("KEY" in key or "TOKEN" in key or "SECRET" in key or "PASSWORD" in key) and value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return None if math.isnan(number) else number


def _is_half_step(value: float) -> bool:
    return abs(value * 2 - round(value * 2)) < 1e-9


def _stable_seed(seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_judge_manifest(
    dataset_path: Path,
    manifest_path: Path,
    seed: int = 20260513,
    sample_per_source: int = 5,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    if manifest_path.exists() and not overwrite:
        return read_jsonl(manifest_path)

    dataset_rows = [row for row in read_jsonl(dataset_path) if is_zh_en_record_with_reference(row)]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        source_id = str(row.get("source_id") or "unknown")
        if source_id in SOURCE_ORDER:
            by_source[source_id].append(row)

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for source_id in sorted(SOURCE_ORDER, key=SOURCE_ORDER.get):
        rows = sorted(by_source.get(source_id, []), key=lambda row: str(row.get("record_id", "")))
        if len(rows) < sample_per_source:
            raise ValueError(f"Need {sample_per_source} records for {source_id}, found {len(rows)}.")
        sampled = rows if len(rows) == sample_per_source else rng.sample(rows, sample_per_source)
        for row in sorted(sampled, key=lambda item: str(item.get("record_id", ""))):
            source_zh = (row.get("original_zh") or row.get("original_text") or "").strip()
            reference_en = (row.get("translation_en") or row.get("translation_text") or "").strip()
            selected.append(
                {
                    "record_id": row["record_id"],
                    "source_id": source_id,
                    "source_name": row.get("source_name", ""),
                    "language_pair": "zh-en",
                    "source_zh": source_zh,
                    "reference_en": reference_en,
                    "judge_sample_seed": seed,
                    "judge_sample_per_source": sample_per_source,
                    "manifest_created_at": now_iso(),
                }
            )
    write_jsonl(manifest_path, selected)
    return selected


def latest_outputs_by_key(outputs_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(outputs_path):
        system_id = str(row.get("system_id") or "")
        record_id = str(row.get("record_id") or "")
        if system_id and record_id:
            latest[(system_id, record_id)] = row
    return latest


def build_anonymous_candidates(
    record: dict[str, Any],
    outputs_by_key: dict[tuple[str, str], dict[str, Any]],
    system_ids: list[str],
    seed: int = 20260513,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for system_id in system_ids:
        output = outputs_by_key.get((system_id, record["record_id"]))
        if not output or output.get("status") != "ok" or not (output.get("output_text") or "").strip():
            missing.append({"system_id": system_id, "reason": "missing_or_not_ok"})
            continue
        candidates.append(
            {
                "system_id": system_id,
                "system_name": output.get("system_name") or system_id,
                "translation": (output.get("output_text") or "").strip(),
            }
        )

    rng = random.Random(_stable_seed(seed, record["record_id"]))
    rng.shuffle(candidates)
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = chr(ord("A") + index)
    return candidates, missing


def format_candidate_translations(candidates: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"Candidate {candidate['candidate_id']}:\n{candidate['translation']}" for candidate in candidates
    )


def build_judge_prompt(record: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        source_zh=(record.get("source_zh") or "").strip(),
        reference_en=(record.get("reference_en") or "").strip(),
        candidate_translations=format_candidate_translations(candidates),
    )


class DeepSeekJudgeClient:
    judge_id = JUDGE_ID

    def __init__(self) -> None:
        self.api_key = os.getenv("JUDGE_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = os.getenv("JUDGE_DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.base_url = os.getenv("JUDGE_DEEPSEEK_BASE_URL") or os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/chat/completions",
        )

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Missing JUDGE_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY."
        return True, ""

    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def evaluate(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("JUDGE_TEMPERATURE", "0")),
            "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", "8192")),
            "stream": False,
        }
        if os.getenv("JUDGE_DISABLE_THINKING_PARAM", "1") != "0":
            payload["thinking"] = {"type": "disabled"}
        if os.getenv("JUDGE_RESPONSE_FORMAT_JSON", "0") == "1":
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(
            self._chat_url(),
            headers=headers,
            json=payload,
            timeout=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "600")),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Judge HTTP {response.status_code}: {_redact_secret(response.text[:1000])}")
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        return {
            "raw_response": data,
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "token_usage": data.get("usage") or {},
        }


def parse_strict_json(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response root must be a JSON object.")
    return parsed


def validate_judge_json(parsed: dict[str, Any], expected_candidate_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_candidate_ids)
    candidate_scores = parsed.get("candidate_scores")
    if not isinstance(candidate_scores, list):
        raise ValueError("candidate_scores must be a list.")

    seen: set[str] = set()
    for candidate in candidate_scores:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate score must be an object.")
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id not in expected:
            raise ValueError(f"Unexpected candidate_id: {candidate_id}")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)

        scores = candidate.get("scores")
        reasons = candidate.get("reasons")
        if not isinstance(scores, dict) or not isinstance(reasons, dict):
            raise ValueError(f"{candidate_id} must contain scores and reasons objects.")
        dimension_values: list[float] = []
        for dimension in JUDGE_DIMENSIONS:
            score = _safe_float(scores.get(dimension))
            if score is None or score < 0 or score > 10 or not _is_half_step(score):
                raise ValueError(f"{candidate_id} invalid score for {dimension}: {scores.get(dimension)}")
            if not isinstance(reasons.get(dimension), str) or not reasons.get(dimension, "").strip():
                raise ValueError(f"{candidate_id} missing reason for {dimension}.")
            dimension_values.append(score)
        candidate["computed_average_score"] = statistics.fmean(dimension_values)

        if not isinstance(candidate.get("major_errors", []), list):
            raise ValueError(f"{candidate_id} major_errors must be a list.")
        average_score = _safe_float(candidate.get("average_score"))
        if average_score is None or average_score < 0 or average_score > 10:
            raise ValueError(f"{candidate_id} invalid average_score: {candidate.get('average_score')}")

    if seen != expected:
        raise ValueError(f"candidate_scores ids {sorted(seen)} do not match expected {sorted(expected)}.")

    ranking = parsed.get("ranking")
    if not isinstance(ranking, list) or set(str(item) for item in ranking) != expected:
        raise ValueError("ranking must contain each candidate id exactly once.")
    if not isinstance(parsed.get("ranking_reason"), str) or not parsed.get("ranking_reason", "").strip():
        raise ValueError("ranking_reason must be a non-empty string.")
    return parsed


def evaluate_record_with_retries(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    judge: DeepSeekJudgeClient,
    max_retries: int = 2,
) -> dict[str, Any]:
    prompt = build_judge_prompt(record, candidates)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    errors: list[str] = []
    last_raw: dict[str, Any] = {}
    start = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            response = judge.evaluate(prompt)
            last_raw = response
            parsed = validate_judge_json(parse_strict_json(response["content"]), candidate_ids)
            return {
                "status": "ok",
                "parsed_response": parsed,
                "raw_response": response["raw_response"],
                "reasoning_content": response.get("reasoning_content", ""),
                "token_usage": response.get("token_usage", {}),
                "latency_seconds": time.perf_counter() - start,
                "attempts": attempt + 1,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(_redact_secret(f"{type(exc).__name__}: {exc}"))
            if attempt < max_retries:
                time.sleep(float(os.getenv("JUDGE_RETRY_BASE_SECONDS", "2")) * (2 ** attempt))
    return {
        "status": "error",
        "parsed_response": {},
        "raw_response": last_raw.get("raw_response", last_raw),
        "reasoning_content": last_raw.get("reasoning_content", ""),
        "token_usage": last_raw.get("token_usage", {}),
        "latency_seconds": time.perf_counter() - start,
        "attempts": max_retries + 1,
        "errors": errors,
    }


def run_judge_evaluation(
    manifest: list[dict[str, Any]],
    outputs_path: Path,
    raw_path: Path,
    mapping_path: Path,
    system_ids: list[str] | None = None,
    seed: int = 20260513,
    limit: int = 0,
    force: bool = False,
    retry_failed: bool = False,
    sleep_seconds: float = 0.2,
) -> None:
    system_ids = system_ids or DEFAULT_CANDIDATE_SYSTEM_IDS
    if force:
        for path in [raw_path, mapping_path]:
            if path.exists():
                path.unlink()
    outputs_by_key = latest_outputs_by_key(outputs_path)
    judge = DeepSeekJudgeClient()
    available, reason = judge.available()
    if not available:
        raise RuntimeError(reason)

    existing_ok: set[str] = set()
    existing_non_ok: set[str] = set()
    if raw_path.exists() and not force:
        for row in read_jsonl(raw_path):
            record_id = str(row.get("record_id") or "")
            if row.get("status") == "ok":
                existing_ok.add(record_id)
            else:
                existing_non_ok.add(record_id)

    records = manifest[:limit] if limit else manifest
    for index, record in enumerate(records, start=1):
        record_id = record["record_id"]
        if record_id in existing_ok or (record_id in existing_non_ok and not retry_failed):
            print(f"[judge] {index}/{len(records)} {record_id} skipped", flush=True)
            continue
        candidates, missing = build_anonymous_candidates(record, outputs_by_key, system_ids, seed=seed)
        candidate_map = [
            {
                "candidate_id": candidate["candidate_id"],
                "system_id": candidate["system_id"],
                "system_name": candidate["system_name"],
            }
            for candidate in candidates
        ]
        mapping_row = {
            "record_id": record_id,
            "source_id": record.get("source_id", ""),
            "judge_id": judge.judge_id,
            "candidate_map": candidate_map,
            "missing_candidates": missing,
            "created_at": now_iso(),
        }
        if len(candidates) < 2:
            raw_row = {
                **mapping_row,
                "status": "error",
                "error": "Need at least two available candidate translations.",
                "parsed_response": {},
                "raw_response": {},
                "judge_model": judge.model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "latency_seconds": 0.0,
            }
            append_jsonl(raw_path, raw_row)
            append_jsonl(mapping_path, mapping_row)
            print(f"[judge] {index}/{len(records)} {record_id} error insufficient_candidates", flush=True)
            continue
        result = evaluate_record_with_retries(
            record,
            candidates,
            judge,
            max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "2")),
        )
        raw_row = {
            **mapping_row,
            "status": result["status"],
            "error": "; ".join(result["errors"]) if result["status"] != "ok" else "",
            "parsed_response": result["parsed_response"],
            "raw_response": result["raw_response"],
            "reasoning_content": result["reasoning_content"],
            "token_usage": result["token_usage"],
            "judge_model": judge.model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "latency_seconds": result["latency_seconds"],
            "attempts": result["attempts"],
            "created_at": now_iso(),
        }
        append_jsonl(raw_path, raw_row)
        append_jsonl(mapping_path, mapping_row)
        print(f"[judge] {index}/{len(records)} {record_id} {result['status']}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def aggregate_judge_results(raw_path: Path, scores_path: Path, summary_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest_by_record: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        record_id = str(row.get("record_id") or "")
        if record_id:
            latest_by_record[record_id] = row

    score_rows: list[dict[str, Any]] = []
    for row in latest_by_record.values():
        if row.get("status") != "ok":
            continue
        mapping = {item["candidate_id"]: item for item in row.get("candidate_map", [])}
        parsed = row.get("parsed_response") or {}
        ranking = [str(item) for item in parsed.get("ranking", [])]
        rank_position = {candidate_id: index + 1 for index, candidate_id in enumerate(ranking)}
        for candidate_score in parsed.get("candidate_scores", []):
            candidate_id = str(candidate_score.get("candidate_id") or "")
            candidate_meta = mapping.get(candidate_id)
            if not candidate_meta:
                continue
            scores = candidate_score.get("scores") or {}
            score_row: dict[str, Any] = {
                "record_id": row.get("record_id", ""),
                "source_id": row.get("source_id", ""),
                "source_label": SOURCE_LABELS.get(row.get("source_id", ""), row.get("source_id", "")),
                "system_id": candidate_meta["system_id"],
                "system_name": candidate_meta["system_name"],
                "judge_id": row.get("judge_id", JUDGE_ID),
                "judge_model": row.get("judge_model", ""),
                "judge_prompt_version": row.get("judge_prompt_version", ""),
                "candidate_id": candidate_id,
                "average_score": _safe_float(candidate_score.get("computed_average_score")),
                "judge_reported_average_score": _safe_float(candidate_score.get("average_score")),
                "rank_position": rank_position.get(candidate_id),
                "major_errors": candidate_score.get("major_errors", []),
            }
            for dimension in JUDGE_DIMENSIONS:
                score_row[dimension] = _safe_float(scores.get(dimension))
            score_rows.append(score_row)

    write_jsonl(scores_path, score_rows)
    summary_rows = summarize_judge_scores(score_rows)
    write_judge_summary_csv(summary_path, summary_rows)
    return score_rows, summary_rows


def summarize_judge_scores(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row.get("source_id") or ""), str(row.get("system_id") or ""))].append(row)

    for (source_id, system_id), rows in sorted(
        grouped.items(),
        key=lambda item: (SOURCE_ORDER.get(item[0][0], 999), item[0][1]),
    ):
        source_rows.append(_summarize_group(source_id, SOURCE_LABELS.get(source_id, source_id), system_id, rows))

    overall_rows: list[dict[str, Any]] = []
    by_system_source = {(row["system_id"], row["source_id"]): row for row in source_rows}
    system_ids = sorted({row.get("system_id", "") for row in source_rows})
    for system_id in system_ids:
        available_sources = [
            by_system_source[(system_id, source_id)]
            for source_id in sorted(SOURCE_ORDER, key=SOURCE_ORDER.get)
            if (system_id, source_id) in by_system_source
        ]
        if not available_sources:
            continue
        system_name = available_sources[0].get("system_name", system_id)
        overall: dict[str, Any] = {
            "source_id": OVERALL_SOURCE_ID,
            "source_label": OVERALL_SOURCE_LABEL,
            "system_id": system_id,
            "system_name": system_name,
            "n": sum(int(row.get("n") or 0) for row in available_sources),
            "n_sources": len(available_sources),
            "judge_id": available_sources[0].get("judge_id", JUDGE_ID),
            "judge_model": available_sources[0].get("judge_model", ""),
        }
        for dimension in JUDGE_DIMENSIONS:
            overall[dimension] = _mean([
                value for value in (_safe_float(row.get(dimension)) for row in available_sources) if value is not None
            ])
        overall["Average"] = _mean([
            value for value in (_safe_float(row.get("Average")) for row in available_sources) if value is not None
        ])
        overall["Mean Rank"] = _mean([
            value for value in (_safe_float(row.get("Mean Rank")) for row in available_sources) if value is not None
        ])
        overall_rows.append(overall)
    return source_rows + overall_rows


def _summarize_group(source_id: str, source_label: str, system_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_id": source_id,
        "source_label": source_label,
        "system_id": system_id,
        "system_name": rows[0].get("system_name", system_id) if rows else system_id,
        "n": len(rows),
        "n_sources": 1,
        "judge_id": rows[0].get("judge_id", JUDGE_ID) if rows else JUDGE_ID,
        "judge_model": rows[0].get("judge_model", "") if rows else "",
    }
    for dimension in JUDGE_DIMENSIONS:
        summary[dimension] = _mean([
            value for value in (_safe_float(row.get(dimension)) for row in rows) if value is not None
        ])
    summary["Average"] = _mean([
        value for value in (_safe_float(row.get("average_score")) for row in rows) if value is not None
    ])
    summary["Mean Rank"] = _mean([
        value for value in (_safe_float(row.get("rank_position")) for row in rows) if value is not None
    ])
    return summary


def write_judge_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "source_label",
        "system_id",
        "system_name",
        "n",
        "n_sources",
        "judge_id",
        "judge_model",
        *JUDGE_DIMENSIONS,
        "Average",
        "Mean Rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
