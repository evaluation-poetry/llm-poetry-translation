from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "bilingual_modern_chinese_poetry.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
REFERENCE_FORBIDDEN_FIELDS = {"translation_en", "translation_text", "reference_en"}


@dataclass(frozen=True)
class BaselinePaths:
    dataset_path: Path = DEFAULT_DATASET_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR

    @property
    def manifest_path(self) -> Path:
        return self.results_dir / "baseline_manifest_100.jsonl"

    @property
    def outputs_path(self) -> Path:
        return self.results_dir / "baseline_outputs.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.results_dir / "baseline_scores.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.results_dir / "baseline_summary.csv"

    @property
    def report_tex_path(self) -> Path:
        return self.reports_dir / "baseline_results.tex"

    @property
    def report_pdf_path(self) -> Path:
        return self.reports_dir / "baseline_results.pdf"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_dotenv(path: Path | None = None, override: bool = False) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_han(text: str) -> bool:
    return bool(HAN_RE.search(text or ""))


def english_ratio(text: str) -> float:
    text = text or ""
    latin = len(LATIN_RE.findall(text))
    han = len(HAN_RE.findall(text))
    denom = latin + han
    return latin / denom if denom else 0.0


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def line_count_difference(hypothesis: str, reference: str) -> int:
    return abs(len(nonempty_lines(hypothesis)) - len(nonempty_lines(reference)))


def length_ratio(hypothesis: str, reference: str) -> float:
    ref_len = len((reference or "").strip())
    if ref_len == 0:
        return 0.0
    return len((hypothesis or "").strip()) / ref_len


def quality_flags(row: dict[str, Any]) -> list[str]:
    flags = row.get("quality_flags") or []
    if isinstance(flags, str):
        return [flags] if flags else []
    return [str(flag) for flag in flags if str(flag)]


def is_zh_en_record_with_reference(row: dict[str, Any]) -> bool:
    source = (row.get("original_zh") or row.get("original_text") or "").strip()
    reference = (row.get("translation_en") or row.get("translation_text") or "").strip()
    if row.get("language_pair") != "zh-en":
        return False
    if not source or not reference:
        return False
    if not has_han(source):
        return False
    if has_han(reference):
        return False
    return True


def is_clean_zh_en_record(row: dict[str, Any]) -> bool:
    if not is_zh_en_record_with_reference(row):
        return False
    return not quality_flags(row)


def _largest_remainder_allocation(counts: Counter[str], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    raw = {key: sample_size * value / total for key, value in counts.items()}
    allocation = {key: min(counts[key], math.floor(value)) for key, value in raw.items()}
    remaining = sample_size - sum(allocation.values())
    remainders = sorted(raw, key=lambda key: (raw[key] - math.floor(raw[key]), counts[key]), reverse=True)
    while remaining > 0:
        progressed = False
        for key in remainders:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return allocation


def build_manifest(
    dataset_rows: list[dict[str, Any]],
    sample_size: int = 100,
    include_flagged: bool = False,
) -> list[dict[str, Any]]:
    eligible_rows = [
        row for row in dataset_rows
        if is_zh_en_record_with_reference(row) and (include_flagged or not quality_flags(row))
    ]
    if sample_size <= 0:
        sample_size = len(eligible_rows)
    if len(eligible_rows) < sample_size:
        sample_kind = "zh-en records with references" if include_flagged else "clean zh-en records"
        raise ValueError(f"Need {sample_size} {sample_kind}, found {len(eligible_rows)}.")
    source_counts = Counter(str(row.get("source_id") or "unknown") for row in eligible_rows)
    allocation = (
        dict(source_counts)
        if sample_size == len(eligible_rows)
        else _largest_remainder_allocation(source_counts, sample_size)
    )
    selected: list[dict[str, Any]] = []
    per_source_seen: Counter[str] = Counter()
    for row in eligible_rows:
        source_id = str(row.get("source_id") or "unknown")
        if per_source_seen[source_id] >= allocation.get(source_id, 0):
            continue
        source_zh = (row.get("original_zh") or row.get("original_text") or "").strip()
        reference_en = (row.get("translation_en") or row.get("translation_text") or "").strip()
        selected.append(
            {
                "record_id": row["record_id"],
                "source_id": source_id,
                "source_name": row.get("source_name", ""),
                "page_url": row.get("page_url", ""),
                "title_zh": row.get("title_zh", ""),
                "poet_zh": row.get("poet_zh", ""),
                "translator": row.get("translator", ""),
                "quality_flags": quality_flags(row),
                "language_pair": "zh-en",
                "source_zh": source_zh,
                "reference_en": reference_en,
                "source_lines": row.get("original_lines") or nonempty_lines(source_zh),
                "reference_lines": row.get("translation_lines") or nonempty_lines(reference_en),
                "manifest_created_at": now_iso(),
            }
        )
        per_source_seen[source_id] += 1
        if len(selected) == sample_size:
            break
    if len(selected) != sample_size:
        raise ValueError(f"Expected {sample_size} selected records, got {len(selected)}.")
    return selected


def prepare_manifest(
    dataset_path: Path,
    manifest_path: Path,
    sample_size: int = 100,
    overwrite: bool = False,
    include_flagged: bool = False,
) -> list[dict[str, Any]]:
    if manifest_path.exists() and not overwrite:
        return read_jsonl(manifest_path)
    dataset_rows = read_jsonl(dataset_path)
    manifest = build_manifest(dataset_rows, sample_size=sample_size, include_flagged=include_flagged)
    write_jsonl(manifest_path, manifest)
    return manifest


PROMPT_VERSION_DIRECT = "direct_v1"
PROMPT_VERSION_UNDERSTAND_TRANSLATE = "understand_translate_v1"
PROMPT_VERSION_UNDERSTAND_TRANSLATE_FINAL_ONLY = "understand_translate_v2_final_only"

VISIBLE_REASONING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in [
        r"internal\s+(understanding|workflow)",
        r"understanding\s+workflow",
        r"\btranslation\s+planning\b",
        r"\bthought\s+and\s+emotion\b",
        r"\bmodernity\b.*\bpoeticity\b",
        r"^\s*#{1,6}\s+.*(content|expression|translation|workflow)",
        r"\bI'll\s+work\s+through\b",
        r"\bI\s+will\s+work\s+through\b",
        r"\bbefore\s+translat(?:ing|e)\b",
        r"\bfinal\s+english\s+poem\b",
    ]
]


def prompt_version() -> str:
    return os.getenv("POETRY_PROMPT_VERSION", PROMPT_VERSION_UNDERSTAND_TRANSLATE).strip() or PROMPT_VERSION_UNDERSTAND_TRANSLATE


def has_visible_reasoning_contamination(text: str) -> bool:
    """Detect visible workflow or analysis text that should not be scored as translation."""
    return any(pattern.search(text or "") for pattern in VISIBLE_REASONING_PATTERNS)


def build_translation_messages(source_zh: str, version: str | None = None) -> list[dict[str, str]]:
    version = version or prompt_version()
    if version == PROMPT_VERSION_DIRECT:
        system = (
            "You are a professional literary translator. Translate modern Chinese poetry into English. "
            "Preserve the poem's meaning, imagery, tone, and line breaks as much as possible. "
            "Return only the English poem, with no explanation, notes, title, or metadata."
        )
        user = "Translate the following modern Chinese poem into English:\n\n" + source_zh.strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if version == PROMPT_VERSION_UNDERSTAND_TRANSLATE_FINAL_ONLY:
        system = (
            "You are a professional literary translator of modern Chinese poetry into contemporary English free verse. "
            "Use an internal understand-then-translate process silently, then provide only the final translation.\n\n"
            "Silent internal process:\n"
            "- Understand the poem's content, imagery, rhetoric, rhythm, lineation, ambiguity, emotional movement, "
            "modern consciousness, and poetic compression.\n"
            "- Plan how to preserve meaning, tone, line breaks, stanza breaks, defamiliarization, and natural English.\n\n"
            "Final response rules:\n"
            "- Output only the final English poem.\n"
            "- Do not output reasoning, analysis, workflow steps, translation planning, explanations, notes, headings, "
            "Markdown, bullet points, title, author name, translator name, source name, metadata, or apologies.\n"
            "- Do not label the answer as a translation.\n"
            "- Preserve the poem's line breaks and stanza breaks as much as possible.\n"
            "- Use natural contemporary English; do not force rhyme or archaic diction unless the source strongly requires it."
        )
        user = "Translate the following modern Chinese poem into English:\n\n" + source_zh.strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if version != PROMPT_VERSION_UNDERSTAND_TRANSLATE:
        raise ValueError(f"Unknown prompt version: {version}")

    system = (
        "You are a professional literary translator of modern Chinese poetry into contemporary English free verse. "
        "Follow an internal understand-then-translate workflow before producing the final translation.\n\n"
        "Internal understanding workflow:\n"
        "1. Content: identify what the poem describes, including speaker, scene, events, and implied relations.\n"
        "2. Expression: analyze language texture, visual and sensory imagery, rhetorical techniques, rhythm, lineation, "
        "defamiliarization, ambiguity, and compression.\n"
        "3. Thought and emotion: infer the dominant ideas, affective movement, tonal shifts, and emotional restraint.\n"
        "4. Modernity: notice contemporary consciousness, modern experience, social or existential tension, and departures "
        "from classical poetic convention.\n"
        "5. Poeticity: identify the most poetically charged lines or images and use them as anchors for the English version.\n"
        "6. Translation planning: decide how to balance fidelity, imagery, line breaks, rhythm, ambiguity, and readable English.\n\n"
        "Translation requirements:\n"
        "- Translate the Chinese poem into English only after the internal understanding step.\n"
        "- Preserve meaning, imagery, tone, line breaks, stanza breaks, ambiguity, and poetic compression as much as possible.\n"
        "- Use natural contemporary English; do not force rhyme or archaic diction unless the source strongly requires it.\n"
        "- Do not add explanations, notes, titles, author names, translator names, markdown, or metadata.\n"
        "- Output only the final English poem."
    )
    user = "Translate the following modern Chinese poem into English:\n\n" + source_zh.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def assert_no_reference_leak(messages: list[dict[str, str]], record: dict[str, Any]) -> None:
    payload = "\n".join(message.get("content", "") for message in messages)
    reference = (record.get("reference_en") or "").strip()
    if reference and reference in payload:
        raise ValueError(f"Reference leak detected for record {record.get('record_id')}.")
    for field in REFERENCE_FORBIDDEN_FIELDS:
        if field in payload:
            raise ValueError(f"Prompt mentions forbidden reference field {field}.")


def diagnostic_scores(hypothesis: str, reference: str) -> dict[str, float | int]:
    return {
        "line_count_diff": line_count_difference(hypothesis, reference),
        "length_ratio": length_ratio(hypothesis, reference),
        "english_ratio": english_ratio(hypothesis),
        "empty_output": 1 if not (hypothesis or "").strip() else 0,
    }
