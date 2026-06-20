from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_RESULTS_DIR, read_jsonl  # noqa: E402
from poetry_reasoning.evaluation.judge import JUDGE_DIMENSIONS  # noqa: E402


SYSTEM_LABELS = {
    "baidu_translate": "Baidu Translate",
    "deepseek_v4_flash_non_thinking": "DeepSeek V4 Flash (non-thinking)",
    "deepseek_v4_flash_thinking": "DeepSeek V4 Flash (thinking)",
    "qwen36_plus_non_thinking": "Qwen3.6 Plus (non-thinking)",
    "qwen36_plus_thinking": "Qwen3.6 Plus (thinking)",
    "claude_sonnet46_non_thinking": "Claude Sonnet 4.6 (non-thinking)",
    "claude_sonnet46_thinking": "Claude Sonnet 4.6 (thinking)",
}

CURATED_CASES = [
    {
        "case_id": "C1",
        "record_id": "8708f20283b61f97",
        "local_label": "P12",
        "source_hint": "21st Century Chinese Poetry",
        "systems": ["claude_sonnet46_non_thinking", "claude_sonnet46_thinking"],
        "focus": (
            "Culture-specific rendering, lunar-calendar phrasing, colloquial reassurance, "
            "and a central sunrise image."
        ),
        "analysis_note": (
            "Use this as the main Claude case. The non-thinking version is preferred by both "
            "judge and human layers, with a much larger human gap. The thinking version is "
            "readable but tends toward stiffer literal phrasing."
        ),
    },
    {
        "case_id": "C2",
        "record_id": "61dbce7cbc688130",
        "local_label": "P10",
        "source_hint": "Poetry International Chinese",
        "systems": ["qwen36_plus_non_thinking", "qwen36_plus_thinking"],
        "focus": "Lexical precision, idiomatic choices, and a key night-surf line.",
        "analysis_note": (
            "Use this as the main Qwen case. It is short enough for close reading and shows "
            "a clear non-thinking advantage under both judge and human scoring."
        ),
    },
    {
        "case_id": "C3",
        "record_id": "7c66535bb93c9034",
        "local_label": "P11",
        "source_hint": "21st Century Chinese Poetry",
        "systems": ["claude_sonnet46_non_thinking", "claude_sonnet46_thinking"],
        "focus": "Poetic restraint, lineation, and abstract image language.",
        "analysis_note": (
            "Use this as a secondary Claude case when the discussion needs an example focused "
            "on lineation and suggestiveness rather than cultural transfer."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected poetry translation case studies.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--judge-scores-path", type=Path)
    parser.add_argument(
        "--human-scores-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "viz_menu_20260612" / "D_case" / "work" / "per_poem_system_human.csv",
        help="Optional private per-poem human score CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "case_studies" / "case_studies_redacted.md",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include source poems, references, and selected model outputs. Use only for private local files.",
    )
    return parser.parse_args()


def latest_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        system_id = str(row.get("system_id") or "")
        record_id = str(row.get("record_id") or "")
        if system_id and record_id:
            latest[(system_id, record_id)] = row
    return latest


def load_human_scores(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    scores: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = str(row.get("record_id") or "")
            system_id = str(row.get("system_id") or "")
            if record_id and system_id:
                scores[(record_id, system_id)] = row
    return scores


def load_judge_scores(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        record_id = str(row.get("record_id") or "")
        system_id = str(row.get("system_id") or "")
        if record_id and system_id:
            scores[(record_id, system_id)] = row
    return scores


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("record_id")): row for row in read_jsonl(path) if row.get("record_id")}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def quote_block(text: str) -> str:
    if not text.strip():
        return "_Not available in local artifacts._"
    return "\n".join("> " + line if line else ">" for line in text.splitlines())


def write_case(
    case: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    outputs_by_key: dict[tuple[str, str], dict[str, Any]],
    judge_scores: dict[tuple[str, str], dict[str, Any]],
    human_scores: dict[tuple[str, str], dict[str, str]],
    include_text: bool,
) -> str:
    record_id = case["record_id"]
    manifest_row = manifest_by_id.get(record_id, {})
    source_label = manifest_row.get("source_name") or case["source_hint"]
    page_url = manifest_row.get("page_url") or ""
    source_line_count = len([line for line in (manifest_row.get("source_zh") or "").splitlines() if line.strip()])

    lines = [
        f"## {case['case_id']}: {case['local_label']} / `{record_id}`",
        "",
        md_table(
            ["Field", "Value"],
            [
                ["Source subset", str(source_label)],
                ["Record id", f"`{record_id}`"],
                ["Local label", str(case["local_label"])],
                ["Source URL", page_url or "NA"],
                ["Source line count", str(source_line_count or "NA")],
                ["Focus", str(case["focus"])],
            ],
        ),
        "",
        str(case["analysis_note"]),
        "",
        "### Scores",
        "",
    ]

    score_rows: list[list[str]] = []
    for system_id in case["systems"]:
        judge_row = judge_scores.get((record_id, system_id), {})
        human_row = human_scores.get((record_id, system_id), {})
        score_rows.append(
            [
                SYSTEM_LABELS.get(system_id, system_id),
                fmt(judge_row.get("average_score")),
                fmt(human_row.get("mean6")),
                fmt(human_row.get("MF")),
                fmt(human_row.get("IR")),
                fmt(human_row.get("EV")),
                fmt(human_row.get("LR")),
                fmt(human_row.get("MD")),
                fmt(human_row.get("PT")),
            ]
        )
    lines.append(md_table(["System", "Judge avg", "Human mean", "MF", "IR", "EV", "LR", "MD", "PT"], score_rows))

    lines.extend(["", "### Judge Dimension Detail", ""])
    for system_id in case["systems"]:
        judge_row = judge_scores.get((record_id, system_id), {})
        lines.extend([f"**{SYSTEM_LABELS.get(system_id, system_id)}**", ""])
        dimension_rows = [[dimension, fmt(judge_row.get(dimension))] for dimension in JUDGE_DIMENSIONS]
        lines.append(md_table(["Dimension", "Score"], dimension_rows))
        lines.append("")

    lines.extend(
        [
            "### Analysis Guide",
            "",
            "- Semantic fidelity: identify any meaning shifts or omissions.",
            "- Imagery and rhetoric: compare how images, metaphors, and rhetorical pressure survive in English.",
            "- Thought and emotion: note whether the emotional movement stays implicit or becomes explanatory.",
            "- Lineation and rhythm: inspect line breaks, pauses, stanza pressure, and cadence.",
            "- Modernity and defamiliarization: check whether strange modern poetic texture is preserved.",
            "- Cultural and idiomatic transfer: inspect local terms, idioms, names, and calendar or place references.",
            "- Poeticity and English naturalness: judge whether the output reads as English poetry rather than a gloss.",
            "",
        ]
    )

    if include_text:
        lines.extend(["### Source Poem", "", quote_block(manifest_row.get("source_zh") or ""), ""])
        lines.extend(["### Human Reference", "", quote_block(manifest_row.get("reference_en") or ""), ""])
        lines.extend(["### Candidate Translations", ""])
        for system_id in case["systems"]:
            output = outputs_by_key.get((system_id, record_id), {})
            lines.extend(
                [
                    f"**{SYSTEM_LABELS.get(system_id, system_id)}**",
                    "",
                    quote_block(output.get("output_text") or ""),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "### Text Availability",
                "",
                "Complete source poems, references, and candidate translations are redacted in this public-safe export.",
                "Rerun this script with `--include-text` to generate a private local packet.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.manifest_path:
        manifest_path = args.manifest_path
    elif (args.results_dir / "judge_manifest_20.jsonl").exists():
        manifest_path = args.results_dir / "judge_manifest_20.jsonl"
    else:
        manifest_path = args.results_dir / "baseline_manifest_full397.jsonl"
    outputs_path = args.outputs_path or args.results_dir / "full397_outputs.jsonl"
    judge_scores_path = args.judge_scores_path or args.results_dir / "judge_scores.jsonl"

    manifest_by_id = load_manifest(manifest_path)
    outputs_by_key = latest_by_key(read_jsonl(outputs_path)) if outputs_path.exists() else {}
    judge_scores = load_judge_scores(judge_scores_path) if judge_scores_path.exists() else {}
    human_scores = load_human_scores(args.human_scores_path)

    header = [
        "# Selected Poetry Translation Case Studies",
        "",
        "This file was generated by `scripts/export_case_studies.py`.",
        "",
    ]
    if args.include_text:
        header.extend(
            [
                "Warning: this private export may contain copyrighted poem text, published reference translations, and raw model outputs. Do not commit it without redistribution rights.",
                "",
            ]
        )
    else:
        header.extend(
            [
                "This redacted export does not include complete source poems, references, or candidate translations.",
                "",
            ]
        )

    body = [write_case(case, manifest_by_id, outputs_by_key, judge_scores, human_scores, args.include_text) for case in CURATED_CASES]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(header + body), encoding="utf-8")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
