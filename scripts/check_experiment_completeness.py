from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_RESULTS_DIR, read_jsonl  # noqa: E402
from poetry_reasoning.evaluation.judge import DEFAULT_CANDIDATE_SYSTEM_IDS  # noqa: E402


CLAUDE_SYSTEM_IDS = ["claude_sonnet46_non_thinking", "claude_sonnet46_thinking"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check completeness of Claude and LLM-as-judge experiment artifacts.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--judge-scores-path", type=Path)
    parser.add_argument("--target-translation-n", type=int, default=397)
    parser.add_argument("--target-judge-n", type=int, default=20)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_path = args.outputs_path or args.results_dir / "full397_outputs.jsonl"
    judge_scores_path = args.judge_scores_path or args.results_dir / "judge_scores.jsonl"

    latest_outputs: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(outputs_path):
        system_id = str(row.get("system_id") or "")
        record_id = str(row.get("record_id") or "")
        if system_id in CLAUDE_SYSTEM_IDS and record_id:
            latest_outputs[(system_id, record_id)] = row

    output_counts: Counter[str] = Counter()
    output_failures: Counter[str] = Counter()
    for (system_id, _record_id), row in latest_outputs.items():
        if row.get("status") == "ok" and (row.get("output_text") or "").strip():
            output_counts[system_id] += 1
        else:
            output_failures[system_id] += 1

    judge_counts: Counter[str] = Counter()
    for row in read_jsonl(judge_scores_path):
        system_id = str(row.get("system_id") or "")
        if system_id in DEFAULT_CANDIDATE_SYSTEM_IDS:
            judge_counts[system_id] += 1

    complete = True
    print("[completeness] Claude translations")
    for system_id in CLAUDE_SYSTEM_IDS:
        ok = output_counts[system_id]
        failed = output_failures[system_id]
        print(f"  {system_id}: ok={ok} failed={failed} target={args.target_translation_n}")
        complete = complete and ok >= args.target_translation_n

    print("[completeness] Judge scores")
    for system_id in DEFAULT_CANDIDATE_SYSTEM_IDS:
        count = judge_counts[system_id]
        print(f"  {system_id}: scored={count} target={args.target_judge_n}")
        complete = complete and count >= args.target_judge_n

    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
