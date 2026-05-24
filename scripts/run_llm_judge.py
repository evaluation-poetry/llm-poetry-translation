from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_DATASET_PATH, DEFAULT_RESULTS_DIR, load_dotenv  # noqa: E402
from poetry_reasoning.evaluation.judge import (  # noqa: E402
    DEFAULT_CANDIDATE_SYSTEM_IDS,
    aggregate_judge_results,
    build_judge_manifest,
    run_judge_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 20-poem LLM-as-judge evaluation.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--scores-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--sample-per-source", type=int, default=5)
    parser.add_argument("--systems", nargs="+", default=DEFAULT_CANDIDATE_SYSTEM_IDS)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N sampled poems.")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun judge calls from scratch.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry existing non-ok judge rows.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing raw judge outputs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    manifest_path = args.manifest_path or args.results_dir / "judge_manifest_20.jsonl"
    outputs_path = args.outputs_path or args.results_dir / "full397_outputs.jsonl"
    raw_path = args.raw_path or args.results_dir / "judge_raw.jsonl"
    mapping_path = args.mapping_path or args.results_dir / "judge_candidate_mapping_private.jsonl"
    scores_path = args.scores_path or args.results_dir / "judge_scores.jsonl"
    summary_path = args.summary_path or args.results_dir / "judge_summary.csv"

    manifest = build_judge_manifest(
        args.dataset_path,
        manifest_path,
        seed=args.seed,
        sample_per_source=args.sample_per_source,
        overwrite=args.force_manifest,
    )
    if not args.aggregate_only:
        run_judge_evaluation(
            manifest,
            outputs_path=outputs_path,
            raw_path=raw_path,
            mapping_path=mapping_path,
            system_ids=args.systems,
            seed=args.seed,
            limit=args.limit,
            force=args.force,
            retry_failed=args.retry_failed,
            sleep_seconds=args.sleep_seconds,
        )

    scores, summaries = aggregate_judge_results(raw_path, scores_path, summary_path)
    print(f"Judge manifest: {manifest_path} ({len(manifest)} records)")
    print(f"Judge raw: {raw_path}")
    print(f"Judge mapping: {mapping_path}")
    print(f"Judge scores: {scores_path} ({len(scores)} rows)")
    print(f"Judge summary: {summary_path} ({len(summaries)} rows)")


if __name__ == "__main__":
    main()
