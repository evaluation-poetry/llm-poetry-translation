from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_RESULTS_DIR, load_dotenv  # noqa: E402
from poetry_reasoning.baselines.scoring import score_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score baseline poetry translation outputs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--scores-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--summary-by-source-path", type=Path)
    parser.add_argument(
        "--metric-profile",
        choices=["light", "bertscore", "comet", "all"],
        default="all",
        help="light computes SacreBLEU/chrF++/TER and diagnostics only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    manifest_path = args.manifest_path or args.results_dir / "baseline_manifest_100.jsonl"
    outputs_path = args.outputs_path or args.results_dir / "baseline_outputs.jsonl"
    scores_path = args.scores_path or args.results_dir / "baseline_scores.jsonl"
    summary_path = args.summary_path or args.results_dir / "baseline_summary.csv"
    summary_by_source_path = args.summary_by_source_path
    scores, summaries = score_outputs(
        manifest_path,
        outputs_path,
        scores_path,
        summary_path,
        metric_profile=args.metric_profile,
        summary_by_source_path=summary_by_source_path,
    )
    print(f"Scores: {scores_path} ({len(scores)} rows)")
    print(f"Summary: {summary_path} ({len(summaries)} systems)")
    if summary_by_source_path:
        print(f"Source summary: {summary_by_source_path}")


if __name__ == "__main__":
    main()
