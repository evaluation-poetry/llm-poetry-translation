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
    validate_shard_output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 20-poem LLM-as-judge evaluation.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path, action="append")
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--reuse-mapping-path", type=Path)
    parser.add_argument("--mapping-verification-path", type=Path)
    parser.add_argument("--scores-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--sample-per-source", type=int, default=5)
    parser.add_argument("--systems", nargs="+", default=DEFAULT_CANDIDATE_SYSTEM_IDS)
    parser.add_argument("--judge-provider", choices=["deepseek", "gpt"], default="deepseek")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--gate-translation",
        action="append",
        default=[],
        metavar="SYSTEM_ID=PATH",
        help="Required three-system translation gate artifact (repeat exactly three times in GPT mode).",
    )
    parser.add_argument("--gate-scores-path", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N sampled poems.")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun judge calls from scratch.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry existing non-ok judge rows.")
    aggregation = parser.add_mutually_exclusive_group()
    aggregation.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing raw judge outputs.")
    aggregation.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Run a shard without writing scores or summary; aggregate merged artifacts separately.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def parse_gate_translation_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values:
        system_id, separator, path_text = value.partition("=")
        if not separator or not system_id.strip() or not path_text.strip():
            raise ValueError("--gate-translation must use SYSTEM_ID=PATH")
        specs.append((system_id.strip(), Path(path_text.strip())))
    return specs


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.shard_count > 1 and (args.raw_path is None or args.mapping_path is None):
        raise ValueError("shard runs require explicit --raw-path and --mapping-path")
    manifest_path = args.manifest_path or args.results_dir / "judge_manifest_20.jsonl"
    outputs_paths = args.outputs_path or [args.results_dir / "full397_outputs.jsonl"]
    if args.judge_provider == "gpt":
        raw_path = args.raw_path or args.results_dir / "judge_gpt_raw.jsonl"
        mapping_path = args.mapping_path or args.results_dir / "judge_gpt_candidate_mapping_private.jsonl"
        scores_path = args.scores_path or args.results_dir / "judge_gpt_scores.jsonl"
        summary_path = args.summary_path or args.results_dir / "judge_gpt_summary.csv"
    else:
        raw_path = args.raw_path or args.results_dir / "judge_raw.jsonl"
        mapping_path = args.mapping_path or args.results_dir / "judge_candidate_mapping_private.jsonl"
        scores_path = args.scores_path or args.results_dir / "judge_scores.jsonl"
        summary_path = args.summary_path or args.results_dir / "judge_summary.csv"
    mapping_verification_path = args.mapping_verification_path
    if args.reuse_mapping_path is not None and mapping_verification_path is None:
        mapping_verification_path = args.results_dir / "judge_gpt_mapping_verification.json"
    gate_translation_specs = parse_gate_translation_specs(args.gate_translation)

    validate_shard_output_paths(
        args.shard_count,
        args.shard_index,
        raw_path,
        mapping_path,
        no_aggregate=args.no_aggregate,
    )

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
            outputs_path=outputs_paths,
            raw_path=raw_path,
            mapping_path=mapping_path,
            system_ids=args.systems,
            seed=args.seed,
            limit=args.limit,
            force=args.force,
            retry_failed=args.retry_failed,
            sleep_seconds=args.sleep_seconds,
            judge_provider=args.judge_provider,
            reuse_mapping_path=args.reuse_mapping_path,
            mapping_verification_path=mapping_verification_path,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            gate_dataset_path=args.dataset_path,
            gate_translation_specs=gate_translation_specs,
            gate_scores_path=args.gate_scores_path,
            no_aggregate=args.no_aggregate,
        )

    if args.no_aggregate:
        print(f"Judge manifest: {manifest_path} ({len(manifest)} records)")
        print(f"Judge raw shard: {raw_path}")
        print(f"Judge mapping shard: {mapping_path}")
        return

    scores, summaries = aggregate_judge_results(
        raw_path,
        scores_path,
        summary_path,
        private_mapping_path=mapping_path if args.judge_provider == "gpt" else None,
    )
    print(f"Judge manifest: {manifest_path} ({len(manifest)} records)")
    print(f"Judge raw: {raw_path}")
    print(f"Judge mapping: {mapping_path}")
    if mapping_verification_path is not None:
        print(f"Judge mapping verification: {mapping_verification_path}")
    print(f"Judge scores: {scores_path} ({len(scores)} rows)")
    print(f"Judge summary: {summary_path} ({len(summaries)} rows)")


if __name__ == "__main__":
    main()
