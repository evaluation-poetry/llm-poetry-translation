from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_DIR,
    append_jsonl,
    load_dotenv,
    prepare_manifest,
    read_jsonl,
)
from poetry_reasoning.baselines.providers import (  # noqa: E402
    claude_provider_ids,
    default_provider_ids,
    full397_provider_ids,
    model_family_provider_ids,
    provider_by_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chinese-to-English poetry baseline translation APIs.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N manifest records.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split manifest into N deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only the 0-based shard index.")
    parser.add_argument(
        "--existing-outputs-path",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL outputs to consult when skipping already completed records.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["default"],
        help="System ids, 'default', 'full397', 'claude_full397', 'model_family_full397', or 'all'.",
    )
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        help="Include zh-en records with quality flags. Use this for the full 397-poem run.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun outputs that already exist.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry existing non-ok outputs while keeping ok rows.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    manifest_path = args.manifest_path or args.results_dir / "baseline_manifest_100.jsonl"
    outputs_path = args.outputs_path or args.results_dir / "baseline_outputs.jsonl"
    if args.force and outputs_path.exists():
        outputs_path.unlink()
    manifest = prepare_manifest(
        args.dataset_path,
        manifest_path,
        sample_size=args.sample_size,
        overwrite=args.force_manifest,
        include_flagged=args.include_flagged,
    )
    if args.limit:
        manifest = manifest[: args.limit]
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")
    if args.shard_count > 1:
        manifest = [
            record for index, record in enumerate(manifest)
            if index % args.shard_count == args.shard_index
        ]
        print(
            f"[shard] index={args.shard_index} count={args.shard_count} records={len(manifest)}",
            flush=True,
        )

    providers = provider_by_id(include_optional=True)
    if args.systems == ["all"]:
        selected_ids = list(providers)
    elif args.systems == ["full397"]:
        selected_ids = full397_provider_ids()
    elif args.systems == ["claude_full397"]:
        selected_ids = claude_provider_ids()
    elif args.systems == ["model_family_full397"]:
        selected_ids = model_family_provider_ids()
    elif args.systems == ["default"]:
        selected_ids = default_provider_ids()
    else:
        selected_ids = args.systems
    unknown = [system_id for system_id in selected_ids if system_id not in providers]
    if unknown:
        raise SystemExit(f"Unknown system ids: {', '.join(unknown)}")

    existing = set()
    if not args.force:
        existing_paths = [*args.existing_outputs_path, outputs_path]
        for existing_path in existing_paths:
            for row in read_jsonl(existing_path):
                if args.retry_failed and row.get("status") != "ok":
                    continue
                existing.add((row.get("system_id"), row.get("record_id")))

    for system_id in selected_ids:
        provider = providers[system_id]
        is_available, reason = provider.available()
        if not is_available:
            print(f"[{system_id}] not available: {reason}")
            for record in manifest:
                key = (system_id, record["record_id"])
                if key in existing:
                    continue
                append_jsonl(outputs_path, provider.not_available_result(record, reason).to_dict())
            continue

        print(f"[{system_id}] running {len(manifest)} records")
        progress_every = int(os.getenv("PROGRESS_EVERY_N", "25"))
        status_counts: Counter[str] = Counter()
        for index, record in enumerate(manifest, start=1):
            key = (system_id, record["record_id"])
            if key in existing:
                continue
            start = time.perf_counter()
            try:
                result = provider.translate(record)
            except Exception as exc:
                result = provider.error_result(record, exc, latency=time.perf_counter() - start)
            append_jsonl(outputs_path, result.to_dict())
            status_counts[result.status] += 1
            print(f"[{system_id}] {index}/{len(manifest)} {record['record_id']} {result.status}")
            if progress_every > 0 and (index % progress_every == 0 or index == len(manifest)):
                summary = " ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
                print(f"[{system_id}] progress {index}/{len(manifest)} {summary}", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Manifest: {manifest_path}")
    print(f"Outputs: {outputs_path}")


if __name__ == "__main__":
    main()
