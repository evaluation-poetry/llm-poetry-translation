from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_RESULTS_DIR, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge append-only translation output JSONL files by latest key.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--extra-path", type=Path, action="append", default=[])
    parser.add_argument("--extra-glob", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_path or args.results_dir / "full397_outputs.jsonl"
    output_path = args.output_path or base_path
    paths = [base_path, *args.extra_path]
    for pattern in args.extra_glob:
        paths.extend(sorted(args.results_dir.glob(pattern)))

    latest: dict[tuple[str, str], dict] = {}
    passthrough: list[dict] = []
    for path in paths:
        for row in read_jsonl(path):
            system_id = str(row.get("system_id") or "")
            record_id = str(row.get("record_id") or "")
            if system_id and record_id:
                latest[(system_id, record_id)] = row
            else:
                passthrough.append(row)

    merged = passthrough + [
        latest[key] for key in sorted(latest, key=lambda item: (item[0], item[1]))
    ]
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    write_jsonl(tmp_path, merged)
    tmp_path.replace(output_path)
    print(f"Merged {len(paths)} files into {output_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
