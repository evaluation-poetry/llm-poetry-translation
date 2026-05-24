from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poetry_reasoning.baselines.common import DEFAULT_RESULTS_DIR, nonempty_lines, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark COMET and line-level TER on a small output subset.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--outputs-path", type=Path)
    parser.add_argument("--poems", type=int, default=3)
    parser.add_argument("--metrics", nargs="+", default=["ter", "comet"], choices=["ter", "comet"])
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def paired_line_segments(hypothesis: str, reference: str) -> list[tuple[str, str]]:
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


def load_benchmark_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = args.manifest_path or args.results_dir / "baseline_manifest_100.jsonl"
    outputs_path = args.outputs_path or args.results_dir / "baseline_outputs.jsonl"
    manifest = read_jsonl(manifest_path)[: args.poems]
    manifest_by_id = {row["record_id"]: row for row in manifest}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(outputs_path):
        record_id = row.get("record_id")
        if record_id in manifest_by_id:
            latest[(str(row.get("system_id")), str(record_id))] = row

    rows: list[dict[str, Any]] = []
    for (system_id, record_id), output in sorted(latest.items()):
        if output.get("status") != "ok":
            continue
        record = manifest_by_id[record_id]
        rows.append(
            {
                "system_id": system_id,
                "system_name": output.get("system_name", system_id),
                "record_id": record_id,
                "source_zh": record.get("source_zh", ""),
                "hypothesis": (output.get("output_text") or "").strip(),
                "reference_en": record.get("reference_en", ""),
            }
        )
    return manifest, rows


def benchmark_ter(rows: list[dict[str, Any]], poems: int) -> None:
    from sacrebleu.metrics import TER

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[row["system_id"]].append(row)

    metric = TER()
    print(f"Line-level TER benchmark: {poems} poems x {len(by_system)} systems", flush=True)
    start_all = time.perf_counter()
    for system_id, system_rows in sorted(by_system.items()):
        hypotheses: list[str] = []
        references: list[str] = []
        for row in system_rows:
            for hypothesis, reference in paired_line_segments(row["hypothesis"], row["reference_en"]):
                hypotheses.append(hypothesis)
                references.append(reference)
        start = time.perf_counter()
        score = metric.corpus_score(hypotheses, [references]).score if hypotheses else None
        elapsed = time.perf_counter() - start
        score_text = "--" if score is None else f"{score:.3f}"
        print(
            f"  {system_id}: poems={len(system_rows)}, line_segments={len(hypotheses)}, "
            f"TER={score_text}, time={elapsed:.2f}s",
            flush=True,
        )
    elapsed_all = time.perf_counter() - start_all
    estimate = elapsed_all * (100 / max(poems, 1))
    print(f"Line-level TER total: {elapsed_all:.2f}s", flush=True)
    print(f"Line-level TER linear estimate for 100 poems: {estimate:.1f}s ({estimate / 60:.1f}min)", flush=True)


def benchmark_comet(rows: list[dict[str, Any]], poems: int, batch_size: int) -> None:
    from comet import download_model, load_from_checkpoint

    model_name = "Unbabel/wmt22-comet-da"
    print(f"COMET benchmark: {poems} poems, {len(rows)} candidate translations", flush=True)
    start_download = time.perf_counter()
    model_path = download_model(model_name)
    download_elapsed = time.perf_counter() - start_download
    print(f"COMET download/cache lookup: {download_elapsed:.2f}s", flush=True)

    start_load = time.perf_counter()
    model = load_from_checkpoint(model_path)
    load_elapsed = time.perf_counter() - start_load
    print(f"COMET model load: {load_elapsed:.2f}s", flush=True)

    data = [{"src": row["source_zh"], "mt": row["hypothesis"], "ref": row["reference_en"]} for row in rows]
    start_predict = time.perf_counter()
    prediction = model.predict(data, batch_size=batch_size, gpus=0)
    predict_elapsed = time.perf_counter() - start_predict
    scores = getattr(prediction, "scores", None) or prediction.get("scores", [])
    total_elapsed = download_elapsed + load_elapsed + predict_elapsed
    estimate_predict = predict_elapsed * (500 / max(len(rows), 1))
    print(f"COMET predict: {predict_elapsed:.2f}s for {len(rows)} rows", flush=True)
    print(f"COMET total benchmark: {total_elapsed:.2f}s", flush=True)
    print(f"COMET linear predict-only estimate for 500 rows: {estimate_predict:.1f}s ({estimate_predict / 60:.1f}min)", flush=True)
    if scores:
        print(f"COMET score preview: mean={sum(scores) / len(scores):.4f}, n={len(scores)}", flush=True)


def main() -> None:
    args = parse_args()
    manifest, rows = load_benchmark_rows(args)
    print(f"Loaded benchmark subset: poems={len(manifest)}, rows={len(rows)}", flush=True)
    if "ter" in args.metrics:
        benchmark_ter(rows, poems=len(manifest))
    if "comet" in args.metrics:
        benchmark_comet(rows, poems=len(manifest), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
