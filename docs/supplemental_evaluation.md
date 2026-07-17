# Supplemental Evaluation Files

The supplemental release contains generation parameters, prompts, scoring code, statistical-test code, and inferential results. It does not include poem records, translations, raw judge responses, candidate mappings, per-record scores, runtime summaries, or system-level summary tables.

## Parameters and prompts

The machine-readable configurations are:

- `configs/supplemental/generation.json`: model requests and the open-weight serving setup.
- `configs/supplemental/scoring.json`: automatic metric models and options.
- `configs/supplemental/judges.json`: judge sampling, score scale, and request settings.
- `configs/supplemental/statistical_tests.json`: resampling counts, seeds, correction method, and analysis entry points.

The exact translation, repair, direct-baseline, and judge prompts are in [`prompts.md`](prompts.md). The supplemental generation runs use `understand_translate_v1`; the judge runs use `agents_md_judge_v1_20260514`.

## Automatic scoring

`scripts/score_baseline_outputs.py` calls the scoring implementation in `src/poetry_reasoning/baselines/scoring.py`. The `all` profile computes COMET, BERTScore F1, SacreBLEU, chrF++, line-paired TER, line-count difference, and length ratio. Metrics use the final translation only.

Example using locally prepared inputs:

```powershell
.\.venv\Scripts\python scripts\score_baseline_outputs.py `
  --manifest-path results\local_manifest.jsonl `
  --outputs-path results\local_outputs.jsonl `
  --scores-path results\local_scores.jsonl `
  --summary-path results\local_summary.csv `
  --metric-profile all
```

The command requires a local manifest and model outputs. Its generated per-record scores and summaries remain local and are not part of this release.

## LLM-as-judge scoring

`scripts/run_llm_judge.py` implements fixed-seed sampling, anonymous candidate labels, strict response validation, retries, and score aggregation. Request parameters are recorded in `configs/supplemental/judges.json`.

```powershell
.\.venv\Scripts\python scripts\run_llm_judge.py --help
```

## Statistical tests

The following entry points correspond to the CSV files in `results/statistical_tests/`:

| Entry point | Analysis |
|---|---|
| `scripts/stats_alignment.py` | Automatic, judge, and human-score correlations |
| `scripts/stats_crossjudge.py` | Cross-judge agreement, rank stability, alignment, and self-preference interaction |
| `scripts/stats_human_paired.py` | Human paired tests, mixed models, inter-annotator agreement, and detectable effects |
| `scripts/stats_intensity.py` | Within-family reasoning-intensity comparisons |
| `scripts/stats_judge_paired.py` | Paired judge comparisons and detectable effects |
| `scripts/stats_open_qwen35.py` | Same-checkpoint open-weight mode comparison and exclusion sensitivity |
| `scripts/stats_rebuttal_auxiliary.py` | Annotator direction and source, length, and panel sensitivity checks |

Each script exposes its input and output arguments through `--help`. For example:

```powershell
.\.venv\Scripts\python scripts\stats_crossjudge.py --help
.\.venv\Scripts\python scripts\stats_intensity.py --help
.\.venv\Scripts\python scripts\stats_open_qwen35.py --help
```

The published CSVs contain aggregate inferential fields such as sample size, effect estimate, confidence interval, raw p-value, and Holm-adjusted p-value. They contain no record identifiers or text fields.

Run the release-boundary check with:

```powershell
.\.venv\Scripts\python scripts\audit_statistical_release.py results\statistical_tests
```
