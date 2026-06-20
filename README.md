# LLM Poetry Translation

Public code supplement for the paper **"Less Thinking, More Poetry: A Controlled Comparison of Reasoning and Non-Reasoning LLMs for Modern Chinese Poetry Translation"**.

Repository: [evaluation-poetry/llm-poetry-translation](https://github.com/evaluation-poetry/llm-poetry-translation)

## Scope

This repository provides the runnable experiment code and public documentation needed to reproduce the study protocol:

- Chinese-to-English modern poetry translation baselines.
- Paired non-thinking and thinking-enabled inference settings.
- Automatic metric scoring on the full local dataset.
- Fixed-seed LLM-as-judge evaluation on a source-balanced 20-poem subset.
- Public supplement documentation for prompts, dimensions, evaluation design, and case-study construction.

Due to copyright, provider, and privacy constraints, the public repository does not redistribute poem texts, reference translations, compiled datasets, raw model outputs, private candidate mappings, raw human-evaluation workbooks, or full private result logs. The file `data/source_urls.json` records only public source websites and seed pages used for provenance. Readers who wish to reproduce the experiments must prepare their own local dataset in compliance with the relevant source-site terms.

## Public Supplement

Because the paper has limited space, detailed reproducibility material is kept in this repository:

```text
docs/dataset_schema.md          Expected local JSONL schema
docs/prompts.md                 Full translation and LLM-as-judge prompts
docs/judge_dimensions.md        Detailed 11-dimension judge rubric
docs/evaluation_protocol.md     Automatic, judge, and human-evaluation protocol
docs/case_studies.md            Public-safe case-study guide and selected case index
scripts/export_case_studies.py  Local exporter for redacted or private full case packets
```

## Repository Layout

```text
data/source_urls.json      Public source URLs only
data/raw/                  Local-only raw workspace, ignored by git
data/interim/              Local-only parsed workspace, ignored by git
data/processed/            Local-only canonical dataset workspace, ignored by git
scripts/                  Experiment entry points and utilities
src/poetry_reasoning/     Core experiment package
docs/                     Public protocol and supplement documentation
.env.example              Environment variable template
```

## Reproducibility Boundary

The code reproduces the experimental protocol and reporting pipeline. Exact API outputs may differ over time because external providers, model versions, decoding settings, and service-side implementations can change.

The numerical tables in the paper are based on controlled private experiment logs. This repository lets readers rerun the same protocol with locally prepared data and their own API credentials.

During translation, systems receive only the Chinese source poem. English references, English titles, translator names, source subsets, model identities, and system identities are never included in translation prompts. Reference translations are reserved for offline scoring and judge evaluation.

All automatic metrics and judge scores must be computed from final translation content only, never from hidden reasoning traces or provider-side thinking summaries.

## Environment

Use Python 3.10 or newer. On Windows, call the virtual-environment interpreter directly rather than `py -3`, which may point to an older Python installation.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill only the credentials required for the systems you plan to run. The `.env` file is local and must not be committed.

## Data Preparation

The default local dataset path is:

```text
data/processed/bilingual_modern_chinese_poetry.jsonl
```

Each JSONL row should contain at least:

```json
{
  "record_id": "unique_id",
  "source_id": "modern_chinese_poetry",
  "source_name": "21st Century Chinese Poetry",
  "language_pair": "zh-en",
  "original_zh": "Chinese source poem text",
  "translation_en": "Reference English translation"
}
```

Optional fields such as `page_url`, `title_zh`, `title_en`, `poet_zh`, `poet_en`, `translator`, `original_lines`, `translation_lines`, and `quality_flags` are preserved in generated manifests when present. See [docs/dataset_schema.md](docs/dataset_schema.md) for details.

## Run Locally

Run a small trial first. Providers without configured credentials are recorded as `not_available`, so you can check the pipeline before adding any keys.

```powershell
.\.venv\Scripts\python scripts\run_baseline_translations.py `
  --dataset-path data\processed\bilingual_modern_chinese_poetry.jsonl `
  --manifest-path results\baseline_manifest_full397.jsonl `
  --outputs-path results\full397_outputs_smoke.jsonl `
  --sample-size 0 `
  --include-flagged `
  --force-manifest `
  --force `
  --limit 3 `
  --systems full397
```

Run the full translation protocol:

```powershell
.\.venv\Scripts\python scripts\run_baseline_translations.py `
  --dataset-path data\processed\bilingual_modern_chinese_poetry.jsonl `
  --manifest-path results\baseline_manifest_full397.jsonl `
  --outputs-path results\full397_outputs.jsonl `
  --sample-size 0 `
  --include-flagged `
  --force-manifest `
  --retry-failed `
  --systems full397
```

Score final translations only:

```powershell
.\.venv\Scripts\python scripts\score_baseline_outputs.py `
  --manifest-path results\baseline_manifest_full397.jsonl `
  --outputs-path results\full397_outputs.jsonl `
  --scores-path results\full397_scores.jsonl `
  --summary-path results\full397_summary.csv `
  --summary-by-source-path results\full397_summary_by_source.csv `
  --metric-profile all
```

Run the fixed-seed LLM-as-judge evaluation:

```powershell
.\.venv\Scripts\python scripts\run_llm_judge.py `
  --dataset-path data\processed\bilingual_modern_chinese_poetry.jsonl `
  --manifest-path results\judge_manifest_20.jsonl `
  --outputs-path results\full397_outputs.jsonl `
  --raw-path results\judge_raw.jsonl `
  --mapping-path results\judge_candidate_mapping_private.jsonl `
  --scores-path results\judge_scores.jsonl `
  --summary-path results\judge_summary.csv `
  --seed 20260513
```

Generated outputs are written under `results/`, which is ignored by git.

## Case-Study Export

The public documentation describes selected qualitative cases without redistributing complete copyrighted poem texts or full model outputs. To generate local case-study packets from your private artifacts:

```powershell
.\.venv\Scripts\python scripts\export_case_studies.py `
  --output-path outputs\case_studies\case_studies_redacted.md
```

To include source poems, references, and candidate translations in a private local file, use:

```powershell
.\.venv\Scripts\python scripts\export_case_studies.py `
  --include-text `
  --output-path outputs\case_studies\case_studies_private_full.md
```

The `outputs/` directory is ignored by git. Do not commit full case packets unless you have the required redistribution rights.

## Run On A Server

The server workflow uses the same Python entry points as the local workflow. For long API runs, split the task into deterministic shards and run each shard in the background.

Example for shard 0 of 4:

```bash
mkdir -p logs results
PYTHONPATH=src nohup .venv/bin/python scripts/run_baseline_translations.py \
  --dataset-path data/processed/bilingual_modern_chinese_poetry.jsonl \
  --manifest-path results/baseline_manifest_full397.jsonl \
  --outputs-path results/full397_outputs_shard0.jsonl \
  --sample-size 0 \
  --include-flagged \
  --force-manifest \
  --retry-failed \
  --systems full397 \
  --shard-count 4 \
  --shard-index 0 \
  > logs/translate_shard0.log 2>&1 &
```

After all shards finish, merge them:

```bash
PYTHONPATH=src .venv/bin/python scripts/merge_jsonl_outputs.py \
  --base-path results/full397_outputs_shard0.jsonl \
  --extra-path results/full397_outputs_shard1.jsonl \
  --extra-path results/full397_outputs_shard2.jsonl \
  --extra-path results/full397_outputs_shard3.jsonl \
  --output-path results/full397_outputs.jsonl
```

Then run the same scoring and LLM-as-judge commands shown in the local workflow.

## Citation Link

Use this public repository URL in the paper:

```tex
The source code and experimental scripts of this work are publicly available\footnote{\url{https://github.com/evaluation-poetry/llm-poetry-translation}}.
```
