# LLM Poetry Translation

Public code supplement for the paper **"Less Thinking, More Poetry: A Controlled Comparison of Reasoning and Non-Reasoning LLMs for Modern Chinese Poetry Translation"**.

本仓库是论文 **"Less Thinking, More Poetry: A Controlled Comparison of Reasoning and Non-Reasoning LLMs for Modern Chinese Poetry Translation"** 的代码，比较推理模式与非推理模式下大语言模型的翻译表现。

## Scope / 仓库范围

This repository provides the runnable code needed to reproduce the study protocol: translation baselines, automatic metric scoring, fixed-seed LLM-as-judge evaluation, and small utility scripts for merging and checking experiment outputs.

本仓库提供复现实验流程所需的可运行代码，包括翻译基线、自动指标评分、固定随机种子的 LLM-as-judge 评测，以及用于合并和检查实验输出的辅助脚本。

Due to copyright and privacy restrictions, this repository does not include poem texts, reference translations, curated datasets, raw model outputs, reasoning traces, judge scores, or human evaluation scores. Because the source poems and published translations are protected by copyright and governed by the terms of their source websites, this repository only provides publicly available source links in data/source_urls.json. Readers who wish to reproduce the experiments should compile their own dataset in compliance with those terms.No automated data-collection code is included in this public repository.

受到版权和隐私限制，本仓库不包含诗歌正文、参考译文、已整理数据集、原始模型输出、推理、评审分数、人工评测分数。由于诗歌原文与已发表译文受版权及来源网站条款约束，本仓库仅在 data/source_urls.json 中列出公开来源链接。读者如需复现实验，应在合规前提下自行整理本地数据集。本公开仓库不包含自动化数据采集代码。


```text
data/source_urls.json      Public source URLs only
scripts/                  Experiment entry points and utilities
src/poetry_reasoning/     Core experiment package
.env.example              Environment variable template
```

## Reproducibility / 复现边界

The code reproduces the experimental protocol and reporting pipeline. Exact API outputs may differ over time because external model providers, decoding settings, and service-side implementations can change.

本仓库能够复现完整的实验流程。由于外部模型服务、解码设置和服务端实现可能随时间变化，API 生成的逐条译文不保证与论文实验完全一致。

The paper's numerical tables are based on the authors' controlled private experiment logs. This repository is intended to let readers rerun the protocol with locally prepared data and their own API credentials.

论文中的数值表格来自作者在受控条件下的实验记录。本仓库的用途，是帮助读者在自行准备数据并配置 API 凭证后复现实验流程。

During translation, the systems receive only the Chinese source poem. Reference translations are used only for offline scoring or judge evaluation and are never included in translation prompts.

翻译阶段中，模型只接收中文原诗。参考译文仅用于离线评分或评审环节，不会被写入翻译提示词。

## Environment / 环境配置

Use Python 3.10 or newer. On Windows, invoke the virtual-environment interpreter directly instead of `py -3`, because `py -3` may resolve to an older Python installation.

请使用 Python 3.10 或更高版本。在 Windows 上，请直接调用虚拟环境中的解释器，不建议使用可能指向旧版 Python 的 `py -3`。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill only the credentials required for the systems you plan to run. The `.env` file is local and must not be committed.

将 `.env.example` 复制为 `.env`，只填写实际运行所需系统的凭证。`.env` 属于本地私有文件，不应提交到仓库。

## Data Preparation / 数据准备

The default local dataset path is:

默认本地数据集路径为：

```text
data/processed/bilingual_modern_chinese_poetry.jsonl
```

Each JSONL row should contain at least the following fields:

每条 JSONL 记录至少应包含以下字段：

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

Optional fields such as `page_url`, `title_zh`, `poet_zh`, `translator`, and `quality_flags` are preserved in generated manifests when present.

如果数据中包含 `page_url`、`title_zh`、`poet_zh`、`translator`、`quality_flags` 等可选字段，生成 manifest 时会一并保留。

## Run Locally / 本地运行

Run a small trial first. Providers without configured credentials will be recorded as `not_available`, which allows the pipeline to be checked before any keys are used.

建议先进行一次小规模试跑。未配置凭证的服务会被记录为 `not_available`，这样可以在使用密钥前先检查流程是否连通。

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

运行完整翻译流程：

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

仅对最终译文进行评分：

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

运行固定随机种子的 LLM-as-judge 评测：

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

Generated outputs are written under `results/`, which is intentionally ignored by git.

所有运行结果都会写入 `results/` 目录；该目录已被 git 忽略，不属于公开仓库内容。

## Run On A Server / 服务器运行

The server workflow uses the same Python entry points as the local workflow. For long API runs, split the task into deterministic shards and run each shard in the background.

服务器端与本地端使用同一套 Python 入口。对于耗时较长的 API 任务，可以将任务切分为确定性的多个分片，并在后台分别运行。

Example for shard 0 of 4:

以下示例运行 4 个分片中的第 0 个：

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

所有分片完成后，合并输出文件：

```bash
PYTHONPATH=src .venv/bin/python scripts/merge_jsonl_outputs.py \
  --base-path results/full397_outputs_shard0.jsonl \
  --extra-path results/full397_outputs_shard1.jsonl \
  --extra-path results/full397_outputs_shard2.jsonl \
  --extra-path results/full397_outputs_shard3.jsonl \
  --output-path results/full397_outputs.jsonl
```

Then run the same scoring and LLM-as-judge commands shown in the local workflow.

随后继续运行本地流程中相同的自动评分和 LLM-as-judge 命令。
