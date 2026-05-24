# LLM Poetry Translation

Public code supplement for a paper on **reasoning vs. non-reasoning LLM settings for Chinese-to-English modern Chinese poetry translation**.

本仓库是一篇论文的公开代码补充材料，研究主题是：**推理模式与非推理模式的大语言模型在中国现代诗中译英任务中的表现差异**。

## What Is Included / 仓库内容

This repository includes runnable experiment code only. It does not include copyrighted poem text, reference translations, collected datasets, raw model outputs, hidden reasoning, judge responses, human evaluation workbooks, private mappings, API keys, or result tables.

本仓库只包含可运行的实验代码，不包含受版权保护的诗歌正文、参考译文、已整理数据集、原始模型输出、隐藏推理、评审原始响应、人工评测工作簿、私有映射、API 密钥或结果表格。

The public source URLs are listed in `data/source_urls.json`. Readers must prepare any local dataset themselves under the terms of the source websites.

公开来源链接见 `data/source_urls.json`。读者如需复现实验，应在遵守来源网站条款的前提下自行准备本地数据集。

No crawler or data-collection code is included in this public repository.

本公开仓库不包含爬取或数据采集代码。

```text
data/source_urls.json      Public source URLs only
scripts/                  Translation, scoring, judge, and utility entry points
src/poetry_reasoning/     Experiment package
.env.example              Environment variable template
```

## Environment / 环境配置

Use Python 3.10 or newer. On Windows, use the project virtual environment explicitly instead of `py -3`, because `py -3` may resolve to an older interpreter.

请使用 Python 3.10 或更高版本。在 Windows 上请明确使用项目虚拟环境，不建议使用可能指向旧版本解释器的 `py -3`。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill only the credentials needed for the systems you run. Never commit `.env`.

复制 `.env.example` 为 `.env`，只填写实际需要运行的系统凭证。不要提交 `.env`。

## Local Data Format / 本地数据格式

The default local dataset path is:

默认本地数据集路径为：

```text
data/processed/bilingual_modern_chinese_poetry.jsonl
```

Each JSONL row should contain at least:

每条 JSONL 至少应包含：

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

如存在 `page_url`、`title_zh`、`poet_zh`、`translator`、`quality_flags` 等字段，生成 manifest 时会保留这些信息。

## Run Locally / 本地运行

Run a short smoke translation first. Providers without configured credentials will be recorded as `not_available`, which is useful for checking the pipeline without exposing keys.

建议先运行一个很小的 smoke translation。未配置凭证的系统会被记录为 `not_available`，可用于检查流程而不暴露密钥。

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

只对最终译文评分：

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

运行固定随机种子的 LLM-as-judge 评价：

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

## Run On A Server / 服务器运行

The server workflow uses the same Python entry points. For long API runs, split the task into deterministic shards and run each shard in the background.

服务器端与本地使用同一套 Python 入口。长时间 API 任务可按确定性 shard 切片，并在后台运行。

Example for shard 0 of 4:

4 个切片中的第 0 个示例：

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

全部切片完成后合并：

```bash
PYTHONPATH=src .venv/bin/python scripts/merge_jsonl_outputs.py \
  --base-path results/full397_outputs_shard0.jsonl \
  --extra-path results/full397_outputs_shard1.jsonl \
  --extra-path results/full397_outputs_shard2.jsonl \
  --extra-path results/full397_outputs_shard3.jsonl \
  --output-path results/full397_outputs.jsonl
```

Then run the same scoring and judge commands shown in the local workflow.

之后继续运行本地流程中相同的评分和 LLM-as-judge 命令。

## Verification / 上传前检查

```powershell
.\.venv\Scripts\python -m compileall -q src scripts
git status --short
git ls-tree -r --name-only HEAD
```

Before pushing, confirm that the tracked file list contains no `results/`, `reports/`, `.env`, `.venv`, raw data, workbooks, PDFs, archives, private mappings, or local absolute paths.

推送前请确认 Git 跟踪文件清单中没有 `results/`、`reports/`、`.env`、`.venv`、原始数据、工作簿、PDF、压缩包、私有映射或本机绝对路径。
