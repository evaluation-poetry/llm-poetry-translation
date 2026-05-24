# Data Directory

This directory contains public source URLs only. It does not contain poem text, reference translations, collected datasets, model outputs, or result files.

本目录只包含公开来源链接，不包含诗歌正文、参考译文、已整理数据集、模型输出或结果文件。

- `source_urls.json`: public source URLs used to identify where the research material can be found.
- `raw/`, `interim/`, `processed/`: local-only workspaces ignored by git.

Readers who reproduce the experiment should prepare their own local JSONL dataset under `data/processed/`, subject to the source websites' terms.

读者如需复现实验，应在遵守来源网站条款的前提下，自行在 `data/processed/` 下准备本地 JSONL 数据集。
