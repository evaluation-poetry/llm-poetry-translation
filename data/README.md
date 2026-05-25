# Data Directory / 数据目录

This directory intentionally contains no raw dataset.

本目录有意不存放原始数据集。

The study uses published modern Chinese poems and English translations whose copyrights and reuse conditions belong to their respective sources. For that reason, this public repository cannot redistribute poem texts, reference translations, compiled datasets, model outputs, or evaluation results.

本研究涉及已发表的中国现代诗及其英文译文，其版权与再使用条件归属各来源方。因此，本公开仓库不能重新分发诗歌正文、参考译文、已整理数据集、模型输出或评测结果。

The only public data file is `source_urls.json`, which records source websites and seed pages used to identify the research material. These URLs are references for provenance; they are not a license to redistribute the underlying texts.

唯一公开的数据文件是 `source_urls.json`，其中记录了用于定位研究材料的来源网站和入口页面。这些链接用于标明来源，并不构成对底层文本的再分发授权。

The `raw/`, `interim/`, and `processed/` folders are local workspaces kept only by `.gitkeep` files. Their real contents are ignored by git.

`raw/`、`interim/` 和 `processed/` 是本地工作目录，公开仓库中仅保留 `.gitkeep` 占位文件；这些目录中的真实内容均被 git 忽略。

To reproduce the experiments, prepare a local JSONL dataset at `data/processed/bilingual_modern_chinese_poetry.jsonl` after checking the relevant source-site terms and permissions. The expected schema is documented in the top-level `README.md`.

如需复现实验，请先确认相关来源网站的使用条款与权限，再在本地整理 JSONL 数据集，并保存为 `data/processed/bilingual_modern_chinese_poetry.jsonl`。所需字段格式见仓库根目录的 `README.md`。
