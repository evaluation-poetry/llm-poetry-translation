# Data Policy

This study uses published modern Chinese poems and English translations whose copyrights and reuse conditions belong to their respective sources. For that reason, this public repository does not redistribute poem texts, reference translations, compiled datasets, model outputs, private mappings, or raw evaluation files.

The only public data file is `source_urls.json`, which records source websites and seed pages used to identify the research material. These URLs are provenance references; they are not a license to redistribute the underlying texts.

The `raw/`, `interim/`, and `processed/` folders are local workspaces kept only by `.gitkeep` files. Their real contents are ignored by git.

To reproduce the experiments, prepare a local JSONL dataset at:

```text
data/processed/bilingual_modern_chinese_poetry.jsonl
```

Check source-site terms and permissions before preparing or using local copies. The expected schema is documented in `docs/dataset_schema.md`.
