# Dataset Schema

The public repository does not include the dataset. To reproduce the experiments, prepare a local JSONL file at:

```text
data/processed/bilingual_modern_chinese_poetry.jsonl
```

Each line must be one JSON object. The experiment code preserves `record_id` and `source_id` in manifests, outputs, scores, and summaries.

## Required Fields

| Field | Type | Description |
| --- | --- | --- |
| `record_id` | string | Stable unique record identifier. |
| `source_id` | string | One of the four source-based subset identifiers. |
| `source_name` | string | Human-readable source name. |
| `language_pair` | string | Must be `zh-en`. |
| `original_zh` | string | Chinese source poem. This is the only text sent to translation systems. |
| `translation_en` | string | Human English reference. Used only for offline metrics and judge evaluation. |

The code also accepts legacy aliases `original_text` and `translation_text`, but the canonical public schema uses `original_zh` and `translation_en`.

## Optional Fields

| Field | Type | Description |
| --- | --- | --- |
| `page_url` | string | Source page URL for provenance. |
| `source_url` | string | Source website URL. |
| `title_zh` | string | Chinese title metadata. Never used in translation prompts. |
| `title_en` | string | English title metadata. Never used in translation prompts. |
| `poet_zh` | string | Chinese poet metadata. Never used in translation prompts. |
| `poet_en` | string | English poet metadata. Never used in translation prompts. |
| `translator` | string | Published translator metadata. Never used in translation prompts. |
| `original_lines` | list[string] | Optional source line segmentation. |
| `translation_lines` | list[string] | Optional reference line segmentation. |
| `quality_flags` | list[string] | Optional local quality flags. |
| `copyright_note` | string | Local rights note. |
| `access_date` | string | Local collection or access date. |
| `provenance` | object | Local parser or collection provenance. |

## Source Subsets

The four subsets are source-based provenance groups, not an official genre taxonomy.

| `source_id` | Label | Full experiment count |
| --- | --- | ---: |
| `modern_chinese_poetry` | 21st Century Chinese Poetry | 264 |
| `poetry_international_chinese` | Poetry International Chinese | 122 |
| `paper_republic_read` | Read Paper Republic | 6 |
| `belt_road_literary_network` | Belt and Road Literary Network | 5 |

## Minimal Example

```json
{
  "record_id": "example_record_id",
  "source_id": "modern_chinese_poetry",
  "source_name": "21st Century Chinese Poetry",
  "language_pair": "zh-en",
  "original_zh": "Chinese source poem text",
  "translation_en": "Reference English translation"
}
```

## Prompt Safety Rules

The translation pipeline constructs provider messages from the Chinese source poem (original_zh) only. Do not add the following fields to translation prompts:

- `translation_en`
- English title
- translator name
- source subset
- model identity
- system identity

Reference translations are used only after generation, for automatic metrics and LLM-as-judge evaluation.
