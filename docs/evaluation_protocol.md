# Evaluation Protocol

The study evaluates only final English translations. Hidden reasoning traces, provider-side thinking summaries, and visible workflow contamination are not used as metric or judge inputs.

## Translation Systems

The full experiment contains seven systems:

| System id | Public label | Inference setting |
| --- | --- | --- |
| `baidu_translate` | Baidu Translate | Non-LLM baseline |
| `deepseek_v4_flash_non_thinking` | DeepSeek V4 Flash | Non-thinking |
| `deepseek_v4_flash_thinking` | DeepSeek V4 Flash | Thinking |
| `qwen36_plus_non_thinking` | Qwen3.6 Plus | Non-thinking |
| `qwen36_plus_thinking` | Qwen3.6 Plus | Thinking |
| `claude_sonnet46_non_thinking` | Claude Sonnet 4.6 | Non-thinking |
| `claude_sonnet46_thinking` | Claude Sonnet 4.6 | Thinking |

The primary comparison is paired within each model family: non-thinking versus thinking-enabled with the same prompt family and the same local input records.

## Translation Input Rule

Each provider receives only the Chinese source poem. The following information is excluded from translation prompts:

- Human English reference translation.
- English title.
- Translator name.
- Source subset label.
- Model identity.
- System identity.

This source-only design prevents reference leakage and keeps the thinking switch as the main manipulated provider variable.

## Automatic Metrics

Automatic metrics are computed on all 397 local records for all seven systems when outputs are available.

| Metric | Implementation | Interpretation |
| --- | --- | --- |
| COMET | `Unbabel/wmt22-comet-da` | Source-aware reference metric, higher is better. |
| BERTScore F1 | `microsoft/deberta-xlarge-mnli` by default | Semantic similarity to reference, higher is better. |
| SacreBLEU/BLEU | SacreBLEU corpus BLEU | N-gram reference overlap, higher is better. |
| chrF++ | SacreBLEU chrF with word order 2 | Character/word overlap, higher is better. |
| Line-level TER | SacreBLEU TER over paired poem lines | Edit diagnostic, lower is better. |
| LineDiff | Mean absolute non-empty line-count difference | Formal diagnostic, lower is better. |
| LenRatio | Hypothesis length divided by reference length | Formal diagnostic, closer to 1 is often better. |

Line-level TER pairs non-empty hypothesis and reference lines by position, padding missing lines with empty strings. This is a poetry-specific diagnostic, not a conventional sentence-level TER claim.

## LLM-as-Judge Evaluation

The judge layer uses a fixed source-balanced 20-poem sample:

- Seed: `20260513`.
- Five poems from each source subset.
- `belt_road_literary_network` contributes all 5 local records.
- Candidate order is independently shuffled per record using a stable seed.
- Candidate-to-system mapping is written to a private mapping file and must not be shared with annotators.

For each poem, the judge sees:

1. Chinese source poem.
2. Authoritative human English reference translation.
3. Seven anonymous candidate translations.

The judge is DeepSeek V4 Pro in non-thinking mode. It scores 11 dimensions from 0 to 10 in 0.5 increments and returns strict JSON. The dimensions are defined in [judge_dimensions.md](judge_dimensions.md). The full prompt is in [prompts.md](prompts.md).

The source-level mean is the average across the 5 sampled poems within each source subset. OMP, Overall Modern Poetry, is the unweighted mean of the four source means. It is not the raw 20-poem mean.

## Human Evaluation

Human evaluation uses the same 20 poems and seven systems. The private workbook contains 140 anonymous items. Each row gives the source poem, human reference, and one anonymous candidate translation with no system identity or inference setting.

Seven annotators score each item on six dimensions with a 1-6 integer scale:

| Code | Dimension |
| --- | --- |
| MF | Meaning and Fidelity |
| IR | Imagery, Rhetoric, and Cultural Transfer |
| EV | Thought, Emotion, Voice, and Style |
| LR | Lineation and Rhythm |
| MD | Modernity and Defamiliarization |
| PT | Poeticity and Overall Aesthetic Force |

Scale anchors:

- 1: severe translation failure.
- 4: usable baseline quality.
- 5: comparable to the human reference.
- 6: can exceed the reference in poetic expression.

OMP is computed with the same unweighted source-mean strategy as the LLM-as-judge layer.

## Interpretation

Automatic metrics, LLM-as-judge results, and human evaluation answer related but different questions. Automatic metrics reward similarity to a single reference over all 397 poems. Judge and human layers use a source-balanced 20-poem sample and poetry-specific rubrics. The paper therefore frames differences across these layers as metric/judge/human disagreement rather than as a single universal winner.
