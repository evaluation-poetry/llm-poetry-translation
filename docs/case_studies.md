# Case Studies

The case-study format follows the spirit of Appendix A.3 and A.4 in *Can ChatGPT Really Understand Modern Chinese Poetry?*: present an example, show the model behavior, then analyze the example dimension by dimension. This repository adapts that structure from poetry understanding to poetry translation.

Because this is a public repository, the case-study documentation does not redistribute complete copyrighted poems, complete published references, or complete raw model outputs. Instead, it provides:

- selected case ids and source subsets,
- system pairs to compare,
- judge and human score summaries,
- qualitative focus points,
- a local exporter that can generate full private case packets from local artifacts.

Use `scripts/export_case_studies.py --include-text` only for private local analysis or for material you have permission to redistribute.

## Recommended Case-Study Structure

Each full private case should use this order:

1. **Case metadata.** Record id, source subset, source URL if available, line count, systems compared, prompt version, and evaluation layers used.
2. **Source and reference.** Chinese source and published English reference, if local rights permit use in the target document.
3. **Candidate translations.** Anonymous or named system outputs, depending on whether the audience is annotator-facing or paper-facing.
4. **Score block.** LLM-as-judge average over 11 dimensions and human mean over six dimensions when available.
5. **Dimension analysis.** Short paragraphs for semantic fidelity, imagery/rhetoric, thought/emotion/voice, lineation/rhythm, modernity/defamiliarization, cultural transfer, poeticity, and English naturalness.
6. **Interpretive conclusion.** What this case shows about non-thinking versus thinking, metric/judge/human disagreement, or source-specific difficulty.

The public paper can use a compressed version of this structure. The private supplement can include complete texts when rights and venue rules allow.

## Selected Excellent Cases

These cases were selected from local qualitative materials because they show high-quality LLM translations and clear differences between the non-thinking and thinking variants.

| Case | Record id | Source subset | Main systems | Judge score | Human mean | Focus |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | `8708f20283b61f97` | 21st Century Chinese Poetry | Claude Sonnet 4.6 non-thinking vs thinking | 8.36 vs 7.73 | 3.95 vs 1.90 | Culture-specific rendering, lunar-calendar phrase, colloquial reassurance, and a sunrise image. |
| C2 | `61dbce7cbc688130` | Poetry International Chinese | Qwen3.6 Plus non-thinking vs thinking | 9.36 vs 8.05 | 4.12 vs 3.36 | Lexical precision, idiomatic choices, and a key night-surf line. |
| C3 | `7c66535bb93c9034` | 21st Century Chinese Poetry | Claude Sonnet 4.6 non-thinking vs thinking | 9.05 vs 7.50 | 4.29 vs 3.21 | Poetic restraint, lineation, and whether abstract image language remains suggestive rather than explanatory. |

### Case C1: Claude Sonnet 4.6 Non-thinking

This is the strongest qualitative case for the human-evaluation preference. Human annotators separate the two Claude translations much more sharply than the LLM judge does. The non-thinking version is preferred because it renders culture- and idiom-bearing expressions more naturally and keeps the central sunrise image more compact and poetic. The thinking version is not malformed; its weakness is more subtle. It is line-aligned and readable, but it tends toward literal or stiff phrasing, which is exactly the type of degradation that automatic format diagnostics can miss.

Analysis dimensions to emphasize:

- **Cultural and Idiomatic Transfer:** compare how the translation handles place-name/culture-specific phrasing and lunar-calendar language.
- **Voice, Tone, and Style:** examine whether colloquial reassurance remains idiomatic in English.
- **Poeticity:** focus on whether the sunrise image stays vivid or is reduced to plain explanation.
- **Lineation and Rhythm:** check whether pauses and stanza breaks are preserved.

### Case C2: Qwen3.6 Plus Non-thinking

This case demonstrates that the non-thinking advantage is not limited to Claude. Qwen3.6 Plus non-thinking is preferred over its thinking variant by both the LLM judge and human annotators. The case is short enough to support close reading and is useful for showing how local lexical choices can change the force of a poem.

Analysis dimensions to emphasize:

- **Semantic Fidelity:** inspect whether the candidate preserves the event or scene without overexplaining it.
- **Imagery and Rhetoric:** examine the night-surf image as the pivotal line.
- **English Naturalness:** compare idiomatic alternatives that are close in meaning but different in poetic force.
- **Overall Impression:** connect local lexical choices to the larger reading effect.

### Case C3: Claude Sonnet 4.6 Non-thinking, Secondary Case

This is a backup case for the Claude family. It has a large LLM-judge gap and a clear human preference for the non-thinking version. It is useful when C1 is too focused on cultural transfer and the analysis needs a second example centered on lineation, abstract image language, and poetic restraint.

Analysis dimensions to emphasize:

- **Lineation and Rhythm:** compare whether syntactic flow respects line breaks.
- **Thought and Emotion:** inspect whether the translation keeps the emotion understated rather than stated outright.
- **Modernity and Defamiliarization:** check whether unusual image language remains strange in English.
- **Poeticity:** examine whether the candidate keeps suggestiveness without turning into paraphrase.

## Local Export Commands

Generate a public-safe redacted packet:

```powershell
.\.venv\Scripts\python scripts\export_case_studies.py `
  --output-path outputs\case_studies\case_studies_redacted.md
```

Generate a private full-text packet:

```powershell
.\.venv\Scripts\python scripts\export_case_studies.py `
  --include-text `
  --output-path outputs\case_studies\case_studies_private_full.md
```

The private packet includes source poems, references, and selected system outputs if those local artifacts are present. Keep it out of git unless redistribution is permitted.

## Why Not Publish Complete Cases Here?

The source poems and published translations are copyright-controlled, and raw model outputs belong to private experiment logs, so the repository ships code and templates rather than the texts themselves.
