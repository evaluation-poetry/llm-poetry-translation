# LLM-as-Judge Dimensions

The LLM-as-judge layer scores anonymous candidate translations on 11 poetry-oriented dimensions. The judge uses a 0-10 scale in 0.5 increments. Higher is better.

These dimensions are broader than automatic metrics. They are intended to capture source fidelity, relation to the human reference, poetic form, modern poetic style, and English literary readability. They are diagnostic signals, not an independent gold standard.

## 1. Semantic Fidelity

Measures whether the candidate preserves the source poem's core meaning. A strong translation keeps the main scene, speaker relations, events, images, implied logical relations, and ambiguities without adding unsupported narrative content. It may choose different wording from the reference, but it should not distort what the Chinese poem says or implies.

## 2. Similarity to Reference

Measures thematic, stylistic, and interpretive proximity to the authoritative human English reference. The reference is treated as an important benchmark, not as the only acceptable translation. A candidate can score highly while differing from the reference if the difference remains faithful to the Chinese source and works poetically in English.

## 3. Imagery and Rhetoric

Measures preservation or persuasive transformation of images and rhetorical devices. This includes metaphor, simile, personification, parallelism, symbolic images, sensory texture, ambiguity, and rhetorical tension. A strong translation keeps image logic alive rather than flattening it into paraphrase.

## 4. Thought and Emotion

Measures whether the translation conveys the poem's intellectual movement and emotional pressure. This dimension tracks mood, restraint, shifts of feeling, implicit argument, affective density, and the deeper thought carried by the poem's images and lineation.

## 5. Lineation and Rhythm

Measures the effectiveness of line breaks, stanza breaks, pauses, pacing, rhythm, sound patterning, and free-verse musicality in English. A strong translation respects the source's formal pressure while remaining readable as English poetry. It should avoid arbitrary prose wrapping or mechanically preserving lines when the result breaks poetic cadence.

## 6. Modernity and Defamiliarization

Measures whether the translation retains modern poetic texture. This includes experimental perception, estrangement, unusual collocations, contemporary consciousness, urban or existential pressure, and departures from conventional poetic diction. A strong translation does not over-normalize the source into plain explanatory prose.

## 7. Cultural and Idiomatic Transfer

Measures handling of culture-specific expressions, historical references, idioms, proper names, local images, calendar terms, and Chinese-specific phrases. A strong translation either transfers the item clearly or creates an English phrasing that preserves its force without excessive footnoting.

## 8. Voice, Tone, and Style

Measures preservation of the source poem's speaking posture and stylistic character. Relevant qualities include quietness, irony, fragmentation, solemnity, colloquial directness, lyric density, narrative plainness, or experimental compression. A strong translation sounds like a coherent poetic voice rather than a neutral gloss.

## 9. Poeticity

Measures poetic density, suggestiveness, aesthetic tension, image resonance, and memorability. A strong translation has lines that feel crafted rather than merely transferred. This dimension is intentionally strict: a semantically adequate translation can still score lower if it lacks poetic force.

## 10. English Naturalness

Measures whether the candidate reads as strong English poetry rather than rigid translationese. This does not require ordinary prose grammar. Poetic deviation is acceptable when it is controlled, expressive, and readable. Awkward calques, unidiomatic syntax, and unintentionally stiff phrasing lower this score.

## 11. Overall Impression

Measures the overall strength of the candidate translation after considering all preceding dimensions. The score should reflect the judge's integrated assessment of fidelity, poetic form, style, English quality, and literary effect.

## Relationship to Human Evaluation

Human evaluation in the paper uses six dimensions on a 1-6 integer scale:

| Human dimension | Relation to judge dimensions |
| --- | --- |
| MF, Meaning and Fidelity | Mainly Semantic Fidelity and Similarity to Reference |
| IR, Imagery/Rhetoric/Cultural Transfer | Imagery and Rhetoric plus Cultural and Idiomatic Transfer |
| EV, Thought/Emotion/Voice/Style | Thought and Emotion plus Voice, Tone, and Style |
| LR, Lineation and Rhythm | Lineation and Rhythm |
| MD, Modernity and Defamiliarization | Modernity and Defamiliarization |
| PT, Poeticity/Overall Aesthetic Force | Poeticity and Overall Impression |
