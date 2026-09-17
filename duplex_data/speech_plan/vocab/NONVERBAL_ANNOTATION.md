# Nonverbal vocal-event annotation guide

Use this with the speech-plan annotator (`python -m duplex_data.speech_plan.annotator`).
Machine catalog: [`nonverbal_events.json`](nonverbal_events.json).

## How to annotate

1. Run extract so `content` / prosody lanes exist (nonverbal stays empty until you label).
2. Open the clip in the annotator.
3. On the **NONVERBAL** lane, drag a span over the audible event.
4. Set **value**:
   - **Indic speech:** pick from **Indic (native transcript)** — store the **exact native script** (`हम्म`, `हाँ`, `హ్మ్`, `ம்ம்`, …). Do **not** use only Latin `hmm`/`hm` when the utterance is Indic.
   - **English / non-Indic:** English event id (`laugh`, `sigh`, `uh`, `backchannel`, …)
5. Save. `source` should be `human`.

## Form vs function

| Layer | What to store | Examples |
| --- | --- | --- |
| **Form (Indic)** | `Event.value` = **native script** | `हम्म`, `हाँ`, `హ్మ్`, `ഉം`, `ம்` |
| **Transliteration** | vocab metadata only | `hmm`, `hã`, `hm` |
| **Form (English)** | `Event.value` = event id | `laugh`, `sigh`, `uh` |
| **Function** | THINK / ACK via `indic_forms` | Hindi `हाँ` → ACK |

## Indic native transcripts (`Event.value`)

These are **direct orthographic transcripts** for the nonverbal lane.

| Language | Native (`Event.value`) | Transliteration | Function | Maps to (English ids) |
| --- | --- | --- | --- | --- |
| Hindi | हम्म | hmm | THINK/ACK | `hmm`, `thinking_sound`, `acknowledgement` |
| Hindi | हाँ | hã | ACK | `acknowledgement`, `agreement_sound`, `backchannel` |
| Hindi | उम्म | umm | THINK | `um`, `hesitation` |
| Hindi | उह | uh | THINK | `uh`, `hesitation` |
| Hindi | आह | aah | ACK | `ah`, `realization` |
| Hindi | ओह | oh | ACK | `oh`, `surprise` |
| Hindi | अरे | are | ACK | `surprise`, `oh` |
| Telugu | హ్మ్ | hm | THINK/ACK | `hm`, `thinking_sound`, `acknowledgement` |
| Telugu | ఆ | aa | ACK | `ah`, `backchannel` |
| Telugu | అహ్ | ah | ACK | `ah`, `realization` |
| Telugu | ఓహ్ | oh | ACK | `oh`, `surprise` |
| Telugu | ఉమ్ | um | THINK | `um`, `hesitation` |
| Malayalam | ഹ്മ് | hm | ACK | `hm`, `acknowledgement` |
| Malayalam | ആ | aa | ACK | `ah`, `backchannel` |
| Malayalam | ഉം | um | THINK/ACK | `um`, `mm`, `acknowledgement` |
| Malayalam | ഓഹ് | oh | ACK | `oh`, `surprise` |
| Tamil | ம் | m | ACK | `mm`, `acknowledgement` |
| Tamil | ம்ம் | mm | THINK/ACK | `mm`, `hmm`, `thinking_sound` |
| Tamil | ஹ்ம்ம் | hmm | ACK | `hmm`, `acknowledgement` |
| Tamil | ஆ | aa | ACK | `ah`, `backchannel` |
| Tamil | உம் | um | THINK | `um`, `hesitation` |
| Tamil | ஓ | o | ACK | `oh`, `realization` |
| Kannada | ಹ್ಮ್ | hm | THINK | `hm`, `thinking_sound` |
| Kannada | ಆ | aa | ACK | `ah`, `backchannel` |
| Kannada | ಉಂ | um | THINK/ACK | `um`, `mm`, `acknowledgement` |
| Kannada | ಓಹ್ | oh | ACK | `oh`, `surprise` |
| Bengali | হুম | hum | ACK | `hmm`, `acknowledgement` |
| Bengali | হুঁ | hũ | ACK | `hmm`, `acknowledgement` |
| Bengali | আহ | ah | ACK | `ah`, `realization` |
| Bengali | ওহ | oh | ACK | `oh`, `surprise` |
| Bengali | উম | um | THINK | `um`, `hesitation` |
| Marathi | हम्म | hmm | THINK | `hmm`, `thinking_sound` |
| Marathi | हो | ho | ACK | `acknowledgement`, `backchannel` |
| Marathi | अह | ah | ACK | `ah`, `realization` |
| Marathi | ओह | oh | ACK | `oh`, `surprise` |
| Marathi | उम्म | umm | THINK | `um`, `hesitation` |

Set clip `language` to `hi` / `te` / `ta` / `kn` / `ml` / `bn` / `mr` so the UI defaults to that language’s native form.

## English categories (event ids)

- **conversational**: `hmm`, `hm`, `mhm`, `uh`, `um`, `uhm`, `ah`, `oh`, `eh`, `mm`, `huh`, `uh_huh`, `mm_hmm`, `yeah`, `wow`, `whoa`, `ooh`, `aww`, `ugh`, `ouch`, `shh`, `tsk`, …
- **emotional / laughter**: `laugh`, `speech_laugh`, `chuckle`, `giggle`, `breathy_giggle`, `nervous_giggle`, `snicker`, `snorting_giggle`, `guffaw`, `cackle`, `hehe`, `haha`, `smile_voice`, …
- **emotional / surprise**: `surprise`, `mild_surprise`, `strong_surprise`, `surprised_gasp`, `fearful_gasp`, `sharp_inhale`, `startle`, `admiration`, `alarm`, `gasp`, …
- **breathing**: `inhale`, `exhale`, `audible_breath`, `deep_breath`, `sigh`, `breath_before_speech`
- **thinking**: `thinking_sound`, `hesitation`, `prolonged_hesitation`, `er`, `erm`, `uhh`, `umm`, `hmm_long`, `word_search`, `tentative`, `self_correction`, `unfinished_utterance`
- **questioning**: `huh_question`, `eh_question`, `hm_question`, `what_token`, `pardon`, `nonunderstanding`, `confirmation_seek`, `questioning_rise`
- **physical**: `cough`, `sneeze`, `yawn`, `throat_clear`, `sniff`, `lip_smack`, `tongue_click`
- **interaction**: `backchannel`, `continuer`, `agreement_sound`, `disagreement_sound`, `realization`, `confusion`, `acknowledgement`, `sympathy_sound`, `mild_surprise_feedback`, `strong_surprise_feedback`, …

## Indic — surprise / question / laugh / think (native `Event.value`)

| Language | Surprise / alarm | Questioning | Laugh | Thinking |
| --- | --- | --- | --- | --- |
| Hindi | वाह, हाय, अरे, उफ़, क्या | क्या?, हाँ?, ए?, हम्म?, अच्छा? | हाहा, हेहे, हीही | हम्म, उम्म, उह |
| Telugu | అయ్యో, అబ్బా, ఏమి | ఏమి?, ఆ?, హ్మ్? | హాహా, హేహే | హ్మ్, ఉమ్ |
| Tamil | ஐயோ, அடா, என்ன | என்ன?, ஆ?, ம்? | ஹாஹா, ஹேஹே | ம்ம், உம், ஹ்ம்ம் |
| Kannada | ಅಯ್ಯೋ, ಅಬ್ಬಾ | ಏನು?, ಆ?, ಹ್ಮ್? | ಹಾಹಾ | ಹ್ಮ್, ಉಂ |
| Malayalam | അയ്യോ | എന്താ?, ആ?, ഹ്മ്? | ഹാഹാ | ഹ്മ്, ഉം |
| Bengali | আরে, বাহ | কী?, হুম? | হাহা, হেহে | হুম, উম |
| Marathi | अरे, वाह | काय?, हो? | हाहा, हेहे | हम्म, उम्म |

Rising tokens with `?` in the native string mark **questioning** forms (prosody + orthography convention for annotators).

## Bibliography

1. DualTurn — arXiv:2603.08216 (2026)
2. Trouvain & Truong 2012 — Comparing NVVs in conversational corpora
3. Trouvain & Werner 2020 — Comparing NVV annotations
4. Truong et al. 2014 — Sigh annotation scheme
5. Trouvain 2014 — Laughing, Breathing, Clicking (speech-laugh)
6. Schuller et al. 2012 — Linguistic vs non-linguistic vocalizations
7. LAION VocalBurst taxonomy — laughter & gasp subtypes
8. Feedback functions (HAL 2023) — continue / nonunderstanding / mild–strong surprise
9. NonVerbalSpeech-38K — arXiv:2508.05385
10. NVSpeech — arXiv:2508.04195
11. IndicVoices — arXiv:2403.01926
12. Prasad et al. 2010 — Hindi discourse particle *hã*
13. WILDRE 2024 — Disfluency corpora for Indian languages
