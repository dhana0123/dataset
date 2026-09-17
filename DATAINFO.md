# Dataset candidates (DATAINFO)

Readable wide view (horizontal scroll): [`index.html`](index.html)

Single catalog for open-source and internal speech data. Use before filling the XLSX (license / commercial / download / metadata).

| # | Pri | Dataset | Scale | Indian langs | Pipeline / type | DualTurn | 2-spk conv | Turn-taking | Overlap | Timestamps | Transcript | Link |
| -: | --- | ------- | ----: | -----------: | --------------- | -------- | ---------: | ----------: | ------: | ---------: | ---------: | ---- |
| 1 | red | IndicVoices | ~23.7K h | 22 | ASR / conversation | orange | some | limited | limited | speech-level | yes (subset) | [HF](https://huggingface.co/datasets/ai4bharat/IndicVoices) |
| 2 | red | IndicVoices-R | 1,704 h | — | TTS | — | — | — | — | — | — | [HF](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| 3 | red | IndicTTS | large | — | TTS | — | — | — | — | — | — | [IITM](https://www.iitm.ac.in/donlab/indictts/database) |
| 4 | red | Indic DiarBench | ~108 h | 22 | Duplex / diarization | red | yes / multi | yes | yes | yes | yes | [HF](https://huggingface.co/datasets/sarvamai/indic-diarbench) |
| 5 | red | Your 100h recordings | ~100 h | target langs | DualTurn (core) | red | yes | yes | yes | can create | can create | Internal |
| 6 | red | Rasa | ~3 langs | yes | Expressive TTS | — | — | — | — | — | — | [GitHub](https://github.com/AI4Bharat/Rasa) |
| 7 | orange | SLR66 — Telugu | speech corpus | Te | ASR / TTS | — | — | — | — | — | — | [OpenSLR 66](https://www.openslr.org/66/) |
| 8 | orange | SLR65 — Tamil | speech corpus | Ta | ASR / TTS | — | — | — | — | — | — | [OpenSLR 65](https://www.openslr.org/65/) |
| 9 | orange | SLR64 — Marathi | speech corpus | Mr | ASR / TTS | — | — | — | — | — | — | [OpenSLR 64](https://www.openslr.org/64/) |
| 10 | orange | SLR63 — Malayalam | speech corpus | Ml | ASR / TTS | — | — | — | — | — | — | [OpenSLR 63](https://www.openslr.org/63/) |
| 11 | orange | SLR78 — Gujarati | speech corpus | Gu | ASR / TTS | — | — | — | — | — | — | [OpenSLR 78](https://www.openslr.org/78/) |
| 12 | orange | SLR79 — Kannada | speech corpus | Kn | ASR / TTS | — | — | — | — | — | — | [OpenSLR 79](https://www.openslr.org/79/) |
| 13 | orange | Gram Vaani / SLR118 | 1,111 h | Hi | Noisy / spontaneous ASR | — | — | — | — | — | — | [OpenSLR 118](https://www.openslr.org/118/) |
| 14 | orange | MUCS / SLR103 | ~300 h | Hi/Mr/Or | ASR | — | — | — | — | — | — | [OpenSLR 103](https://www.openslr.org/103/) |
| 15 | orange | Doctor–Patient Indic | — | 9 | Duplex / dialogue | — | — | — | — | — | — | [HF search](https://huggingface.co/datasets?search=doctor+patient+indic+speech) |
| 16 | yellow | VocalSound | 21K clips | no | Nonverbal detector | — | — | — | — | — | — | [GitHub](https://github.com/YuanGongND/vocalsound) |
| 17 | yellow | ReCANVo | — | no | Nonverbal / affective | — | — | — | — | — | — | [HF search](https://huggingface.co/datasets?search=ReCANVo) |
| 18 | yellow | Skit Emotion TTS | 30 min | en-IN | Emotion TTS | — | — | — | — | — | — | [GitHub](https://github.com/skit-ai/emotion-tts-dataset) |
| 19 | yellow | Indian TTS Emotion 60min | ~68 min | Hi/en-IN | Emotion TTS | — | — | — | — | — | — | [HF](https://huggingface.co/datasets/sarthwa8/indian-tts-emotion-60min) |
| 20 | yellow | otoSpeech Turn 104h | 104 h | no | Full duplex / architecture | — | yes | yes | — | — | — | [HF](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h) |
| 21 | red | Humyn MultiSpeaker ASR | ~1.06 GB sample | 14 | DualTurn base | red | yes | yes | partial | yes | yes | [HF](https://huggingface.co/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR) |
| 22 | red | Indic Natural Conversations | ~1.99 GB sample | 9 | Conversation / DualTurn | red | yes | yes | partial | yes | yes | [HF](https://huggingface.co/datasets/snorbyte/indic-audio-natural-conversations-sample) |
| 23 | orange | VAANI | ~31K h | 100+ | ASR / accents | orange | no | no | no | speech-level | partial | [HF](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) |
| 24 | yellow | Google FLEURS | ~12h / lang | 14 | ASR benchmark | yellow | no | no | no | no turn ts | yes | [HF](https://huggingface.co/datasets/google/fleurs) |
| 25 | yellow | TTS Indian Languages | 66.4 min | yes | TTS pilot | — | — | — | — | — | — | [HF](https://huggingface.co/datasets/praneeetha/tts-indian-languages) |
| 26 | yellow | Indian TTS Emotion 28.8 min | 28.8 min | yes | Emotion TTS | — | — | — | — | — | — | [HF](https://huggingface.co/datasets/champTUSHARg007/indian-tts-dataset) |

**Pri:** red = P0 core · orange = P1 scale/lang · yellow = P2 niche/benchmark  
**DualTurn:** red = train/eval · orange = limited duplex value · yellow = ASR-only · — = not DualTurn-focused

### Tracking checklist (from roadmap)

- [ ] Find open-source datasets *(this table is the seed list)*
- [ ] Put every candidate into XLSX
- [ ] Check license
- [ ] Check commercial use
- [ ] Download candidates
- [ ] Organize metadata

See also: [`TODO.md`](../TODO.md), speech-plan annotator under [`duplex_data/speech_plan/`](duplex_data/speech_plan/).
