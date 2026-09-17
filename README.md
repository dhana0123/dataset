# Indic Inner Monologue packer (`data/`)

Standalone CLI for Indic duplex train packs. **Not part of moshi** — use a separate venv.

## Install

```bash
cd data
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# if torch missing / CPU-only after installs:
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Optional IndicConformer + WhisperX: `pip install -e ".[indic]"` and install [AI4Bharat NeMo](https://github.com/AI4Bharat/NeMo) `nemo-v2`.

## Gated HF models (same account as `HF_TOKEN`)

- https://huggingface.co/datasets/ai4bharat/IndicVoices
- https://huggingface.co/pyannote/speaker-diarization-community-1

## Smoke (3 Hindi clips → private Hub)

```bash
export HF_TOKEN=...   # do not paste tokens into chat
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -m duplex_data.prepare \
  --out ./moshi_im_smoke3 \
  --indicvoices --streaming \
  --indicvoices-config hindi \
  --max-clips 3 \
  --asr-backend whisper \
  --device cuda \
  --hf-dataset BelluAi/dupxel-indic
```

Defaults: **Whisper** ASR, **community-1** diarization, Hub push **on**.

## Telugu TTS A/B listen test (IndicF5 + Parler)

Side-by-side samples for evaluating TTS before synthetic duplex training.
Use a **separate** venv — **do not use `venv-moshi`** (broken `wandb` / dep clashes).

```bash
python3.10 -m venv ~/tts-eval/.venv && source ~/tts-eval/.venv/bin/activate
pip install -U pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/ai4bharat/IndicF5.git"
pip install "git+https://github.com/huggingface/parler-tts.git"
pip install soundfile huggingface_hub transformers accelerate numpy
pip install --force-reinstall --no-cache-dir "wandb>=0.19"

export HF_TOKEN=...   # write access; do not paste into chat

cd ~/dataset   # or path to this repo on the GPU box
# Default IndicF5 ref is repo-root audio.flac (auto-converted to 24 kHz wav).
# --ref-text MUST be the exact words spoken in that file.

# Girl voice (Lalitha) is default (--voice female):
python scripts/compare_te_tts.py \
  --models parler \
  --voice female \
  --parler-styles happy,joyful,sad,angry,excited \
  --text "హ్మ్, ఓకే, ఇది నా కాంటాక్ట్ నెంబరు. మీరు అడ్రెస్ చెప్తే నేను అక్కడికి వస్తాను. అక్కడ ఒక ఓటిపి ఆర్డర్ చేయండి." \
  --hf-repo BelluAi/te-tts-ab-listen

# Leela (high-pitched, fast, cheerful — Parler example caption):
python scripts/compare_te_tts.py \
  --models parler \
  --parler-styles leela,happy,joyful \
  --text "హ్మ్, ఓకే, ఇది నా కాంటాక్ట్ నెంబరు. మీరు అడ్రెస్ చెప్తే నేను అక్కడికి వస్తాను. అక్కడ ఒక ఓటిపి ఆర్డర్ చేయండి." \
  --hf-repo BelluAi/te-tts-ab-listen

# Male: --voice male (Prakash). List styles: python scripts/compare_te_tts.py --list-styles
# Packs: basic | emotions | positive | callcenter | ab | all
```

Example: `--parler-styles leela` or `--parler-styles emotions`.

If you see `ImportError: cannot import name 'Imports' from 'wandb.proto...'`:

```bash
pip install --force-reinstall --no-cache-dir "wandb>=0.19"
```

Uploads private Hub dataset by default. Listen on the Hub **Files** tab.

## Synthetic duplex (open Sarvam-M → Parler → IM pack)

PersonaPlex-style pipeline: **local open-source** [`sarvamai/sarvam-30b`](https://huggingface.co/sarvamai/sarvam-30b) (default; strong Indic Te/Hi/Ta/Kn) writes two-speaker Telugu scripts → Indic Parler TTS → stereo stitch → same `wav/` + `SPEAKER_MAIN` JSON + `train.jsonl` as the packer. **No Sarvam API key.**

Needs a GPU with enough VRAM for Sarvam-30B (MoE; use `--llm-load-in-4bit` if tight) plus Parler. Prefer the TTS/eval venv with `parler-tts` installed.

**Transformers mismatch (default = auto-fallback):** Sarvam-30B needs `ALL_ATTENTION_FUNCTIONS` (`transformers>=4.57`). Typical parler-tts venvs ship older transformers (e.g. 4.46). By default the CLI **warns and loads** [`sarvamai/sarvam-m`](https://huggingface.co/sarvamai/sarvam-m) instead so synthetic runs continue. Pass `--strict-llm` to fail hard and enforce 30B.

To keep using 30B:

```bash
pip install -U "transformers>=4.57.0" accelerate
```

If that breaks Parler, split LLM and TTS:

```bash
# 1) scripts only (LLM venv with newer transformers)
python -m duplex_data.synthetic --out ./moshi_im_synth_te5 --num-dialogs 5 \
  --domains recruitment,customer_support,banking --dry-run-scripts --strict-llm

# 2) TTS+pack from saved scripts (older transformers OK if Parler needs it)
python -m duplex_data.synthetic --out ./moshi_im_synth_te5 --num-dialogs 5 \
  --scripts-dir ./moshi_im_synth_te5/scripts --hf-dataset BelluAi/dupxel-indic
```

Or pin the mid-size model explicitly: `--llm-model sarvamai/sarvam-m`

```bash
cd data && pip install -e ".[synthetic]"
# also: pip install "git+https://github.com/huggingface/parler-tts.git"

export HF_TOKEN=...   # download weights + optional Hub push

# Placeholder scripts only (no LLM):
python -m duplex_data.synthetic \
  --out ./moshi_im_synth_smoke \
  --lang te \
  --num-dialogs 2 \
  --domains recruitment,customer_support \
  --dry-run-scripts --skip-llm

# Full: default LLM (30B, or auto sarvam-m on old transformers) + Parler + pack (+ Hub):
python -m duplex_data.synthetic \
  --out ./moshi_im_synth_te \
  --lang te \
  --num-dialogs 5 \
  --domains recruitment,customer_support,banking \
  --agent-style leela \
  --user-voice male \
  --device cuda \
  --hf-dataset BelluAi/dupxel-indic

# List domains:
python -m duplex_data.synthetic --out /tmp/x --list-domains
```

Reuse scripts later: `--scripts-dir ./moshi_im_synth_te/scripts` (skips LLM).

## Output (moshi train consumes this)

- `wav/*.wav` — 24 kHz stereo (L=agent, R=user)
- `wav/*.json` — agent word times + `extraction` labels
- `train.jsonl` — `{path, duration, extraction?}`
- `dataset_meta.json`

## Speech-event plans (research annotation)

Separate from the IM packer: extract overlapping speech-control **events**
(content / pitch / energy / rate / emphasis / pause / boundary) into a 2D
temporal JSON, then annotate global state (and edit auto events) in a local UI.

```bash
cd data
pip install -e ".[speech-plan]"

# 1) Audio → draft speech_plan.json (+ copied wav)
python -m duplex_data.speech_plan.extract path/to/clip.wav \
  --out ./speech_plans --lang auto --asr-backend whisper --device cpu

# Or a folder of clips:
python -m duplex_data.speech_plan.extract path/to/wav_dir \
  --out ./speech_plans --lang te --asr-backend whisper --device cuda

# 1b) Existing transcript: skip ASR, WhisperX forced-align only
python -m duplex_data.speech_plan.align \
  --audio clip.wav --transcript clip.txt --lang te \
  --out ./alignments/clip.json --device cuda

# Or wire align into extract (same draft speech_plan.json path):
python -m duplex_data.speech_plan.extract clip.wav \
  --transcript clip.txt --out ./speech_plans --lang te --device cuda

# Optional aligner override (HF Wav2Vec2ForCTC id):
#   --align-model facebook/mms-300m-1130-forced-aligner

# 2) Annotate (global state, emphasis, boundaries, edit spans)
python -m duplex_data.speech_plan.annotator --root ./speech_plans
# open http://127.0.0.1:8765
```

Console scripts: `duplex-speech-extract`, `duplex-speech-align`, `duplex-speech-annotate`.

**Auto vs human**

| Lane / field | Source |
| --- | --- |
| CONTENT + word times | ASR (Whisper word timestamps, or IndicConformer + WhisperX), **or** provided transcript + WhisperX align |
| PAUSE / PITCH / ENERGY / RATE | Auto from timing + Praat (parselmouth) / librosa |
| EMPHASIS / BOUNDARY | Weak auto-propose; confirm in UI |
| NONVERBAL | Human — English event ids **or Indic native-script transcripts** (`हम्म`, `హ్మ్`, …) |
| GLOBAL STATE | Human (calm / serious / excited / reassuring / …) |

### WhisperX align languages

Forced alignment uses language-specific Wav2Vec2 CTC models (WhisperX
`model_name`). IndicConformer is ASR-only and is **not** the aligner.

| Lang | Align model (HF) |
| --- | --- |
| hi | `theainerd/Wav2Vec2-large-xlsr-hindi` |
| te | `anuragshas/wav2vec2-large-xlsr-53-telugu` |
| ml | `gvs/wav2vec2-large-xlsr-malayalam` |
| ur | `kingabzpro/wav2vec2-large-xls-r-300m-Urdu` |
| ta | `Harveenchadha/vakyansh-wav2vec2-tamil-tam-100` |
| kn | `Harveenchadha/vakyansh-wav2vec2-kannada-knm-560` |
| mr | `Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100` |
| gu | `Harveenchadha/vakyansh-wav2vec2-gujarati-gum-100` |
| bn | `Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200` |
| pa | `Harveenchadha/vakyansh-wav2vec2-punjabi-pam-100` |
| or | `Harveenchadha/vakyansh-wav2vec2-odia-orm-100` |

`--align-model <hf_id>` always wins over the table. MMS fallbacks are manual
only after pilot QA (not automatic).

### Nonverbal annotation

Vocab + guide (English taxonomy + Indic direct transcripts):

- [`duplex_data/speech_plan/vocab/nonverbal_events.json`](duplex_data/speech_plan/vocab/nonverbal_events.json)
- [`duplex_data/speech_plan/vocab/NONVERBAL_ANNOTATION.md`](duplex_data/speech_plan/vocab/NONVERBAL_ANNOTATION.md)

For Indic clips, set `Event.value` to the **native script** form (not only Latin `hmm`/`hm`). The annotator dropdown has an **Indic (native transcript)** group.

Each clip folder under `--out` looks like:

```text
speech_plans/<stem>/
  <stem>.wav
  speech_plan.json   # lanes + optional acoustic tracks sidecar
```

Events are overlapping **lanes**, not one serialized chain. Raw F0/intensity
stay under `tracks` for analysis; the annotation target is the discrete
`EVENT(type, start, end, value, source)` vocabulary.
