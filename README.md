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

# Emotion pack (happy/joyful/sad/angry/…) or pick any comma list:
python scripts/compare_te_tts.py \
  --models parler \
  --parler-styles ab \
  --text "హ్మ్, ఓకే, ఇది నా కాంటాక్ట్ నెంబరు. మీరు అడ్రెస్ చెప్తే నేను అక్కడికి వస్తాను. అక్కడ ఒక ఓటిపి ఆర్డర్ చేయండి." \
  --hf-repo BelluAi/te-tts-ab-listen

# List every style + pack:
python scripts/compare_te_tts.py --list-styles

# Styles: clean phone backchannel listener happy joyful excited friendly playful
#         calm soft whisper tired sad angry annoyed fear surprised urgent
#         formal polite empathetic news happy_f soft_f phone_f joyful_f
# Packs:  basic | emotions | positive | callcenter | ab | gender | all
```

Example: `--parler-styles happy,joyful,sad,angry` or `--parler-styles emotions` (bigger).

If you see `ImportError: cannot import name 'Imports' from 'wandb.proto...'`:

```bash
pip install --force-reinstall --no-cache-dir "wandb>=0.19"
```

Uploads private Hub dataset by default. Listen on the Hub **Files** tab.

## Output (moshi train consumes this)

- `wav/*.wav` — 24 kHz stereo (L=agent, R=user)
- `wav/*.json` — agent word times + `extraction` labels
- `train.jsonl` — `{path, duration, extraction?}`
- `dataset_meta.json`
