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

python scripts/compare_te_tts.py \
  --models both \
  --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \
  --text "క్షమించండి, మళ్లీ చెప్పగలరా?" \
  --ref-text "EXACT_TRANSCRIPT_OF_audio.flac" \
  --hf-repo BelluAi/te-tts-ab-listen

# Parler only (no ref):
python scripts/compare_te_tts.py --models parler --text "నమస్కారం" --hf-repo BelluAi/te-tts-ab-listen
```

If you see `ImportError: cannot import name 'Imports' from 'wandb.proto...'`:

```bash
pip install --force-reinstall --no-cache-dir "wandb>=0.19"
```

Outputs `te_tts_ab/wav/{000_indicf5,000_parler}.wav` + `manifest.jsonl`, then uploads (private by default). Listen on the Hub **Files** tab.

## Output (moshi train consumes this)

- `wav/*.wav` — 24 kHz stereo (L=agent, R=user)
- `wav/*.json` — agent word times + `extraction` labels
- `train.jsonl` — `{path, duration, extraction?}`
- `dataset_meta.json`
