#!/usr/bin/env python3
"""Telugu listen-test: IndicF5 + Indic Parler-TTS → local wavs → Hugging Face dataset.

Run in a dedicated TTS venv (not moshi / duplex-data packer venv).

  python scripts/compare_te_tts.py \\
    --models both \\
    --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \\
    --ref-text "exact transcript of audio.flac" \\
    --hf-repo BelluAi/te-tts-ab-listen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_AUDIO = REPO_ROOT / "audio.flac"

DEFAULT_TE_TEXTS = [
    "ఓకే, ఇది నా కాంటాక్ట్ నెంబరు. మీరు అడ్రెస్ చెప్తే నేను అక్కడికి వస్తాను. అక్కడ ఒక ఓటిపి ఆర్డర్ చేయండి.",
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "క్షమించండి, మళ్లీ చెప్పగలరా?",
]

# Telugu speakers: Prakash / Lalitha / Kiran (indic-parler-tts card).
# Prosody lives in the *description*. Use --parler-styles a,b,c for A/B.
PARLER_STYLE_PRESETS: dict[str, str] = {
    # --- baseline / channel ---
    "clean": (
        "Prakash's voice is clear and slightly expressive, speaking Telugu at a "
        "moderate pace with very high quality audio and almost no background noise."
    ),
    "phone": (
        "Prakash speaks Telugu like a real phone call: casual, slightly expressive, "
        "thinking while talking, with natural pauses and breath, moderate pace, "
        "close microphone, very high quality audio, almost no background noise."
    ),
    "backchannel": (
        "Prakash speaks Telugu in a very natural conversational phone style, "
        "with human hesitations and filled pauses like uh, uhh, um, and okay, "
        "soft backchannel acknowledgments, slight false starts, not reading a script, "
        "warm and slightly expressive, moderate speed, close-sounding recording, "
        "very clear high quality audio with almost no background noise."
    ),
    "listener": (
        "Prakash gives short soft Telugu listener responses on a phone call, "
        "with quick backchannels and filled pauses like uh-huh, uhh, mm, and okay, "
        "brief and natural, not exaggerated, close microphone, very high quality audio."
    ),
    # --- positive ---
    "happy": (
        "Prakash speaks Telugu in a happy, upbeat, smiling voice, cheerful and warm, "
        "slightly higher pitch, lively and positive energy, moderate pace, "
        "close microphone, very high quality audio with almost no background noise."
    ),
    "joyful": (
        "Prakash speaks Telugu with a joyful, bright, laughing-smile tone, "
        "very expressive and energetic, delighted and friendly, slightly fast pace, "
        "close-sounding recording, very clear high quality audio with almost no background noise."
    ),
    "excited": (
        "Prakash speaks Telugu excitedly, high energy, fast pace, elevated pitch, "
        "enthusiastic and animated, close microphone, very high quality audio."
    ),
    "friendly": (
        "Prakash speaks Telugu in a warm, friendly, welcoming tone, soft smile in the voice, "
        "relaxed moderate pace, close-sounding recording, very high quality audio."
    ),
    "playful": (
        "Prakash speaks Telugu playfully, light teasing smile, slightly higher pitch, "
        "bouncy and informal, moderate pace, very clear high quality audio."
    ),
    # --- calm / soft ---
    "calm": (
        "Prakash speaks Telugu calmly and steadily, soothing, low expressivity, "
        "slightly slow pace, balanced pitch, close microphone, very high quality audio."
    ),
    "soft": (
        "Prakash speaks Telugu softly and gently, quiet close-up voice, caring tone, "
        "slow to moderate pace, very clear audio with almost no background noise."
    ),
    "whisper": (
        "Prakash whispers Telugu quietly and intimately, breathy soft voice, "
        "very close microphone, slow pace, clear recording with almost no noise."
    ),
    "tired": (
        "Prakash speaks Telugu in a tired, weary voice, slightly lower pitch, "
        "slower pace, soft and drained, close microphone, very high quality audio."
    ),
    # --- negative / intense ---
    "sad": (
        "Prakash speaks Telugu sadly, low pitch, slow pace, soft and melancholic, "
        "slightly monotone with quiet emotion, close recording, very high quality audio."
    ),
    "angry": (
        "Prakash speaks Telugu angrily, sharp and forceful, higher intensity, "
        "faster pace, tense pitch, close microphone, very high quality audio."
    ),
    "annoyed": (
        "Prakash speaks Telugu in an annoyed, impatient tone, clipped and irritated, "
        "slightly fast pace, close-sounding recording, very high quality audio."
    ),
    "fear": (
        "Prakash speaks Telugu with fear and anxiety, shaky, hesitant, slightly higher pitch, "
        "uneven pace, close microphone, very high quality audio."
    ),
    "surprised": (
        "Prakash speaks Telugu with surprise, sudden raised pitch, animated and startled, "
        "expressive, moderate-fast pace, very clear high quality audio."
    ),
    "urgent": (
        "Prakash speaks Telugu urgently, fast and insistent, serious, slightly raised pitch, "
        "close microphone, very high quality audio with almost no background noise."
    ),
    # --- formal / customer-care ---
    "formal": (
        "Prakash speaks Telugu formally and politely, clear diction, measured moderate pace, "
        "professional customer-service tone, close microphone, very high quality audio."
    ),
    "polite": (
        "Prakash speaks Telugu politely and respectfully, soft smile, careful wording pace, "
        "warm but controlled, close-sounding recording, very high quality audio."
    ),
    "empathetic": (
        "Prakash speaks Telugu empathetically, caring and reassuring, soft and warm, "
        "slightly slow pace, close microphone, very high quality audio."
    ),
    "news": (
        "Prakash speaks Telugu like a news reader, clear, neutral, steady pace, "
        "professional broadcast tone, very high quality studio audio."
    ),
    # --- female voice variants (Lalitha) ---
    "happy_f": (
        "Lalitha's voice is happy and smiling, speaking Telugu cheerfully with a slightly "
        "higher pitch, warm and lively, moderate pace, close microphone, very high quality audio."
    ),
    "soft_f": (
        "Lalitha speaks Telugu softly and gently, caring feminine voice, slow moderate pace, "
        "close-sounding recording, very high quality audio with almost no background noise."
    ),
    "phone_f": (
        "Lalitha speaks Telugu like a casual phone call, natural and slightly expressive, "
        "thinking while talking, moderate pace, close microphone, very high quality audio."
    ),
    "joyful_f": (
        "Lalitha speaks Telugu joyfully with a bright smiling tone, very expressive and "
        "delighted, slightly fast pace, close microphone, very high quality audio."
    ),
}

# Named packs for --parler-styles (aliases expand before lookup).
PARLER_STYLE_PACKS: dict[str, list[str]] = {
    "basic": ["clean", "phone", "backchannel"],
    "emotions": ["happy", "joyful", "sad", "angry", "surprised", "fear", "calm"],
    "positive": ["happy", "joyful", "excited", "friendly", "playful"],
    "callcenter": ["formal", "polite", "empathetic", "phone", "calm"],
    "ab": ["backchannel", "happy", "joyful"],
    "gender": ["happy", "happy_f", "soft", "soft_f"],
    "all": list(PARLER_STYLE_PRESETS.keys()),
}

DEFAULT_PARLER_DESC = PARLER_STYLE_PRESETS["backchannel"]

WANDB_FIX = """\
IndicF5 failed because `wandb` is broken in this venv (common in venv-moshi).

Fix on the GPU box (prefer a dedicated TTS venv):

  python3.10 -m venv ~/tts-eval/.venv && source ~/tts-eval/.venv/bin/activate
  pip install -U pip
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install "git+https://github.com/ai4bharat/IndicF5.git"
  pip install "git+https://github.com/huggingface/parler-tts.git"
  pip install soundfile huggingface_hub transformers accelerate numpy
  pip install --force-reinstall --no-cache-dir "wandb>=0.19"

If you must stay in the current venv:

  pip install --force-reinstall --no-cache-dir "wandb>=0.19"
"""

README_MD = """---
license: apache-2.0
task_categories:
  - text-to-speech
language:
  - te
tags:
  - indic
  - telugu
  - indicf5
  - parler-tts
  - listen-test
private: true
---

# Telugu TTS A/B listen test

Side-by-side samples from **IndicF5** and **Indic Parler-TTS** for the same Telugu prompts.

## Layout

- `wav/NNN_parler_<style>.wav` — same text, different Parler styles (e.g. backchannel, happy, joyful)
- `wav/NNN_indicf5.wav` — IndicF5 (optional)
- `manifest.jsonl` — text, model, style, duration, sr

Open any `.wav` in the Hub file browser to listen.
"""


def _slug(i: int, model: str) -> str:
    return f"{i:03d}_{model}"


def parse_styles(raw: str | None, single: str | None) -> list[str]:
    """Comma-separated --parler-styles (or packs), or one --parler-style."""
    if raw:
        tokens = [s.strip() for s in raw.split(",") if s.strip()]
    elif single:
        tokens = [single]
    else:
        tokens = ["backchannel"]

    styles: list[str] = []
    for tok in tokens:
        if tok in PARLER_STYLE_PACKS:
            styles.extend(PARLER_STYLE_PACKS[tok])
        else:
            styles.append(tok)

    # preserve order, drop dupes
    seen: set[str] = set()
    ordered: list[str] = []
    for s in styles:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    unknown = [s for s in ordered if s not in PARLER_STYLE_PRESETS]
    if unknown:
        raise SystemExit(
            f"Unknown parler style(s): {unknown}.\n"
            f"Styles: {', '.join(PARLER_STYLE_PRESETS)}\n"
            f"Packs: {', '.join(PARLER_STYLE_PACKS)}"
        )
    return ordered


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.texts_file:
        texts.extend(
            ln.strip()
            for ln in args.texts_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
    if args.text:
        texts.extend(args.text)
    if not texts:
        texts = list(DEFAULT_TE_TEXTS)
    return texts


def ensure_ref_wav(ref_audio: Path, cache_dir: Path) -> Path:
    """IndicF5 is happiest with WAV; convert flac/ogg/mp3 → 24 kHz mono wav."""
    ref_audio = ref_audio.resolve()
    if not ref_audio.is_file():
        raise SystemExit(f"Missing ref audio: {ref_audio}")
    if ref_audio.suffix.lower() == ".wav":
        return ref_audio

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{ref_audio.stem}_24k.wav"
    if out.is_file() and out.stat().st_mtime >= ref_audio.stat().st_mtime:
        print(f"[IndicF5] using cached ref wav: {out}")
        return out

    print(f"[IndicF5] converting {ref_audio.name} → {out.name} (24 kHz mono)")
    audio, sr = sf.read(str(ref_audio), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        # lightweight linear resample (ok for short voice prompts)
        n_out = int(round(len(audio) * 24000.0 / float(sr)))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = 24000
    sf.write(out, audio, sr)
    return out


def _check_wandb() -> None:
    try:
        from wandb.proto.wandb_telemetry_pb2 import Imports  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — surface broken installs clearly
        raise SystemExit(f"{WANDB_FIX}\nUnderlying error: {exc}") from exc


def gen_indicf5(
    texts: list[str],
    out_dir: Path,
    ref_audio: Path,
    ref_text: str,
) -> list[dict]:
    _check_wandb()
    from transformers import AutoModel

    print("[IndicF5] loading ai4bharat/IndicF5 …")
    try:
        model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    except ImportError as exc:
        raise SystemExit(f"{WANDB_FIX}\nUnderlying error: {exc}") from exc

    rows: list[dict] = []
    for i, text in enumerate(texts):
        print(f"[IndicF5] {i + 1}/{len(texts)}: {text[:80]}")
        audio = model(text, ref_audio_path=str(ref_audio), ref_text=ref_text)
        if getattr(audio, "dtype", None) == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        path = out_dir / f"{_slug(i, 'indicf5')}.wav"
        sf.write(path, audio, 24000)
        rows.append(
            {
                "file": f"wav/{path.name}",
                "model": "indicf5",
                "text": text,
                "sr": 24000,
                "duration_s": round(float(len(audio)) / 24000.0, 3),
                "ref_audio": str(ref_audio),
            }
        )
    del model
    return rows


def gen_parler(
    texts: list[str],
    out_dir: Path,
    description: str,
    style_tag: str = "parler",
) -> list[dict]:
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    repo = "ai4bharat/indic-parler-tts"
    print(f"[Parler/{style_tag}] loading {repo} on {device} …")
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(device)
    tok = AutoTokenizer.from_pretrained(repo)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    desc = desc_tok(description, return_tensors="pt").to(device)

    rows: list[dict] = []
    for i, text in enumerate(texts):
        print(f"[Parler/{style_tag}] {i + 1}/{len(texts)}: {text[:80]}")
        prompt = tok(text, return_tensors="pt").to(device)
        gen = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        audio = gen.cpu().numpy().squeeze().astype(np.float32)
        sr = int(model.config.sampling_rate)
        path = out_dir / f"{_slug(i, f'parler_{style_tag}')}.wav"
        sf.write(path, audio, sr)
        rows.append(
            {
                "file": f"wav/{path.name}",
                "model": "indic-parler-tts",
                "style": style_tag,
                "text": text,
                "description": description,
                "sr": sr,
                "duration_s": round(float(len(audio)) / float(sr), 3),
            }
        )
    del model
    return rows


def write_bundle(out: Path, meta: list[dict]) -> None:
    (out / "README.md").write_text(README_MD, encoding="utf-8")
    man = out / "manifest.jsonl"
    with man.open("w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(meta)} rows → {man}")


def push_hf(local_dir: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Telugu TTS A/B {time.strftime('%Y-%m-%d %H:%M')}",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Telugu IndicF5 vs Parler listen test → Hugging Face"
    )
    p.add_argument(
        "--text",
        action="append",
        default=None,
        help="Telugu line (repeatable). If omitted, uses built-in samples.",
    )
    p.add_argument(
        "--texts-file",
        type=Path,
        default=None,
        help="UTF-8 file, one Telugu line per row (# comments ok).",
    )
    p.add_argument("--out", type=Path, default=Path("./te_tts_ab"))
    p.add_argument(
        "--ref-audio",
        type=Path,
        default=DEFAULT_REF_AUDIO if DEFAULT_REF_AUDIO.is_file() else None,
        help=f"IndicF5 reference audio (default: {DEFAULT_REF_AUDIO.name} if present).",
    )
    p.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Exact transcript of --ref-audio (required for IndicF5).",
    )
    p.add_argument(
        "--list-styles",
        action="store_true",
        help="Print available Parler styles and packs, then exit.",
    )
    p.add_argument(
        "--parler-style",
        choices=list(PARLER_STYLE_PRESETS.keys()),
        default=None,
        help="Single Parler caption preset. Prefer --parler-styles for A/B.",
    )
    p.add_argument(
        "--parler-styles",
        type=str,
        default=None,
        help=(
            "Comma-separated styles and/or packs. "
            "Packs: basic, emotions, positive, callcenter, ab, gender, all. "
            "Example: backchannel,happy,joyful  or  emotions"
        ),
    )
    p.add_argument(
        "--parler-desc",
        type=str,
        default=None,
        help="Custom Parler style caption (forces a single 'custom' run; ignores presets).",
    )
    p.add_argument(
        "--models",
        choices=["both", "indicf5", "parler"],
        default="parler",
    )
    p.add_argument("--hf-repo", type=str, default="BelluAi/te-tts-ab-listen")
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--public",
        action="store_true",
        help="Create/update dataset as public (default: private).",
    )
    args = p.parse_args()

    if args.list_styles:
        print("Styles:")
        for name, desc in PARLER_STYLE_PRESETS.items():
            print(f"  {name:12s}  {desc[:90]}…")
        print("\nPacks:")
        for name, members in PARLER_STYLE_PACKS.items():
            print(f"  {name:12s}  {','.join(members)}")
        return

    texts = load_texts(args)
    print(f"{len(texts)} prompt(s)")

    if args.parler_desc:
        style_runs: list[tuple[str, str]] = [("custom", args.parler_desc)]
    else:
        styles = parse_styles(args.parler_styles, args.parler_style)
        style_runs = [(s, PARLER_STYLE_PRESETS[s]) for s in styles]

    if args.models in ("both", "parler"):
        print(f"[Parler] styles: {[s for s, _ in style_runs]}")

    need_f5 = args.models in ("both", "indicf5")
    if need_f5:
        if not args.ref_audio:
            raise SystemExit(
                "IndicF5 needs --ref-audio (place audio.flac in repo root or pass path)."
            )
        if not args.ref_text:
            raise SystemExit(
                "IndicF5 needs --ref-text = exact words spoken in the reference audio."
            )

    args.out.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(exist_ok=True)

    meta: list[dict] = []
    if need_f5:
        ref_wav = ensure_ref_wav(args.ref_audio, args.out / "prompts")
        meta.extend(gen_indicf5(texts, wav_dir, ref_wav, args.ref_text))
    if args.models in ("both", "parler"):
        # Load once if multiple styles (reuse weights across captions)
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        repo = "ai4bharat/indic-parler-tts"
        print(f"[Parler] loading {repo} once on {device} …")
        model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(device)
        tok = AutoTokenizer.from_pretrained(repo)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        for style_tag, description in style_runs:
            print(f"[Parler/{style_tag}] {description[:100]}…")
            desc = desc_tok(description, return_tensors="pt").to(device)
            for i, text in enumerate(texts):
                print(f"[Parler/{style_tag}] {i + 1}/{len(texts)}: {text[:80]}")
                prompt = tok(text, return_tensors="pt").to(device)
                gen = model.generate(
                    input_ids=desc.input_ids,
                    attention_mask=desc.attention_mask,
                    prompt_input_ids=prompt.input_ids,
                    prompt_attention_mask=prompt.attention_mask,
                )
                audio = gen.cpu().numpy().squeeze().astype(np.float32)
                sr = int(model.config.sampling_rate)
                path = wav_dir / f"{_slug(i, f'parler_{style_tag}')}.wav"
                sf.write(path, audio, sr)
                meta.append(
                    {
                        "file": f"wav/{path.name}",
                        "model": "indic-parler-tts",
                        "style": style_tag,
                        "text": text,
                        "description": description,
                        "sr": sr,
                        "duration_s": round(float(len(audio)) / float(sr), 3),
                    }
                )
        del model

    write_bundle(args.out, meta)

    if args.no_push:
        print(f"Local only: {args.out.resolve()}")
        return

    url = push_hf(args.out, args.hf_repo, private=not args.public)
    print(f"Uploaded: {url}")
    print("Open Files → wav/ and compare same index across styles.")


if __name__ == "__main__":
    main()
