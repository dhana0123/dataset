#!/usr/bin/env python3
"""Telugu listen-test: IndicF5 + Indic Parler-TTS → local wavs → Hugging Face dataset.

Run in a dedicated TTS venv (not moshi / duplex-data packer venv).

  python scripts/compare_te_tts.py \\
    --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \\
    --ref-audio prompts/te_ref.wav \\
    --ref-text "..." \\
    --hf-repo BelluAi/te-tts-ab-listen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_TE_TEXTS = [
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "అవును, సరే. నేను మీ ఖాతా వివరాలు చూస్తున్నాను.",
    "క్షమించండి, మళ్లీ చెప్పగలరా?",
    "అవును అవును, అర్థమైంది.",
]

# Telugu recommended speakers: Prakash, Lalitha (see indic-parler-tts card)
DEFAULT_PARLER_DESC = (
    "Prakash's voice is clear and slightly expressive, speaking Telugu at a "
    "moderate pace with very high quality audio and almost no background noise."
)

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

- `wav/NNN_indicf5.wav` — IndicF5
- `wav/NNN_parler.wav` — Indic Parler-TTS
- `manifest.jsonl` — text, model, duration, sr

Open any `.wav` in the Hub file browser to listen.
"""


def _slug(i: int, model: str) -> str:
    return f"{i:03d}_{model}"


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


def gen_indicf5(
    texts: list[str],
    out_dir: Path,
    ref_audio: Path,
    ref_text: str,
) -> list[dict]:
    from transformers import AutoModel

    print("[IndicF5] loading ai4bharat/IndicF5 …")
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
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


def gen_parler(texts: list[str], out_dir: Path, description: str) -> list[dict]:
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    repo = "ai4bharat/indic-parler-tts"
    print(f"[Parler] loading {repo} on {device} …")
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(device)
    tok = AutoTokenizer.from_pretrained(repo)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    desc = desc_tok(description, return_tensors="pt").to(device)

    rows: list[dict] = []
    for i, text in enumerate(texts):
        print(f"[Parler] {i + 1}/{len(texts)}: {text[:80]}")
        prompt = tok(text, return_tensors="pt").to(device)
        gen = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        audio = gen.cpu().numpy().squeeze().astype(np.float32)
        sr = int(model.config.sampling_rate)
        path = out_dir / f"{_slug(i, 'parler')}.wav"
        sf.write(path, audio, sr)
        rows.append(
            {
                "file": f"wav/{path.name}",
                "model": "indic-parler-tts",
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
        default=None,
        help="IndicF5 reference wav (required unless --models parler).",
    )
    p.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Exact transcript of --ref-audio (required for IndicF5).",
    )
    p.add_argument("--parler-desc", type=str, default=DEFAULT_PARLER_DESC)
    p.add_argument(
        "--models",
        choices=["both", "indicf5", "parler"],
        default="both",
    )
    p.add_argument("--hf-repo", type=str, default="BelluAi/te-tts-ab-listen")
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--public",
        action="store_true",
        help="Create/update dataset as public (default: private).",
    )
    args = p.parse_args()

    texts = load_texts(args)
    print(f"{len(texts)} prompt(s)")

    if args.models in ("both", "indicf5"):
        if not args.ref_audio or not args.ref_text:
            raise SystemExit(
                "IndicF5 needs --ref-audio and --ref-text "
                "(or use --models parler)."
            )
        if not args.ref_audio.is_file():
            raise SystemExit(f"Missing ref audio: {args.ref_audio}")

    args.out.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(exist_ok=True)

    meta: list[dict] = []
    if args.models in ("both", "indicf5"):
        meta.extend(gen_indicf5(texts, wav_dir, args.ref_audio, args.ref_text))
    if args.models in ("both", "parler"):
        meta.extend(gen_parler(texts, wav_dir, args.parler_desc))

    write_bundle(args.out, meta)

    if args.no_push:
        print(f"Local only: {args.out.resolve()}")
        return

    url = push_hf(args.out, args.hf_repo, private=not args.public)
    print(f"Uploaded: {url}")
    print("Open Files → wav/ and play each pair (same index = same text).")


if __name__ == "__main__":
    main()
