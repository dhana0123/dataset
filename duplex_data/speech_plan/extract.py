"""CLI: raw audio → draft SpeechPlan JSON (auto events + empty nonverbal).

Usage:
  python -m duplex_data.speech_plan.extract path/to/clip.wav --out ./speech_plans --lang auto
  python -m duplex_data.speech_plan.extract path/to/wav_dir --out ./speech_plans --lang te
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from duplex_data.asr import language_id_for_nemo, transcribe_agent_words
from duplex_data.speech_plan.acoustics import extract_acoustics
from duplex_data.speech_plan.derive import build_lanes
from duplex_data.speech_plan.schema import GlobalState, SpeechPlan

logger = logging.getLogger("duplex_data.speech_plan.extract")

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load mono float32; if stereo, take left channel (agent convention)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data[:, 0]
    return mono, int(sr)


def process_file(
    audio_path: Path,
    out_dir: Path,
    *,
    language: str = "auto",
    asr_backend: str = "whisper",
    device: str = "cpu",
    asr_model: str | None = None,
    copy_audio: bool = True,
) -> Path:
    mono, sr = load_mono(audio_path)
    duration = float(len(mono) / sr) if sr > 0 else 0.0

    lang = language
    if lang in {"", "auto", "none"}:
        # Whisper can detect; pass a sensible default for Indic backends.
        lang_for_asr = "en" if asr_backend == "whisper" else "hi"
    else:
        lang_for_asr = language_id_for_nemo(lang)

    logger.info("ASR (%s) on %s …", asr_backend, audio_path.name)
    asr = transcribe_agent_words(
        mono,
        sr,
        language=lang_for_asr,
        backend=asr_backend,
        asr_model=asr_model,
        device=device,
    )
    resolved_lang = asr.provenance.language or lang_for_asr

    logger.info("Acoustics on %s …", audio_path.name)
    tracks = extract_acoustics(mono, sr)
    lanes = build_lanes(asr.words, tracks, duration)

    clip_dir = out_dir / audio_path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    if copy_audio:
        dest_audio = clip_dir / f"{audio_path.stem}.wav"
        if audio_path.suffix.lower() == ".wav" and audio_path.resolve() != dest_audio.resolve():
            shutil.copy2(audio_path, dest_audio)
        else:
            # Normalize to wav mono for the annotator
            sf.write(str(dest_audio), mono, sr, subtype="PCM_16")
        rel_audio = dest_audio.name
    else:
        rel_audio = str(audio_path.resolve())
        dest_audio = audio_path

    plan = SpeechPlan(
        audio_path=rel_audio,
        duration=duration,
        language=resolved_lang if language not in {"", "auto"} else resolved_lang,
        global_state=GlobalState(value=None, source="human"),
        nonverbal=[],
        lanes=lanes,
        tracks=tracks.to_dict(),
        asr={
            "transcript": asr.provenance.transcript_text,
            "backend": asr.provenance.transcript_backend,
            "model": asr.provenance.transcript_model,
            "timer_backend": asr.provenance.timer_backend,
            "timer_model": asr.provenance.timer_model,
            "language": asr.provenance.language,
            "words": len(asr.words),
        },
    )
    out_json = clip_dir / "speech_plan.json"
    plan.save(out_json)
    logger.info("Wrote %s (%d words)", out_json, len(asr.words))
    return out_json


def collect_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract draft speech-plan events from audio (ASR + Praat/librosa)."
    )
    parser.add_argument("input", type=Path, help="Audio file or directory of clips")
    parser.add_argument("--out", type=Path, required=True, help="Output root for speech_plans/")
    parser.add_argument("--lang", default="auto", help="Language code (hi/te/ta/kn/en) or auto")
    parser.add_argument(
        "--asr-backend",
        default="whisper",
        choices=["whisper", "indic-conformer"],
        help="ASR backend (default: whisper)",
    )
    parser.add_argument("--asr-model", default=None, help="Optional model name / HF repo")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument(
        "--no-copy-audio",
        action="store_true",
        help="Store absolute audio path instead of copying into the plan folder",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 1

    files = collect_audio(args.input)
    if not files:
        logger.error("No audio files under %s", args.input)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in files:
        try:
            process_file(
                f,
                args.out,
                language=args.lang,
                asr_backend=args.asr_backend,
                device=args.device,
                asr_model=args.asr_model,
                copy_audio=not args.no_copy_audio,
            )
            ok += 1
        except Exception as exc:
            logger.exception("Failed %s: %s", f, exc)
    logger.info("Done: %d / %d clips", ok, len(files))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
