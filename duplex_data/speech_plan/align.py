"""CLI: audio + existing transcript → word timestamps via WhisperX (no ASR).

Usage:
  python -m duplex_data.speech_plan.align \\
    --audio clip.wav --transcript clip.txt --lang te --out ./alignments/clip.json

  # Directory mode: paired *.wav + *.txt (same stem)
  python -m duplex_data.speech_plan.align \\
    --audio ./clips --transcript ./clips --lang hi --out ./alignments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from duplex_data.asr import (
    AlignmentError,
    align_words_whisperx,
    language_id_for_nemo,
    whisperx_align_model_for_language,
)

logger = logging.getLogger("duplex_data.speech_plan.align")

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
TRANSCRIPT_EXTS = {".txt", ".json"}


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def slice_segment(
    mono: np.ndarray,
    sr: int,
    start: float | None,
    end: float | None,
) -> np.ndarray:
    if start is None and end is None:
        return mono
    n = len(mono)
    i0 = 0 if start is None else max(0, int(round(float(start) * sr)))
    i1 = n if end is None else min(n, int(round(float(end) * sr)))
    if i1 <= i0:
        raise ValueError(f"Empty segment: start={start} end={end} (sr={sr})")
    return mono[i0:i1]


def load_transcript(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("text", "transcript", "transcription"):
                if key in obj and obj[key] is not None:
                    return str(obj[key]).strip()
            raise ValueError(
                f"JSON transcript {path} must contain a 'text' (or transcript) field"
            )
        if isinstance(obj, str):
            return obj.strip()
        raise ValueError(f"Unsupported JSON transcript shape in {path}")
    return raw.strip()


def find_transcript_for_audio(audio: Path, transcript_root: Path) -> Path | None:
    """Resolve paired transcript: same stem under transcript_root (file or dir)."""
    if transcript_root.is_file():
        return transcript_root
    for ext in (".txt", ".json"):
        cand = transcript_root / f"{audio.stem}{ext}"
        if cand.is_file():
            return cand
    return None


def collect_pairs(
    audio_path: Path,
    transcript_path: Path,
) -> list[tuple[Path, Path]]:
    """Return (audio, transcript) pairs."""
    if audio_path.is_file():
        if transcript_path.is_file():
            return [(audio_path, transcript_path)]
        tr = find_transcript_for_audio(audio_path, transcript_path)
        if tr is None:
            raise FileNotFoundError(
                f"No transcript for {audio_path.name} under {transcript_path}"
            )
        return [(audio_path, tr)]

    if not audio_path.is_dir():
        raise FileNotFoundError(f"Audio path not found: {audio_path}")

    pairs: list[tuple[Path, Path]] = []
    for p in sorted(audio_path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        if transcript_path.is_file():
            raise ValueError(
                "Directory audio mode requires --transcript as a directory of paired files"
            )
        # Prefer transcript next to audio, then under transcript_path with same stem
        local_txt = p.with_suffix(".txt")
        local_json = p.with_suffix(".json")
        if local_txt.is_file():
            pairs.append((p, local_txt))
            continue
        if local_json.is_file():
            pairs.append((p, local_json))
            continue
        tr = find_transcript_for_audio(p, transcript_path)
        if tr is None:
            # try relative path under transcript_path
            try:
                rel = p.relative_to(audio_path)
                for ext in (".txt", ".json"):
                    cand = transcript_path / rel.with_suffix(ext)
                    if cand.is_file():
                        tr = cand
                        break
            except ValueError:
                pass
        if tr is None:
            logger.warning("Skipping %s (no paired transcript)", p)
            continue
        pairs.append((p, tr))
    return pairs


def words_to_jsonable(words: list) -> list[list]:
    return [[w, [float(s), float(e)]] for w, (s, e) in words]


def align_one(
    audio_path: Path,
    transcript_path: Path,
    *,
    language: str,
    device: str = "cpu",
    align_model: str | None = None,
    segment_start: float | None = None,
    segment_end: float | None = None,
) -> dict:
    mono, sr = load_mono(audio_path)
    mono = slice_segment(mono, sr, segment_start, segment_end)
    text = load_transcript(transcript_path)
    if not text:
        raise AlignmentError(f"Empty transcript: {transcript_path}")

    lang = language_id_for_nemo(language)
    spans, align_repo = align_words_whisperx(
        mono,
        sr,
        text,
        lang,
        device=device,
        align_model=align_model,
    )
    return {
        "audio_path": str(audio_path),
        "language": lang,
        "transcript": text,
        "align_model": align_repo,
        "words": words_to_jsonable(spans),
        "segment_start": segment_start,
        "segment_end": segment_end,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forced-align an existing transcript to audio with WhisperX (no ASR)."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Audio file or directory")
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Transcript .txt/.json, or directory of paired files (same stem)",
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code (hi/te/ta/kn/ml/ur/mr/gu/bn/pa/or/…)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON file (single clip) or directory (batch)",
    )
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument(
        "--align-model",
        default=None,
        help="Optional HF Wav2Vec2ForCTC id (overrides language map)",
    )
    parser.add_argument(
        "--segment-start",
        type=float,
        default=None,
        help="Optional clip start in seconds (speaker segment)",
    )
    parser.add_argument(
        "--segment-end",
        type=float,
        default=None,
        help="Optional clip end in seconds (speaker segment)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.audio.exists():
        logger.error("Audio not found: %s", args.audio)
        return 1
    if not args.transcript.exists():
        logger.error("Transcript not found: %s", args.transcript)
        return 1

    # Validate map early unless override provided
    if not (args.align_model or "").strip():
        try:
            whisperx_align_model_for_language(args.lang)
        except AlignmentError as exc:
            logger.error("%s", exc)
            return 1

    try:
        pairs = collect_pairs(args.audio, args.transcript)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not pairs:
        logger.error("No audio/transcript pairs found")
        return 1

    batch = len(pairs) > 1 or args.audio.is_dir()
    if batch:
        args.out.mkdir(parents=True, exist_ok=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    for audio_p, tr_p in pairs:
        try:
            payload = align_one(
                audio_p,
                tr_p,
                language=args.lang,
                device=args.device,
                align_model=args.align_model,
                segment_start=args.segment_start,
                segment_end=args.segment_end,
            )
            if batch:
                out_path = args.out / f"{audio_p.stem}.json"
            else:
                out_path = args.out
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "Wrote %s (%d words, model=%s)",
                out_path,
                len(payload["words"]),
                payload["align_model"],
            )
            ok += 1
        except Exception as exc:
            logger.exception("Failed %s: %s", audio_p, exc)

    logger.info("Done: %d / %d", ok, len(pairs))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
