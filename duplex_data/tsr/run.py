"""CLI: Audio → TSR → LLM → TSR → TTS → Audio.

Usage:
  python -m duplex_data.tsr.run --audio user.wav --lang te --out ./tsr_runs/clip1
  python -m duplex_data.tsr.run --audio user.wav --transcript user.txt --lang te \\
      --out ./tsr_runs/clip1 --skip-tts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from duplex_data.tsr.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Temporal Speech Representation E2E: audio → TSR → LLM → TSR → TTS"
    )
    parser.add_argument("--audio", type=Path, required=True, help="Input user audio")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--lang", default="en", help="Language code (required with --transcript)")
    parser.add_argument(
        "--asr-backend",
        default="whisper",
        choices=["whisper", "indic-conformer"],
        help="ASR backend when no --transcript",
    )
    parser.add_argument("--asr-model", default=None)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Existing transcript (.txt/.json); skip ASR, WhisperX align only",
    )
    parser.add_argument("--align-model", default=None, help="HF aligner override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--llm",
        default="template",
        choices=["template", "parse_echo"],
        help="LLM backend (default: offline template)",
    )
    parser.add_argument(
        "--speaker-style",
        default="phone",
        help="Parler agent style key (leela/happy/phone/formal)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Stop after writing input/output TSR (no Parler load)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.audio.exists():
        logging.error("Audio not found: %s", args.audio)
        return 1
    if args.transcript is not None and not args.transcript.exists():
        logging.error("Transcript not found: %s", args.transcript)
        return 1

    try:
        paths = run(
            args.audio,
            args.out,
            language=args.lang,
            asr_backend=args.asr_backend,
            device=args.device,
            asr_model=args.asr_model,
            transcript_path=args.transcript,
            align_model=args.align_model,
            llm_backend=args.llm,
            speaker_style=args.speaker_style,
            skip_tts=args.skip_tts,
        )
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1

    logging.info("Done: %s", paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
