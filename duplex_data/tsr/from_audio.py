"""audio → Temporal Speech Representation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from duplex_data.asr import (
    WordSpan,
    align_words_whisperx,
    language_id_for_nemo,
    transcribe_agent_words,
)
from duplex_data.speech_plan.acoustics import AcousticTracks, extract_acoustics
from duplex_data.speech_plan.derive import build_lanes
from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep, sec_to_ms

logger = logging.getLogger("duplex_data.tsr.from_audio")

HESITATION_GAP_SEC = 0.30
BACKCHANNEL_WORDS = {
    "yeah",
    "yea",
    "yes",
    "yep",
    "hmm",
    "hm",
    "mm",
    "mhm",
    "uh",
    "um",
    "aha",
    "ok",
    "okay",
    "हम्म",
    "हाँ",
    "हां",
    "హ్మ్",
    "అవును",
    "ஆம்",
    "ம்ம்",
}


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def _load_transcript_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("text", "transcript", "transcription"):
                if key in obj and obj[key] is not None:
                    return str(obj[key]).strip()
            raise ValueError(f"JSON transcript {path} needs a 'text' field")
        if isinstance(obj, str):
            return obj.strip()
        raise ValueError(f"Unsupported JSON transcript in {path}")
    return raw.strip()


def _is_backchannel(word: str) -> bool:
    w = word.strip().lower().strip(".,!?…")
    return w in BACKCHANNEL_WORDS


def words_tracks_to_tsr(
    words: Sequence[WordSpan],
    tracks: AcousticTracks,
    *,
    duration_sec: float,
    language: str,
    direction: str = "input",
    meta: dict | None = None,
) -> TemporalSpeechRep:
    """Build TSR from word spans + acoustic tracks (deterministic heuristics)."""
    lanes = build_lanes(words, tracks, duration_sec)
    events: list[TSREvent] = []

    for ev in lanes.get("content") or []:
        text = str(ev.value) if ev.value is not None else ""
        start_ms = sec_to_ms(ev.start)
        end_ms = sec_to_ms(ev.end)
        events.append(
            TSREvent(
                start_ms=start_ms,
                end_ms=end_ms,
                kind="word",
                text=text,
                value=None,
                source=str(ev.source),
            )
        )
        if text and _is_backchannel(text):
            events.append(
                TSREvent(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    kind="backchannel",
                    text=text,
                    value=None,
                    source="auto",
                )
            )

    for ev in lanes.get("pause") or []:
        start_ms = sec_to_ms(ev.start)
        end_ms = sec_to_ms(ev.end)
        gap_sec = float(ev.end) - float(ev.start)
        events.append(
            TSREvent(
                start_ms=start_ms,
                end_ms=end_ms,
                kind="pause",
                text=None,
                value=str(ev.value) if ev.value is not None else None,
                source="auto",
            )
        )
        if gap_sec >= HESITATION_GAP_SEC:
            events.append(
                TSREvent(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    kind="hesitation",
                    text=None,
                    value=None,
                    source="auto",
                )
            )

    kind_map = {
        "pitch": "pitch",
        "energy": "energy",
        "rate": "rate",
        "emphasis": "emphasis",
        "boundary": "boundary",
    }
    for lane_key, kind in kind_map.items():
        for ev in lanes.get(lane_key) or []:
            events.append(
                TSREvent(
                    start_ms=sec_to_ms(ev.start),
                    end_ms=sec_to_ms(ev.end),
                    kind=kind,
                    text=None,
                    value=str(ev.value) if ev.value is not None else None,
                    source=str(ev.source),
                )
            )

    transcript = " ".join(w for w, _ in words)
    duration_ms = sec_to_ms(duration_sec)
    if not any(e.kind == "boundary" for e in events) and words:
        last_end = max(sec_to_ms(e) for _, (_, e) in words)
        events.append(
            TSREvent(
                start_ms=last_end,
                end_ms=last_end,
                kind="boundary",
                text=None,
                value="completing",
                source="auto",
            )
        )

    rep = TemporalSpeechRep(
        direction=direction,  # type: ignore[arg-type]
        language=language,
        transcript=transcript,
        duration_ms=duration_ms,
        events=events,
        meta=dict(meta or {}),
    )
    rep.sort_events()
    return rep


def audio_to_tsr(
    audio_path: Path | str,
    *,
    language: str = "en",
    asr_backend: str = "whisper",
    device: str = "cpu",
    asr_model: str | None = None,
    transcript_path: Path | str | None = None,
    align_model: str | None = None,
) -> TemporalSpeechRep:
    """Run ASR (or align provided transcript) + acoustics → input TSR."""
    audio_path = Path(audio_path)
    mono, sr = _load_mono(audio_path)
    duration_sec = float(len(mono) / sr) if sr > 0 else 0.0
    lang = language_id_for_nemo(language) if language not in {"", "auto"} else language

    if transcript_path is not None:
        if lang in {"", "auto", "none"}:
            raise ValueError("Align-only path requires an explicit --lang")
        text = _load_transcript_text(Path(transcript_path))
        words, align_repo = align_words_whisperx(
            mono, sr, text, lang, device=device, align_model=align_model
        )
        meta = {
            "audio_path": str(audio_path),
            "transcript_backend": "provided",
            "timer_backend": "whisperx",
            "timer_model": align_repo,
            "sample_rate": sr,
        }
        resolved_lang = lang
        transcript = text
    else:
        lang_for_asr = "en" if lang in {"", "auto", "none"} else lang
        asr = transcribe_agent_words(
            mono,
            sr,
            language=lang_for_asr,
            backend=asr_backend,
            asr_model=asr_model,
            device=device,
        )
        words = asr.words
        resolved_lang = asr.provenance.language or lang_for_asr
        transcript = asr.provenance.transcript_text or " ".join(w for w, _ in words)
        meta = {
            "audio_path": str(audio_path),
            "transcript_backend": asr.provenance.transcript_backend,
            "transcript_model": asr.provenance.transcript_model,
            "timer_backend": asr.provenance.timer_backend,
            "timer_model": asr.provenance.timer_model,
            "sample_rate": sr,
        }

    logger.info("Acoustics for TSR on %s …", audio_path.name)
    tracks = extract_acoustics(mono, sr)
    rep = words_tracks_to_tsr(
        words,
        tracks,
        duration_sec=duration_sec,
        language=resolved_lang,
        direction="input",
        meta=meta,
    )
    if transcript and not rep.transcript:
        rep.transcript = transcript
    elif transcript:
        rep.transcript = transcript
    return rep
