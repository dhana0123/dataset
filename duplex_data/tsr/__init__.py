"""Temporal Speech Representation package — duplex I/O control format."""

from .schema import TSREvent, TemporalSpeechRep, sec_to_ms, ms_to_sec, TSR_KINDS
from .serialize import tsr_to_text, text_to_tsr, save_text
from .to_tts import tsr_to_parler_inputs
from .llm import TemplateLLM, generate_tsr

__all__ = [
    "TSREvent",
    "TemporalSpeechRep",
    "sec_to_ms",
    "ms_to_sec",
    "TSR_KINDS",
    "tsr_to_text",
    "text_to_tsr",
    "save_text",
    "tsr_to_parler_inputs",
    "TemplateLLM",
    "generate_tsr",
    "audio_to_tsr",
    "words_tracks_to_tsr",
    "run_pipeline",
]


def __getattr__(name: str):
    # Lazy imports: avoid hard deps (soundfile/parler) at package import time.
    if name in {"audio_to_tsr", "words_tracks_to_tsr"}:
        from . import from_audio

        return getattr(from_audio, name)
    if name == "run_pipeline":
        from .pipeline import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
