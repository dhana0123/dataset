"""LLM adapters: Input TSR → Output TSR (same schema)."""

from __future__ import annotations

from typing import Protocol

from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep
from duplex_data.tsr.serialize import text_to_tsr, tsr_to_text


class TSRGenerator(Protocol):
    def generate_tsr(self, input_tsr: TemporalSpeechRep) -> TemporalSpeechRep: ...


class TemplateLLM:
    """Offline stub: polite English reply with a fixed temporal skeleton."""

    REPLY = "That's great to hear."

    def generate_tsr(self, input_tsr: TemporalSpeechRep) -> TemporalSpeechRep:
        # Fixed timing template matching the plan example shape
        words = [
            ("That's", 0, 250),
            ("great", 250, 500),
            ("to", 600, 750),
            ("hear", 750, 1000),
        ]
        events: list[TSREvent] = []
        for text, s, e in words:
            events.append(
                TSREvent(
                    start_ms=s,
                    end_ms=e,
                    kind="word",
                    text=text,
                    source="llm",
                )
            )
        events.append(
            TSREvent(
                start_ms=250,
                end_ms=500,
                kind="emphasis",
                text="great",
                value="strong",
                source="llm",
            )
        )
        events.append(
            TSREvent(start_ms=500, end_ms=600, kind="pause", source="llm")
        )
        events.append(
            TSREvent(
                start_ms=750,
                end_ms=1000,
                kind="pitch",
                text="hear",
                value="warm/falling",
                source="llm",
            )
        )
        events.append(
            TSREvent(
                start_ms=1000,
                end_ms=1000,
                kind="boundary",
                value="completing",
                source="llm",
            )
        )
        out = TemporalSpeechRep(
            direction="output",
            language=input_tsr.language,
            transcript=self.REPLY,
            duration_ms=1000,
            events=events,
            meta={
                "llm": "template",
                "input_transcript": input_tsr.transcript,
                "input_text_preview": tsr_to_text(input_tsr)[:500],
            },
        )
        out.sort_events()
        return out


def generate_tsr(
    input_tsr: TemporalSpeechRep,
    *,
    backend: str = "template",
) -> TemporalSpeechRep:
    """Dispatch LLM backend. v1: template only."""
    backend = (backend or "template").lower()
    if backend in {"template", "stub", "offline"}:
        return TemplateLLM().generate_tsr(input_tsr)
    if backend == "parse_echo":
        # Round-trip serialize→parse as smoke test of format
        return text_to_tsr(
            tsr_to_text(input_tsr),
            direction="output",
            language=input_tsr.language,
            meta={"llm": "parse_echo"},
        )
    raise ValueError(f"Unknown LLM backend {backend!r}; use template")
