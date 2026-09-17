"""Canonical Temporal Speech Representation (TSR).

Same schema for input (what the user said/did) and output (what TTS should produce).
Timestamps are integers in milliseconds. Events are time-ordered.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Direction = Literal["input", "output"]

TSR_KINDS = (
    "word",
    "pause",
    "hesitation",
    "emphasis",
    "pitch",
    "energy",
    "rate",
    "laugh",
    "breath",
    "sigh",
    "backchannel",
    "interruption",
    "overlap",
    "boundary",
)

TSR_SOURCES = ("asr", "auto", "human", "llm")


def sec_to_ms(sec: float) -> int:
    return int(round(float(sec) * 1000.0))


def ms_to_sec(ms: int) -> float:
    return float(ms) / 1000.0


@dataclass
class TSREvent:
    start_ms: int
    end_ms: int
    kind: str
    text: str | None = None
    value: str | None = None
    source: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TSREvent:
        return cls(
            start_ms=int(d["start_ms"]),
            end_ms=int(d["end_ms"]),
            kind=str(d["kind"]),
            text=d.get("text"),
            value=None if d.get("value") is None else str(d["value"]),
            source=str(d.get("source") or "auto"),
        )


@dataclass
class TemporalSpeechRep:
    direction: Direction
    language: str
    transcript: str
    duration_ms: int
    events: list[TSREvent] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def sort_events(self) -> None:
        self.events.sort(key=lambda e: (e.start_ms, e.end_ms, e.kind, e.text or ""))

    def words(self) -> list[TSREvent]:
        return [e for e in self.events if e.kind == "word" and e.text]

    def transcript_from_words(self) -> str:
        return " ".join(e.text for e in self.words() if e.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "language": self.language,
            "transcript": self.transcript,
            "duration_ms": int(self.duration_ms),
            "events": [e.to_dict() for e in self.events],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalSpeechRep:
        events = [TSREvent.from_dict(e) for e in (d.get("events") or [])]
        rep = cls(
            direction=d.get("direction", "input"),  # type: ignore[arg-type]
            language=str(d.get("language") or "auto"),
            transcript=str(d.get("transcript") or ""),
            duration_ms=int(d.get("duration_ms") or 0),
            events=events,
            meta=dict(d.get("meta") or {}),
        )
        rep.sort_events()
        return rep

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TemporalSpeechRep:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
