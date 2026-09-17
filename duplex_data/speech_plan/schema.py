"""SpeechPlan / Event JSON schema (research, not frozen).

Lanes overlap in time — not a single serialized event chain.
NONVERBAL uses the packaged event vocabulary (human-annotated).
Indic nonverbal values are native-script transcripts (e.g. हम्म, హ్మ్).
GLOBAL STATE is clip-level human label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from duplex_data.speech_plan.vocab import list_nonverbal_event_ids

EventSource = Literal["asr", "auto", "human"]
LaneKey = Literal[
    "content",
    "pitch",
    "energy",
    "rate",
    "emphasis",
    "pause",
    "boundary",
    "nonverbal",
]

LANE_KEYS: tuple[LaneKey, ...] = (
    "content",
    "pitch",
    "energy",
    "rate",
    "emphasis",
    "pause",
    "boundary",
    "nonverbal",
)

GLOBAL_STATE_VALUES = (
    "calm",
    "serious",
    "excited",
    "reassuring",
    "uncertain",
    "urgent",
    "sad",
    "disagreement",
    "other",
)

PITCH_VALUES = ("rising", "falling", "stable")
ENERGY_VALUES = ("softer", "stronger")
RATE_VALUES = ("faster", "slower")
EMPHASIS_VALUES = ("strong", "weak")
BOUNDARY_VALUES = ("continuing", "completing", "questioning")


def nonverbal_values() -> tuple[str, ...]:
    return list_nonverbal_event_ids()


NONVERBAL_VALUES: tuple[str, ...] = list_nonverbal_event_ids()


@dataclass
class Event:
    type: str
    start: float
    end: float
    value: str | float | None = None
    intensity: float | None = None
    source: EventSource = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        return cls(
            type=str(d["type"]),
            start=float(d["start"]),
            end=float(d["end"]),
            value=d.get("value"),
            intensity=d.get("intensity"),
            source=d.get("source", "auto"),  # type: ignore[arg-type]
        )


@dataclass
class GlobalState:
    value: str | None = None
    source: EventSource = "human"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> GlobalState:
        if not d:
            return cls()
        return cls(value=d.get("value"), source=d.get("source", "human"))  # type: ignore[arg-type]


def _empty_lanes() -> dict[str, list[Event]]:
    return {k: [] for k in LANE_KEYS}


@dataclass
class SpeechPlan:
    audio_path: str
    duration: float
    language: str = "auto"
    global_state: GlobalState = field(default_factory=GlobalState)
    nonverbal: list[Any] = field(default_factory=list)
    lanes: dict[str, list[Event]] = field(default_factory=_empty_lanes)
    tracks: dict[str, Any] = field(default_factory=dict)
    asr: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in LANE_KEYS:
            self.lanes.setdefault(key, [])
        if self.lanes.get("nonverbal"):
            self.nonverbal = [
                e.to_dict() if isinstance(e, Event) else e for e in self.lanes["nonverbal"]
            ]
        elif self.nonverbal:
            self.lanes["nonverbal"] = [
                e if isinstance(e, Event) else Event.from_dict(e) for e in self.nonverbal
            ]

    def to_dict(self) -> dict[str, Any]:
        nonverbal_events = [e.to_dict() for e in self.lanes.get("nonverbal", [])]
        return {
            "audio_path": self.audio_path,
            "duration": self.duration,
            "language": self.language,
            "global_state": self.global_state.to_dict(),
            "nonverbal": nonverbal_events,
            "lanes": {
                k: [e.to_dict() for e in self.lanes.get(k, [])] for k in LANE_KEYS
            },
            "tracks": self.tracks,
            "asr": self.asr,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpeechPlan:
        lanes_raw = d.get("lanes") or {}
        lanes: dict[str, list[Event]] = {}
        for key in LANE_KEYS:
            lanes[key] = [Event.from_dict(e) for e in lanes_raw.get(key, [])]
        if not lanes.get("nonverbal") and d.get("nonverbal"):
            lanes["nonverbal"] = [
                e if isinstance(e, Event) else Event.from_dict(e) for e in d["nonverbal"]
            ]
        return cls(
            audio_path=str(d["audio_path"]),
            duration=float(d["duration"]),
            language=str(d.get("language", "auto")),
            global_state=GlobalState.from_dict(d.get("global_state")),
            nonverbal=list(d.get("nonverbal") or []),
            lanes=lanes,
            tracks=dict(d.get("tracks") or {}),
            asr=dict(d.get("asr") or {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SpeechPlan:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
