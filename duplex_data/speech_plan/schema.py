"""SpeechPlan / Event JSON schema (research, not frozen).

Lanes overlap in time — not a single serialized event chain.
NONVERBAL is always empty in v1; GLOBAL STATE is clip-level human label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
        # Ensure all lane keys exist; nonverbal lane + top-level list stay empty in v1.
        for key in LANE_KEYS:
            self.lanes.setdefault(key, [])
        self.nonverbal = []
        self.lanes["nonverbal"] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "duration": self.duration,
            "language": self.language,
            "global_state": self.global_state.to_dict(),
            "nonverbal": [],
            "lanes": {
                k: [e.to_dict() for e in self.lanes.get(k, [])]
                for k in LANE_KEYS
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
        return cls(
            audio_path=str(d["audio_path"]),
            duration=float(d["duration"]),
            language=str(d.get("language", "auto")),
            global_state=GlobalState.from_dict(d.get("global_state")),
            nonverbal=[],
            lanes=lanes,
            tracks=dict(d.get("tracks") or {}),
            asr=dict(d.get("asr") or {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SpeechPlan:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
