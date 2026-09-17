"""Optional bridges between SpeechPlan (seconds/lanes) and TSR (ms/events)."""

from __future__ import annotations

from duplex_data.speech_plan.schema import Event, GlobalState, SpeechPlan
from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep, ms_to_sec, sec_to_ms

_LANE_FOR_KIND = {
    "word": "content",
    "pause": "pause",
    "emphasis": "emphasis",
    "pitch": "pitch",
    "energy": "energy",
    "rate": "rate",
    "boundary": "boundary",
    "laugh": "nonverbal",
    "breath": "nonverbal",
    "sigh": "nonverbal",
    "backchannel": "nonverbal",
    "hesitation": "nonverbal",
    "interruption": "nonverbal",
    "overlap": "nonverbal",
}


def speech_plan_to_tsr(
    plan: SpeechPlan,
    *,
    direction: str = "input",
) -> TemporalSpeechRep:
    events: list[TSREvent] = []
    for lane, items in (plan.lanes or {}).items():
        for ev in items:
            kind = "word" if lane == "content" else lane
            if lane == "nonverbal":
                kind = str(ev.type or "backchannel").lower()
            events.append(
                TSREvent(
                    start_ms=sec_to_ms(ev.start),
                    end_ms=sec_to_ms(ev.end),
                    kind=kind if kind != "content" else "word",
                    text=str(ev.value) if lane == "content" and ev.value is not None else None,
                    value=(
                        None
                        if lane == "content"
                        else (str(ev.value) if ev.value is not None else None)
                    ),
                    source=str(ev.source),
                )
            )
    transcript = " ".join(
        e.text for e in events if e.kind == "word" and e.text
    )
    rep = TemporalSpeechRep(
        direction=direction,  # type: ignore[arg-type]
        language=plan.language,
        transcript=transcript,
        duration_ms=sec_to_ms(plan.duration),
        events=events,
        meta={"from": "speech_plan", "audio_path": plan.audio_path, **(plan.asr or {})},
    )
    rep.sort_events()
    return rep


def tsr_to_speech_plan(rep: TemporalSpeechRep) -> SpeechPlan:
    lanes: dict[str, list[Event]] = {k: [] for k in (
        "content", "pitch", "energy", "rate", "emphasis", "pause", "boundary", "nonverbal"
    )}
    for ev in rep.events:
        lane = _LANE_FOR_KIND.get(ev.kind, "nonverbal")
        if ev.kind == "word":
            lanes["content"].append(
                Event(
                    type="WORD",
                    start=ms_to_sec(ev.start_ms),
                    end=ms_to_sec(ev.end_ms),
                    value=ev.text,
                    source=ev.source if ev.source in {"asr", "auto", "human"} else "auto",  # type: ignore[arg-type]
                )
            )
        else:
            lanes[lane].append(
                Event(
                    type=ev.kind.upper(),
                    start=ms_to_sec(ev.start_ms),
                    end=ms_to_sec(ev.end_ms),
                    value=ev.value or ev.text,
                    source=ev.source if ev.source in {"asr", "auto", "human"} else "auto",  # type: ignore[arg-type]
                )
            )
    return SpeechPlan(
        audio_path=str(rep.meta.get("audio_path") or ""),
        duration=ms_to_sec(rep.duration_ms),
        language=rep.language,
        global_state=GlobalState(value=None, source="human"),
        nonverbal=[],
        lanes=lanes,
        tracks={},
        asr={"transcript": rep.transcript, "words": len(rep.words())},
    )
