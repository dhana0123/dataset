"""TSR ↔ LLM-readable line format (+ JSON via schema)."""

from __future__ import annotations

import re
from pathlib import Path

from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep

_EVENT_LINE = re.compile(
    r"^(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*ms\s+"
    r"(?P<label>\S+)\s+"
    r"(?P<tags>.+?)\s*$"
)
_BOUNDARY_LINE = re.compile(
    r"^(?P<start>\d+)\s*ms\s+(?P<label>BOUNDARY)\s*(?P<tags>.*)$",
    re.IGNORECASE,
)


def _format_span(start_ms: int, end_ms: int) -> str:
    if start_ms == end_ms and start_ms >= 0:
        # Point events (often boundary) can use single timestamp form
        return f"{start_ms}ms"
    return f"{start_ms}-{end_ms}ms"


def _event_label(ev: TSREvent) -> str:
    if ev.kind == "word" and ev.text:
        return ev.text
    if ev.kind == "pause":
        return "PAUSE"
    if ev.kind == "boundary":
        return "BOUNDARY"
    if ev.kind == "hesitation" and not ev.text:
        return "PAUSE"
    return (ev.text or ev.kind).upper() if ev.kind != "word" else (ev.text or "")


def _event_tags(ev: TSREvent, *, primary_word: bool = False) -> str:
    parts: list[str] = []
    if ev.kind == "word":
        parts.append("word")
    elif ev.kind == "pause":
        parts.append("pause")
    elif ev.kind == "boundary":
        parts.append("boundary")
        if ev.value:
            parts.append(str(ev.value))
        return "|".join(parts) if parts else "boundary"
    else:
        parts.append(ev.kind)
    if ev.value and ev.kind not in {"pause"}:
        parts.append(str(ev.value))
    elif ev.value and ev.kind == "pause":
        pass
    return "|".join(parts)


def tsr_to_text(rep: TemporalSpeechRep) -> str:
    """Serialize TSR to inspectable / LLM-readable lines."""
    lines = [
        f"DIRECTION: {rep.direction}",
        f"LANGUAGE: {rep.language}",
        f"TRANSCRIPT: {rep.transcript}",
        f"DURATION_MS: {rep.duration_ms}",
        "EVENTS:",
    ]

    ordered = sorted(rep.events, key=lambda e: (e.start_ms, e.end_ms, e.kind))
    by_span: dict[tuple[int, int, str], list[TSREvent]] = {}
    for ev in ordered:
        if ev.kind == "word" and ev.text:
            key = (ev.start_ms, ev.end_ms, ev.text)
            by_span.setdefault(key, []).append(ev)

    leftovers: list[TSREvent] = []
    for ev in ordered:
        if ev.kind == "word":
            continue
        if ev.kind in {
            "backchannel",
            "emphasis",
            "pitch",
            "energy",
            "rate",
            "laugh",
            "breath",
            "sigh",
        }:
            attached = False
            if ev.text:
                key = (ev.start_ms, ev.end_ms, ev.text)
                if key in by_span:
                    by_span[key].append(ev)
                    attached = True
            if not attached:
                for (s, e, _text), group in by_span.items():
                    if s == ev.start_ms and e == ev.end_ms:
                        group.append(ev)
                        attached = True
                        break
            if not attached:
                leftovers.append(ev)
        else:
            leftovers.append(ev)

    for key in sorted(by_span.keys(), key=lambda k: (k[0], k[1], k[2])):
        group = by_span[key]
        start_ms, end_ms, text = key
        tags = ["word"]
        for ev in group:
            if ev.kind == "word":
                continue
            tags.append(ev.kind)
            if ev.value and ev.kind in {"pitch", "energy", "rate", "emphasis"}:
                tags.append(str(ev.value))
        seen: set[str] = set()
        tag_list: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                tag_list.append(t)
        lines.append(f"{start_ms}-{end_ms}ms     {text}       {'|'.join(tag_list)}")

    for ev in sorted(leftovers, key=lambda e: (e.start_ms, e.end_ms, e.kind)):
        if ev.kind == "hesitation":
            lines.append(
                f"{ev.start_ms}-{ev.end_ms}ms     PAUSE       pause|hesitation"
            )
            continue
        if ev.kind == "pause":
            has_hes = any(
                h.kind == "hesitation"
                and h.start_ms == ev.start_ms
                and h.end_ms == ev.end_ms
                for h in leftovers
            )
            if has_hes:
                continue
            lines.append(f"{ev.start_ms}-{ev.end_ms}ms     PAUSE       pause")
            continue
        if ev.kind == "boundary":
            tag = "boundary"
            if ev.value:
                tag = f"boundary|{ev.value}"
            if ev.start_ms == ev.end_ms:
                lines.append(f"{ev.start_ms}ms      BOUNDARY       {tag}")
            else:
                lines.append(
                    f"{ev.start_ms}-{ev.end_ms}ms     BOUNDARY       {tag}"
                )
            continue
        label = ev.kind.upper()
        tags = ev.kind
        if ev.value:
            tags = f"{ev.kind}|{ev.value}"
        lines.append(f"{ev.start_ms}-{ev.end_ms}ms     {label}       {tags}")

    return "\n".join(lines) + "\n"


def _parse_tags(tags: str) -> list[str]:
    return [t.strip() for t in tags.split("|") if t.strip()]


def text_to_tsr(
    text: str,
    *,
    direction: str = "output",
    language: str | None = None,
    meta: dict | None = None,
) -> TemporalSpeechRep:
    """Parse LLM / inspect line format into TSR."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    transcript = ""
    duration_ms = 0
    lang = language or "auto"
    dir_ = direction
    events: list[TSREvent] = []
    in_events = False

    for ln in lines:
        upper = ln.upper()
        if upper.startswith("DIRECTION:"):
            dir_ = ln.split(":", 1)[1].strip().lower()
            continue
        if upper.startswith("LANGUAGE:"):
            lang = ln.split(":", 1)[1].strip()
            continue
        if upper.startswith("TRANSCRIPT:"):
            transcript = ln.split(":", 1)[1].strip()
            continue
        if upper.startswith("DURATION_MS:"):
            duration_ms = int(ln.split(":", 1)[1].strip())
            continue
        if upper.startswith("EVENTS:"):
            in_events = True
            continue
        if not in_events:
            continue

        m_b = _BOUNDARY_LINE.match(ln.strip())
        if m_b and "-" not in ln.split("ms", 1)[0]:
            start = int(m_b.group("start"))
            tags = _parse_tags(m_b.group("tags") or "boundary")
            value = None
            for t in tags:
                if t not in {"boundary"}:
                    value = t
            events.append(
                TSREvent(
                    start_ms=start,
                    end_ms=start,
                    kind="boundary",
                    text=None,
                    value=value,
                    source="llm",
                )
            )
            continue

        m = _EVENT_LINE.match(ln.strip())
        if not m:
            # try boundary with range
            m2 = re.match(
                r"^(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*ms\s+BOUNDARY\s+(?P<tags>.+)$",
                ln.strip(),
                re.IGNORECASE,
            )
            if m2:
                tags = _parse_tags(m2.group("tags"))
                value = next((t for t in tags if t != "boundary"), None)
                events.append(
                    TSREvent(
                        start_ms=int(m2.group("start")),
                        end_ms=int(m2.group("end")),
                        kind="boundary",
                        value=value,
                        source="llm",
                    )
                )
            continue

        start = int(m.group("start"))
        end = int(m.group("end"))
        label = m.group("label")
        tags = _parse_tags(m.group("tags"))

        if label.upper() == "PAUSE" or "pause" in tags:
            events.append(
                TSREvent(
                    start_ms=start,
                    end_ms=end,
                    kind="pause",
                    source="llm",
                )
            )
            if "hesitation" in tags:
                events.append(
                    TSREvent(
                        start_ms=start,
                        end_ms=end,
                        kind="hesitation",
                        source="llm",
                    )
                )
            continue

        if label.upper() == "BOUNDARY" or "boundary" in tags:
            value = next((t for t in tags if t not in {"boundary"}), None)
            events.append(
                TSREvent(
                    start_ms=start,
                    end_ms=end,
                    kind="boundary",
                    value=value,
                    source="llm",
                )
            )
            continue

        # Word (+ optional tags)
        if "word" in tags or label.lower() not in {
            "pause",
            "boundary",
            "emphasis",
            "pitch",
            "energy",
            "rate",
        }:
            events.append(
                TSREvent(
                    start_ms=start,
                    end_ms=end,
                    kind="word",
                    text=label,
                    source="llm",
                )
            )
            for t in tags:
                if t in {"word"}:
                    continue
                if t in {
                    "backchannel",
                    "emphasis",
                    "hesitation",
                    "laugh",
                    "breath",
                    "sigh",
                    "interruption",
                    "overlap",
                }:
                    events.append(
                        TSREvent(
                            start_ms=start,
                            end_ms=end,
                            kind=t,
                            text=label,
                            source="llm",
                        )
                    )
                elif t in {"rising", "falling", "stable", "warm/falling"}:
                    events.append(
                        TSREvent(
                            start_ms=start,
                            end_ms=end,
                            kind="pitch",
                            text=label,
                            value=t,
                            source="llm",
                        )
                    )
                elif t in {"stronger", "softer", "strong", "weak", "faster", "slower"}:
                    kind = (
                        "emphasis"
                        if t in {"strong", "weak"}
                        else ("energy" if t in {"stronger", "softer"} else "rate")
                    )
                    events.append(
                        TSREvent(
                            start_ms=start,
                            end_ms=end,
                            kind=kind,
                            text=label,
                            value=t,
                            source="llm",
                        )
                    )
            continue

    if not transcript:
        transcript = " ".join(e.text for e in events if e.kind == "word" and e.text)
    if duration_ms <= 0 and events:
        duration_ms = max(e.end_ms for e in events)

    rep = TemporalSpeechRep(
        direction=dir_ if dir_ in {"input", "output"} else "output",  # type: ignore[arg-type]
        language=lang,
        transcript=transcript,
        duration_ms=duration_ms,
        events=events,
        meta=dict(meta or {}),
    )
    rep.sort_events()
    return rep


def save_text(rep: TemporalSpeechRep, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tsr_to_text(rep), encoding="utf-8")
