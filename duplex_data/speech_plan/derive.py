"""Derive discrete speech-control events from words + acoustic tracks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from duplex_data.asr import WordSpan
from duplex_data.speech_plan.acoustics import AcousticTracks
from duplex_data.speech_plan.schema import Event

PAUSE_MIN_SEC = 0.18
PITCH_SLOPE_THRESHOLD = 15.0  # Hz / sec for rising/falling
PITCH_MIN_SPAN_SEC = 0.12
ENERGY_DELTA_DB = 3.0
RATE_RATIO_FAST = 0.75  # duration < 0.75 * median → faster
RATE_RATIO_SLOW = 1.35
EMPHASIS_ENERGY_Z = 0.8
EMPHASIS_DUR_RATIO = 1.2


def words_to_content_events(words: Sequence[WordSpan]) -> list[Event]:
    return [
        Event(type="WORD", start=float(s), end=float(e), value=w, source="asr")
        for w, (s, e) in words
    ]


def derive_pauses(words: Sequence[WordSpan], *, min_gap: float = PAUSE_MIN_SEC) -> list[Event]:
    events: list[Event] = []
    ordered = sorted(words, key=lambda x: x[1][0])
    for i in range(len(ordered) - 1):
        _, (_, end_a) = ordered[i]
        _, (start_b, _) = ordered[i + 1]
        gap = float(start_b) - float(end_a)
        if gap >= min_gap:
            events.append(
                Event(
                    type="PAUSE",
                    start=float(end_a),
                    end=float(start_b),
                    value=round(gap * 1000.0),  # ms
                    source="auto",
                )
            )
    return events


def derive_rate(words: Sequence[WordSpan]) -> list[Event]:
    if len(words) < 2:
        return []
    durs = [max(1e-4, float(e) - float(s)) for _, (s, e) in words]
    med = float(np.median(durs))
    if med <= 0:
        return []
    events: list[Event] = []
    for (w, (s, e)), d in zip(words, durs):
        ratio = d / med
        if ratio <= RATE_RATIO_FAST:
            events.append(
                Event(type="RATE", start=float(s), end=float(e), value="faster", source="auto")
            )
        elif ratio >= RATE_RATIO_SLOW:
            events.append(
                Event(type="RATE", start=float(s), end=float(e), value="slower", source="auto")
            )
    return events


def _interp_track(
    times: Sequence[float],
    values: Sequence[float | None],
    t0: float,
    t1: float,
) -> np.ndarray:
    """Return finite values whose times fall in [t0, t1]."""
    out: list[float] = []
    for t, v in zip(times, values):
        if v is None:
            continue
        if t0 <= t <= t1:
            out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def derive_pitch(tracks: AcousticTracks, *, min_span: float = PITCH_MIN_SPAN_SEC) -> list[Event]:
    """Discretize F0 into rising / falling / stable spans over voiced regions."""
    times = np.asarray(tracks.times, dtype=np.float64)
    f0 = tracks.f0_hz
    if times.size < 3:
        return []

    # Build voiced contiguous segments
    voiced_mask = np.array([v is not None for v in f0], dtype=bool)
    events: list[Event] = []
    i = 0
    n = len(f0)
    while i < n:
        if not voiced_mask[i]:
            i += 1
            continue
        j = i
        while j < n and voiced_mask[j]:
            j += 1
        seg_t = times[i:j]
        seg_f = np.array([float(f0[k]) for k in range(i, j)], dtype=np.float64)
        if seg_t.size >= 2 and float(seg_t[-1] - seg_t[0]) >= min_span:
            # Linear slope
            x = seg_t - seg_t[0]
            if np.std(x) < 1e-6:
                slope = 0.0
            else:
                slope = float(np.polyfit(x, seg_f, 1)[0])
            if slope >= PITCH_SLOPE_THRESHOLD:
                value = "rising"
            elif slope <= -PITCH_SLOPE_THRESHOLD:
                value = "falling"
            else:
                value = "stable"
            events.append(
                Event(
                    type="PITCH",
                    start=float(seg_t[0]),
                    end=float(seg_t[-1]),
                    value=value,
                    intensity=round(abs(slope), 2),
                    source="auto",
                )
            )
        i = j
    return events


def derive_energy(
    words: Sequence[WordSpan],
    tracks: AcousticTracks,
    *,
    delta_db: float = ENERGY_DELTA_DB,
) -> list[Event]:
    if not words or not tracks.times:
        return []
    # Per-word mean intensity
    means: list[tuple[WordSpan, float]] = []
    for span in words:
        w, (s, e) = span
        vals = _interp_track(tracks.times, tracks.intensity_db, float(s), float(e))
        if vals.size == 0:
            continue
        means.append((span, float(np.nanmean(vals))))
    if len(means) < 2:
        return []
    med = float(np.median([m for _, m in means]))
    events: list[Event] = []
    for (w, (s, e)), m in means:
        if m >= med + delta_db:
            events.append(
                Event(type="ENERGY", start=float(s), end=float(e), value="stronger", source="auto")
            )
        elif m <= med - delta_db:
            events.append(
                Event(type="ENERGY", start=float(s), end=float(e), value="softer", source="auto")
            )
    return events


def propose_emphasis(
    words: Sequence[WordSpan],
    tracks: AcousticTracks,
) -> list[Event]:
    """Weak auto-propose; human is source of truth in the UI."""
    if len(words) < 2 or not tracks.times:
        return []
    durs = [max(1e-4, float(e) - float(s)) for _, (s, e) in words]
    med_dur = float(np.median(durs))
    energy_means: list[float] = []
    for _, (s, e) in words:
        vals = _interp_track(tracks.times, tracks.intensity_db, float(s), float(e))
        energy_means.append(float(np.nanmean(vals)) if vals.size else np.nan)
    finite = [e for e in energy_means if np.isfinite(e)]
    if len(finite) < 2:
        return []
    mean_e = float(np.mean(finite))
    std_e = float(np.std(finite)) or 1.0

    events: list[Event] = []
    for (w, (s, e)), d, em in zip(words, durs, energy_means):
        if not np.isfinite(em):
            continue
        z = (em - mean_e) / std_e
        if z >= EMPHASIS_ENERGY_Z and d >= med_dur * EMPHASIS_DUR_RATIO:
            events.append(
                Event(type="EMPHASIS", start=float(s), end=float(e), value="strong", source="auto")
            )
        elif z >= EMPHASIS_ENERGY_Z:
            events.append(
                Event(type="EMPHASIS", start=float(s), end=float(e), value="weak", source="auto")
            )
    return events


def propose_boundaries(
    words: Sequence[WordSpan],
    pauses: Sequence[Event],
    pitch_events: Sequence[Event],
    duration: float,
) -> list[Event]:
    """Heuristic: end-of-clip completing; long pause → continuing/completing; rising F0 near end → questioning."""
    events: list[Event] = []
    if not words:
        return events

    last_word = max(words, key=lambda x: x[1][1])
    _, (lw_s, lw_e) = last_word
    end_t = float(lw_e)

    # Check pitch near last word
    rising_near_end = any(
        p.value == "rising" and float(p.end) >= end_t - 0.4 for p in pitch_events
    )
    if rising_near_end:
        events.append(
            Event(
                type="BOUNDARY",
                start=max(0.0, end_t - 0.05),
                end=min(duration, end_t + 0.05),
                value="questioning",
                source="auto",
            )
        )
    else:
        events.append(
            Event(
                type="BOUNDARY",
                start=max(0.0, end_t - 0.05),
                end=min(duration, end_t + 0.05),
                value="completing",
                source="auto",
            )
        )

    # Long pauses → continuing boundary at pause end (next phrase start)
    for p in pauses:
        gap_ms = float(p.value) if isinstance(p.value, (int, float)) else 0.0
        if gap_ms >= 300:
            events.append(
                Event(
                    type="BOUNDARY",
                    start=float(p.end) - 0.02,
                    end=float(p.end) + 0.02,
                    value="continuing",
                    source="auto",
                )
            )
    return events


def build_lanes(
    words: Sequence[WordSpan],
    tracks: AcousticTracks,
    duration: float,
) -> dict[str, list[Event]]:
    content = words_to_content_events(words)
    pauses = derive_pauses(words)
    pitch = derive_pitch(tracks)
    energy = derive_energy(words, tracks)
    rate = derive_rate(words)
    emphasis = propose_emphasis(words, tracks)
    boundary = propose_boundaries(words, pauses, pitch, duration)
    return {
        "content": content,
        "pitch": pitch,
        "energy": energy,
        "rate": rate,
        "emphasis": emphasis,
        "pause": pauses,
        "boundary": boundary,
        "nonverbal": [],
    }
