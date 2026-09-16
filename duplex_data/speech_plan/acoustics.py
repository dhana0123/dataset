"""Pitch / energy tracks via Praat (parselmouth) with librosa RMS fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("duplex_data.speech_plan.acoustics")

# Downsample tracks for JSON sidecar (analysis only — not LLM vocabulary).
TRACK_HOP_SEC = 0.01


@dataclass
class AcousticTracks:
    times: list[float]
    f0_hz: list[float | None]
    intensity_db: list[float | None]
    rms: list[float | None]

    def to_dict(self) -> dict:
        return {
            "hop_sec": TRACK_HOP_SEC,
            "times": self.times,
            "f0_hz": self.f0_hz,
            "intensity_db": self.intensity_db,
            "rms": self.rms,
        }


def _downsample(times: np.ndarray, values: np.ndarray, hop: float) -> tuple[list[float], list[float | None]]:
    if times.size == 0:
        return [], []
    t0 = float(times[0])
    t1 = float(times[-1])
    if t1 <= t0:
        return [t0], [_finite_or_none(float(values[0]))]
    grid = np.arange(t0, t1 + hop * 0.5, hop)
    out_t: list[float] = []
    out_v: list[float | None] = []
    for t in grid:
        idx = int(np.argmin(np.abs(times - t)))
        out_t.append(round(float(t), 4))
        out_v.append(_finite_or_none(float(values[idx])))
    return out_t, out_v


def _finite_or_none(v: float) -> float | None:
    if not np.isfinite(v) or v == 0.0:
        return None
    return round(v, 3)


def extract_acoustics(
    mono: np.ndarray,
    sample_rate: int,
    *,
    hop_sec: float = TRACK_HOP_SEC,
) -> AcousticTracks:
    """Extract F0 + intensity. Prefer parselmouth (Praat); fall back to librosa."""
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return AcousticTracks([], [], [], [])

    try:
        return _parselmouth_tracks(mono, sample_rate, hop_sec=hop_sec)
    except Exception as exc:
        logger.warning("parselmouth acoustics failed (%s); using librosa fallback", exc)
        return _librosa_tracks(mono, sample_rate, hop_sec=hop_sec)


def _parselmouth_tracks(mono: np.ndarray, sample_rate: int, *, hop_sec: float) -> AcousticTracks:
    import parselmouth

    snd = parselmouth.Sound(mono, sampling_frequency=sample_rate)
    pitch = snd.to_pitch(time_step=hop_sec, pitch_floor=75.0, pitch_ceiling=500.0)
    intensity = snd.to_intensity(minimum_pitch=75.0, time_step=hop_sec)

    pitch_times = np.asarray(pitch.xs(), dtype=np.float64)
    pitch_vals = pitch.selected_array["frequency"].astype(np.float64)
    pitch_vals[pitch_vals == 0] = np.nan

    int_times = np.asarray(intensity.xs(), dtype=np.float64)
    int_vals = np.asarray(intensity.values, dtype=np.float64).reshape(-1)

    # Align both to a common time grid from pitch times (or intensity if empty).
    if pitch_times.size == 0 and int_times.size == 0:
        return AcousticTracks([], [], [], [])

    if pitch_times.size:
        times_list, f0_list = _downsample(pitch_times, pitch_vals, hop_sec)
        times = np.asarray(times_list, dtype=np.float64)
    else:
        times_list, _ = _downsample(int_times, int_vals, hop_sec)
        times = np.asarray(times_list, dtype=np.float64)
        f0_list = [None] * len(times_list)

    intensity_db: list[float | None] = []
    for t in times:
        if int_times.size == 0:
            intensity_db.append(None)
            continue
        idx = int(np.argmin(np.abs(int_times - t)))
        intensity_db.append(_finite_or_none(float(int_vals[idx])))

    # RMS via librosa on same grid (optional companion).
    rms_list = _rms_on_grid(mono, sample_rate, times)

    return AcousticTracks(
        times=times_list if isinstance(times_list, list) else [round(float(t), 4) for t in times],
        f0_hz=f0_list,
        intensity_db=intensity_db,
        rms=rms_list,
    )


def _librosa_tracks(mono: np.ndarray, sample_rate: int, *, hop_sec: float) -> AcousticTracks:
    import librosa

    hop_length = max(1, int(round(hop_sec * sample_rate)))
    f0, voiced_flag, _ = librosa.pyin(
        mono.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
    f0_list: list[float | None] = []
    for i, v in enumerate(f0):
        if voiced_flag is not None and not bool(voiced_flag[i]):
            f0_list.append(None)
        else:
            f0_list.append(_finite_or_none(float(v)) if np.isfinite(v) else None)

    rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
    # pad/trim rms to match f0 length
    if rms.size < len(f0_list):
        rms = np.pad(rms, (0, len(f0_list) - rms.size))
    elif rms.size > len(f0_list):
        rms = rms[: len(f0_list)]

    # Approximate intensity from RMS in dB
    intensity_db: list[float | None] = []
    for r in rms:
        if r <= 1e-10:
            intensity_db.append(None)
        else:
            intensity_db.append(round(float(20.0 * np.log10(r + 1e-10)), 3))

    rms_list = [_finite_or_none(float(r) * 1000.0) for r in rms]  # scale for readability

    return AcousticTracks(
        times=[round(float(t), 4) for t in times],
        f0_hz=f0_list,
        intensity_db=intensity_db,
        rms=rms_list,
    )


def _rms_on_grid(mono: np.ndarray, sample_rate: int, times: np.ndarray) -> list[float | None]:
    try:
        import librosa
    except ImportError:
        return [None] * len(times)
    hop_length = max(1, int(round(TRACK_HOP_SEC * sample_rate)))
    rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
    out: list[float | None] = []
    for t in times:
        idx = int(np.argmin(np.abs(frame_times - t)))
        out.append(_finite_or_none(float(rms[idx]) * 1000.0))
    return out
