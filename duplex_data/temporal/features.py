"""Pack TSR events onto a fixed frame grid (ms → frames)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from duplex_data.tsr.schema import TemporalSpeechRep

# Native IndicConformer-large stride: 10 ms mel hop × subsampling 4.
DEFAULT_FRAME_MS = 40

PITCH_LABELS = ("none", "rising", "falling", "stable", "warm/falling")
ENERGY_LABELS = ("none", "stronger", "softer")
EMPHASIS_LABELS = ("none", "strong", "weak")
BOUNDARY_LABELS = ("none", "completing", "continuing", "questioning")

FEATURE_NAMES = (
    "pause",
    "hesitation",
    "backchannel",
    "emphasis_bin",
    "pitch_rising",
    "pitch_falling",
    "energy_stronger",
    "energy_softer",
    "rate_faster",
    "rate_slower",
    "boundary_bin",
)


def duration_to_num_frames(duration_ms: int, frame_ms: int = DEFAULT_FRAME_MS) -> int:
    return max(1, int(np.ceil(float(duration_ms) / float(frame_ms))))


def ms_to_frame(ms: int, frame_ms: int = DEFAULT_FRAME_MS) -> int:
    return int(ms) // int(frame_ms)


def _span_slice(T: int, start_ms: int, end_ms: int, frame_ms: int) -> tuple[int, int]:
    i0 = max(0, ms_to_frame(start_ms, frame_ms))
    i1 = min(T, max(i0 + 1, ms_to_frame(max(end_ms, start_ms + 1), frame_ms) + 1))
    return i0, i1


def pack_tsr_features(
    rep: TemporalSpeechRep,
    *,
    frame_ms: int = DEFAULT_FRAME_MS,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Return numpy side-features + head targets aligned to frames."""
    T = num_frames or duration_to_num_frames(rep.duration_ms, frame_ms)
    F = len(FEATURE_NAMES)
    feats = np.zeros((T, F), dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(FEATURE_NAMES)}

    pause = np.zeros(T, dtype=np.float32)
    hesitation = np.zeros(T, dtype=np.float32)
    pitch_cls = np.zeros(T, dtype=np.int64)
    energy_cls = np.zeros(T, dtype=np.int64)
    emphasis_cls = np.zeros(T, dtype=np.int64)
    boundary_cls = np.zeros(T, dtype=np.int64)
    f0_norm = np.zeros(T, dtype=np.float32)
    energy_norm = np.zeros(T, dtype=np.float32)

    for ev in rep.events:
        s, e = int(ev.start_ms), int(ev.end_ms)
        i0, i1 = _span_slice(T, s, e, frame_ms)
        if ev.kind == "pause":
            pause[i0:i1] = 1.0
            feats[i0:i1, name_to_i["pause"]] = 1.0
        elif ev.kind == "hesitation":
            hesitation[i0:i1] = 1.0
            feats[i0:i1, name_to_i["hesitation"]] = 1.0
        elif ev.kind == "backchannel":
            feats[i0:i1, name_to_i["backchannel"]] = 1.0
        elif ev.kind == "emphasis":
            feats[i0:i1, name_to_i["emphasis_bin"]] = 1.0
            lab = str(ev.value or "strong")
            idx = EMPHASIS_LABELS.index(lab) if lab in EMPHASIS_LABELS else 1
            emphasis_cls[i0:i1] = idx
        elif ev.kind == "pitch":
            lab = str(ev.value or "stable")
            if lab not in PITCH_LABELS:
                lab = "stable"
            pitch_cls[i0:i1] = PITCH_LABELS.index(lab)
            if "rising" in lab:
                feats[i0:i1, name_to_i["pitch_rising"]] = 1.0
                f0_norm[i0:i1] = 0.7
            elif "falling" in lab:
                feats[i0:i1, name_to_i["pitch_falling"]] = 1.0
                f0_norm[i0:i1] = 0.3
            else:
                f0_norm[i0:i1] = 0.5
        elif ev.kind == "energy":
            lab = str(ev.value or "stronger")
            if lab not in ENERGY_LABELS:
                lab = "stronger"
            energy_cls[i0:i1] = ENERGY_LABELS.index(lab)
            if lab == "stronger":
                feats[i0:i1, name_to_i["energy_stronger"]] = 1.0
                energy_norm[i0:i1] = 0.8
            else:
                feats[i0:i1, name_to_i["energy_softer"]] = 1.0
                energy_norm[i0:i1] = 0.2
        elif ev.kind == "rate":
            lab = str(ev.value or "faster")
            key = "rate_faster" if lab == "faster" else "rate_slower"
            feats[i0:i1, name_to_i[key]] = 1.0
        elif ev.kind == "boundary":
            feats[i0:i1, name_to_i["boundary_bin"]] = 1.0
            lab = str(ev.value or "completing")
            if lab not in BOUNDARY_LABELS:
                lab = "completing"
            boundary_cls[i0:i1] = BOUNDARY_LABELS.index(lab)

    return {
        "features": feats,
        "pause": pause,
        "hesitation": hesitation,
        "pitch_cls": pitch_cls,
        "energy_cls": energy_cls,
        "emphasis_cls": emphasis_cls,
        "boundary_cls": boundary_cls,
        "f0_norm": f0_norm,
        "energy_norm": energy_norm,
        "num_frames": T,
        "frame_ms": frame_ms,
    }


def pack_to_torch(packed: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in packed.items():
        if k in {"num_frames", "frame_ms"}:
            continue
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.integer):
                out[k] = torch.from_numpy(v.astype(np.int64))
            else:
                out[k] = torch.from_numpy(v.astype(np.float32))
    out["num_frames"] = torch.tensor(int(packed["num_frames"]), dtype=torch.long)
    return out
