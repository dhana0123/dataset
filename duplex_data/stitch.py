"""Stitch dual-channel turns into Moshi IM stereo (L=agent, R=user)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .schema import SAMPLE_RATE


@dataclass
class StitchedDialog:
    stereo: np.ndarray  # [2, T] float32 @ SAMPLE_RATE
    sample_rate: int


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return x
    n_out = max(1, int(round(len(x) * dst_sr / float(src_sr))))
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def stitch_turns(
    turns: list[tuple[str, np.ndarray, int]],
    *,
    target_sr: int = SAMPLE_RATE,
    gap_sec: tuple[float, float] = (0.2, 0.55),
    overlap_prob: float = 0.2,
    overlap_sec: tuple[float, float] = (0.1, 0.28),
    rng: random.Random | None = None,
) -> StitchedDialog:
    """Place agent/user mono clips on L/R with gaps and occasional overlap.

    ``turns`` items: ``(role, mono_wav, sr)`` where role is ``agent`` or ``user``.
    """
    rng = rng or random.Random()
    agent_chunks: list[tuple[int, np.ndarray]] = []  # (start_sample, audio)
    user_chunks: list[tuple[int, np.ndarray]] = []
    cursor = 0

    for i, (role, wav, sr) in enumerate(turns):
        audio = _resample(wav, sr, target_sr)
        if i == 0:
            start = 0
        else:
            gap = rng.uniform(*gap_sec)
            start = cursor + int(gap * target_sr)
            if rng.random() < overlap_prob and cursor > 0:
                ov = rng.uniform(*overlap_sec)
                # Pull back at most half the previous gap/turn so short clips stay valid
                max_pull = min(int(ov * target_sr), max(1, cursor // 2))
                start = max(0, cursor - max_pull)
        if role == "agent":
            agent_chunks.append((start, audio))
        else:
            user_chunks.append((start, audio))
        cursor = start + len(audio)

    total = cursor
    for start, audio in agent_chunks + user_chunks:
        total = max(total, start + len(audio))
    if total <= 0:
        total = target_sr  # 1s silence fallback

    left = np.zeros(total, dtype=np.float32)
    right = np.zeros(total, dtype=np.float32)
    for start, audio in agent_chunks:
        end = start + len(audio)
        left[start:end] += audio
    for start, audio in user_chunks:
        end = start + len(audio)
        right[start:end] += audio

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    if peak > 1.0:
        left /= peak
        right /= peak

    stereo = np.stack([left, right], axis=0)
    return StitchedDialog(stereo=stereo, sample_rate=target_sr)
