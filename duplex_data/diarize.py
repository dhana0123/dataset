# Copyright (c) Kyutai / DUPLEX, all rights reserved.
"""Two-speaker diarization → dual-channel gating (pyannote community-1)."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import wave
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("duplex_data.diarize")

MIN_SPEAKER_SEC = 0.5
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

Turn = tuple[str, float, float]
_pipeline_cache: dict[str, object] = {}


class DiarizationSkip(Exception):
    """Clip is not usable as a 2-speaker duplex example."""


@dataclass
class DiarizationResult:
    turns: list[Turn]
    agent_id: str
    user_id: str
    durations: dict[str, float]


def _mask_token(msg: str) -> str:
    return re.sub(r"hf_[A-Za-z0-9]+", "hf_***", msg)


@contextmanager
def _torch_load_compat_for_pyannote():
    import torch

    prev_env = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load  # type: ignore[assignment]
        if prev_env is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = prev_env


def _write_temp_wav(mono: np.ndarray, sample_rate: int) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    pcm = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


def load_diarization_pipeline(device: str = "cuda"):
    """Load pyannote community-1 (needs HF_TOKEN + model accept)."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "Install pyannote.audio>=4: pip install 'pyannote.audio>=4' (from data/)"
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — gated community-1 will fail.")
    else:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    key = f"{DIARIZATION_MODEL}|{device}|{bool(token)}"
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    gate_help = (
        "Accept https://huggingface.co/pyannote/speaker-diarization-community-1 "
        "with the same account as HF_TOKEN. Install: pip install 'pyannote.audio>=4'"
    )
    try:
        logger.info("Loading diarization pipeline %s …", DIARIZATION_MODEL)
        with _torch_load_compat_for_pyannote():
            kwargs = {"token": token} if token else {}
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {DIARIZATION_MODEL}: {_mask_token(str(exc))}\n{gate_help}"
        ) from exc
    if pipeline is None:
        raise RuntimeError(f"Failed to load {DIARIZATION_MODEL}.\n{gate_help}")
    if device.startswith("cuda") and torch.cuda.is_available():
        pipeline.to(torch.device(device))
    else:
        pipeline.to(torch.device("cpu"))
    _pipeline_cache[key] = pipeline
    logger.info("Diarization ready: %s", DIARIZATION_MODEL)
    return pipeline


def _annotation_to_turns(annotation) -> list[Turn]:
    turns: list[Turn] = []
    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start, end = float(segment.start), float(segment.end)
            if end > start:
                turns.append((str(speaker), start, end))
        return turns
    for item in annotation:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            turn, speaker = item[0], item[1]
            start = float(getattr(turn, "start", turn[0]))
            end = float(getattr(turn, "end", turn[1]))
            if end > start:
                turns.append((str(speaker), start, end))
    return turns


def speaker_durations(turns: list[Turn]) -> dict[str, float]:
    dur: dict[str, float] = defaultdict(float)
    for spk, start, end in turns:
        dur[spk] += end - start
    return dict(dur)


def pick_agent_user(durations: dict[str, float]) -> tuple[str, str]:
    if len(durations) < 2:
        raise DiarizationSkip(f"Need 2 speakers, got {list(durations)}")
    ranked = sorted(durations.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0], ranked[1][0]


def validate_two_speakers(
    durations: dict[str, float],
    *,
    min_sec: float = MIN_SPEAKER_SEC,
) -> None:
    if len(durations) != 2:
        raise DiarizationSkip(f"Expected exactly 2 speakers, got {durations}")
    for spk, d in durations.items():
        if d < min_sec:
            raise DiarizationSkip(f"Speaker {spk} only {d:.2f}s (< {min_sec}s); skip clip")


def diarize_two_speakers(
    mono: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cuda",
    min_sec: float = MIN_SPEAKER_SEC,
) -> DiarizationResult:
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0) if mono.shape[0] < mono.shape[-1] else mono.mean(axis=-1)

    pipeline = load_diarization_pipeline(device=device)
    path = _write_temp_wav(mono, sample_rate)
    try:
        try:
            output = pipeline(path, num_speakers=2)
        except TypeError:
            output = pipeline(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    else:
        annotation = output

    turns = _annotation_to_turns(annotation)
    if not turns:
        raise DiarizationSkip("Empty diarization")
    durations = speaker_durations(turns)
    if len(durations) < 2:
        raise DiarizationSkip(f"Diarization found {len(durations)} speaker(s): {durations}")
    if len(durations) > 2:
        top = sorted(durations.items(), key=lambda kv: -kv[1])[:2]
        keep = {t[0] for t in top}
        turns = [t for t in turns if t[0] in keep]
        durations = speaker_durations(turns)
    validate_two_speakers(durations, min_sec=min_sec)
    agent_id, user_id = pick_agent_user(durations)
    return DiarizationResult(
        turns=turns, agent_id=agent_id, user_id=user_id, durations=durations
    )


def mono_to_stereo_from_diarization(
    mono: np.ndarray,
    sample_rate: int,
    result: DiarizationResult,
) -> np.ndarray:
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0) if mono.shape[0] < mono.shape[-1] else mono.mean(axis=-1)
    n = mono.shape[-1]
    agent = np.zeros(n, dtype=np.float32)
    user = np.zeros(n, dtype=np.float32)
    for spk, start, end in result.turns:
        i0 = max(0, int(round(start * sample_rate)))
        i1 = min(n, int(round(end * sample_rate)))
        if i1 <= i0:
            continue
        if spk == result.agent_id:
            agent[i0:i1] = mono[i0:i1]
        elif spk == result.user_id:
            user[i0:i1] = mono[i0:i1]
    return np.stack([agent, user], axis=0)


def diarize_mono_to_stereo(
    mono: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cuda",
    min_sec: float = MIN_SPEAKER_SEC,
) -> np.ndarray:
    result = diarize_two_speakers(mono, sample_rate, device=device, min_sec=min_sec)
    return mono_to_stereo_from_diarization(mono, sample_rate, result)
