"""Dataset: directory of TSR JSON (+ optional wav) for temporal training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from duplex_data.temporal.features import DEFAULT_FRAME_MS, pack_to_torch, pack_tsr_features
from duplex_data.tsr.schema import TemporalSpeechRep

logger = logging.getLogger("duplex_data.temporal.dataset")


class TSRFrameDataset(Dataset):
    """Each item: TSR head targets on a 40 ms grid, plus optional 16 kHz wav."""

    def __init__(
        self,
        root: Path | str,
        *,
        frame_ms: int = DEFAULT_FRAME_MS,
        prefer_input: bool = True,
    ):
        self.root = Path(root)
        self.frame_ms = frame_ms
        self.paths = self._discover(self.root, prefer_input=prefer_input)
        if not self.paths:
            raise FileNotFoundError(f"No TSR JSON under {self.root}")

    @staticmethod
    def _discover(root: Path, *, prefer_input: bool) -> list[Path]:
        paths: list[Path] = []
        if not root.exists():
            return paths
        for p in sorted(root.rglob("*.json")):
            name = p.name.lower()
            if prefer_input and name in {"output_tsr.json"}:
                continue
            if name in {"input_tsr.json", "output_tsr.json"} or name.endswith("_tsr.json"):
                paths.append(p)
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict) and "events" in obj and "duration_ms" in obj:
                paths.append(p)
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        rep = TemporalSpeechRep.load(path)
        packed = pack_tsr_features(rep, frame_ms=self.frame_ms)
        tensors = pack_to_torch(packed)
        tensors.pop("features", None)
        tensors["path"] = str(path)
        wav = _load_wav_16k(_resolve_wav(path, rep.meta))
        tensors["wav"] = torch.from_numpy(wav)
        tensors["wav_length"] = torch.tensor(int(wav.shape[0]), dtype=torch.long)
        return tensors


def collate_tsr_frames(
    batch: list[dict],
    *,
    asr_dim: int,
) -> dict[str, torch.Tensor]:
    """Pad frame targets; placeholder zero H_asr until the encoder overwrites it."""
    lengths = torch.tensor([b["pause"].size(0) for b in batch], dtype=torch.long)
    B = len(batch)
    T = int(lengths.max().item()) if B else 1
    h_asr = torch.zeros(B, T, asr_dim, dtype=torch.float32)

    def _pad(key: str, dtype=torch.float32) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        if dtype == torch.long:
            return pad_sequence(seqs, batch_first=True, padding_value=0)
        return pad_sequence(seqs, batch_first=True, padding_value=0.0)

    wavs = pad_sequence([b["wav"] for b in batch], batch_first=True, padding_value=0.0)
    wav_lengths = torch.stack([b["wav_length"] for b in batch])
    return {
        "h_asr": h_asr,
        "wav": wavs,
        "wav_lengths": wav_lengths,
        "lengths": lengths,
        "pause": _pad("pause"),
        "hesitation": _pad("hesitation"),
        "pitch_cls": _pad("pitch_cls", dtype=torch.long),
        "energy_cls": _pad("energy_cls", dtype=torch.long),
        "emphasis_cls": _pad("emphasis_cls", dtype=torch.long),
        "boundary_cls": _pad("boundary_cls", dtype=torch.long),
        "f0_norm": _pad("f0_norm"),
        "energy_norm": _pad("energy_norm"),
    }


def _resolve_wav(json_path: Path, meta: dict) -> Path | None:
    raw = meta.get("audio_path") if isinstance(meta, dict) else None
    if raw:
        p = Path(str(raw))
        if p.is_file():
            return p
        sibling = json_path.parent / p.name
        if sibling.is_file():
            return sibling
    for cand in sorted(json_path.parent.glob("*.wav")):
        return cand
    return None


def _load_wav_16k(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(1, dtype=np.float32)
    try:
        from duplex_data.asr import ASR_SAMPLE_RATE, resample_mono

        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
            return resample_mono(data[:, 0], int(sr), ASR_SAMPLE_RATE)
        except ImportError:
            import wave

            with wave.open(str(path), "rb") as wf:
                sr = int(wf.getframerate())
                n = int(wf.getnframes())
                raw = wf.readframes(n)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            return resample_mono(pcm, sr, ASR_SAMPLE_RATE)
    except Exception as exc:
        logger.warning("Could not load wav %s (%s); using silence", path, exc)
        return np.zeros(1, dtype=np.float32)
