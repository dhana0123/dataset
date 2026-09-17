"""Dataset: directory of TSR JSON (+ optional wav) for temporal training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from duplex_data.temporal.features import DEFAULT_FRAME_MS, pack_to_torch, pack_tsr_features
from duplex_data.tsr.schema import TemporalSpeechRep

logger = logging.getLogger("duplex_data.temporal.dataset")


class TSRFrameDataset(Dataset):
    """Each item: packed frame features/targets from input_tsr.json or *.json TSR."""

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
            # Accept any JSON that looks like TSR
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
        tensors["path"] = str(path)
        tensors["features"] = tensors["features"]  # [T,F]
        # Placeholder H_asr for encoder=none: zeros with configured dim filled in collate/train
        return tensors


def collate_tsr_frames(
    batch: list[dict],
    *,
    asr_dim: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length frame sequences; synthesize zero H_asr for encoder=none."""
    lengths = torch.tensor([b["features"].size(0) for b in batch], dtype=torch.long)
    features = pad_sequence([b["features"] for b in batch], batch_first=True)
    B, T, F = features.shape
    h_asr = torch.zeros(B, T, asr_dim, dtype=torch.float32)

    def _pad(key: str, dtype=torch.float32) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        if dtype == torch.long:
            return pad_sequence(seqs, batch_first=True, padding_value=0)
        return pad_sequence(seqs, batch_first=True, padding_value=0.0)

    out = {
        "h_asr": h_asr,
        "features": features,
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
    return out
