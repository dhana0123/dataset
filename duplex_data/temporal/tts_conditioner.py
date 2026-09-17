"""TTS TemporalEncoder: SpeechPlan/TSR → conditioning sequence for Parler."""

from __future__ import annotations

import torch
import torch.nn as nn

from duplex_data.temporal.features import FEATURE_NAMES, pack_tsr_features
from duplex_data.tsr.schema import TemporalSpeechRep
from duplex_data.tsr.to_tts import tsr_to_parler_inputs


class TemporalEncoder(nn.Module):
    """Encode packed TSR frame features into d_model sequence for cross-attn concat."""

    def __init__(
        self,
        side_dim: int = len(FEATURE_NAMES),
        d_model: int = 1024,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.side_dim = side_dim
        self.in_proj = nn.Linear(side_dim + 1, d_model)  # +1 word presence channel
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, side: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """
        side: [B, T, F]
        word_mask: [B, T, 1]
        returns E_tts [B, T, d_model]
        """
        x = torch.cat([side, word_mask], dim=-1)
        h = self.in_proj(x)
        return self.encoder(h)


def tsr_to_conditioner_tensors(
    rep: TemporalSpeechRep,
    *,
    frame_ms: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (side [T,F], word_mask [T,1]) from TSR."""
    packed = pack_tsr_features(rep, frame_ms=frame_ms)
    side = torch.from_numpy(packed["features"])
    T = side.size(0)
    word_mask = torch.zeros(T, 1, dtype=torch.float32)
    for ev in rep.events:
        if ev.kind != "word":
            continue
        i0 = max(0, int(ev.start_ms) // frame_ms)
        i1 = min(T, max(i0 + 1, int(max(ev.end_ms, ev.start_ms + 1)) // frame_ms + 1))
        word_mask[i0:i1, 0] = 1.0
    return side, word_mask


def conditioner_proxy_loss(
    e_tts: torch.Tensor,
    target_embed: torch.Tensor,
) -> torch.Tensor:
    """MSE between mean-pooled E_tts and a target description embedding (proxy without Parler LM)."""
    pred = e_tts.mean(dim=1)
    if target_embed.dim() == 1:
        target_embed = target_embed.unsqueeze(0)
    # Project if dims differ
    if pred.size(-1) != target_embed.size(-1):
        # truncate or pad
        d = min(pred.size(-1), target_embed.size(-1))
        pred = pred[..., :d]
        target_embed = target_embed[..., :d]
    return torch.nn.functional.mse_loss(pred, target_embed)


def hash_embed_description(text: str, d_model: int) -> torch.Tensor:
    """Deterministic bag-of-char hash embedding (no HF needed for unit train)."""
    v = torch.zeros(d_model, dtype=torch.float32)
    for i, ch in enumerate(text):
        v[i % d_model] += (ord(ch) % 31) / 31.0
    n = max(len(text), 1)
    return v / n


def tsr_description_target(rep: TemporalSpeechRep, d_model: int) -> torch.Tensor:
    _, desc = tsr_to_parler_inputs(rep)
    return hash_embed_description(desc, d_model)
