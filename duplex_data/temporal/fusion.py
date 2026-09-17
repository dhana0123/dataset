"""TemporalFusion: concat H_asr (+ side features) → E_t → TSR heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from duplex_data.temporal.features import (
    BOUNDARY_LABELS,
    EMPHASIS_LABELS,
    ENERGY_LABELS,
    FEATURE_NAMES,
    PITCH_LABELS,
)


class TemporalFusionModule(nn.Module):
    """Concat + projection (+ optional 2-layer TransformerEncoder)."""

    def __init__(
        self,
        asr_dim: int,
        side_dim: int = len(FEATURE_NAMES),
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.asr_dim = asr_dim
        self.side_dim = side_dim
        self.d_model = d_model
        in_dim = asr_dim + side_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.use_transformer = use_transformer
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = nn.Identity()

        self.pause_head = nn.Linear(d_model, 1)
        self.hesitation_head = nn.Linear(d_model, 1)
        self.pitch_head = nn.Linear(d_model, len(PITCH_LABELS))
        self.energy_cls_head = nn.Linear(d_model, len(ENERGY_LABELS))
        self.emphasis_head = nn.Linear(d_model, len(EMPHASIS_LABELS))
        self.boundary_head = nn.Linear(d_model, len(BOUNDARY_LABELS))
        self.f0_head = nn.Linear(d_model, 1)
        self.energy_reg_head = nn.Linear(d_model, 1)

    def forward(
        self,
        h_asr: torch.Tensor,
        side: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([h_asr, side], dim=-1)
        e = self.in_proj(x)
        if self.use_transformer:
            key_padding = None
            if lengths is not None:
                T = e.size(1)
                idx = torch.arange(T, device=e.device).unsqueeze(0)
                key_padding = idx >= lengths.unsqueeze(1)
            e = self.encoder(e, src_key_padding_mask=key_padding)
        return {
            "E_t": e,
            "pause_logit": self.pause_head(e).squeeze(-1),
            "hesitation_logit": self.hesitation_head(e).squeeze(-1),
            "pitch_logit": self.pitch_head(e),
            "energy_cls_logit": self.energy_cls_head(e),
            "emphasis_logit": self.emphasis_head(e),
            "boundary_logit": self.boundary_head(e),
            "f0_pred": self.f0_head(e).squeeze(-1),
            "energy_pred": self.energy_reg_head(e).squeeze(-1),
        }


def fusion_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, T = out["pause_logit"].shape
    device = out["pause_logit"].device
    if lengths is None:
        mask = torch.ones(B, T, device=device, dtype=torch.float32)
    else:
        idx = torch.arange(T, device=device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()

    def _bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _ce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = logit.size(-1)
        loss = F.cross_entropy(
            logit.reshape(-1, C),
            target.reshape(-1),
            reduction="none",
        ).view(B, T)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target).pow(2)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    parts = {
        "pause": _bce(out["pause_logit"], batch["pause"]),
        "hesitation": _bce(out["hesitation_logit"], batch["hesitation"]),
        "pitch": _ce(out["pitch_logit"], batch["pitch_cls"]),
        "energy_cls": _ce(out["energy_cls_logit"], batch["energy_cls"]),
        "emphasis": _ce(out["emphasis_logit"], batch["emphasis_cls"]),
        "boundary": _ce(out["boundary_logit"], batch["boundary_cls"]),
        "f0": _mse(out["f0_pred"], batch["f0_norm"]),
        "energy_reg": _mse(out["energy_pred"], batch["energy_norm"]),
    }
    total = sum(parts.values())
    stats = {k: float(v.detach().cpu()) for k, v in parts.items()}
    stats["total"] = float(total.detach().cpu())
    return total, stats
