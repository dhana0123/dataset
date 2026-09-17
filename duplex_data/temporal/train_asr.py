"""Phase 1: train TemporalFusion (frozen Conformer or encoder=none)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from duplex_data.temporal.dataset import TSRFrameDataset, collate_tsr_frames
from duplex_data.temporal.features import FEATURE_NAMES
from duplex_data.temporal.fusion import TemporalFusionModule, fusion_loss

logger = logging.getLogger("duplex_data.temporal.train_asr")

# Provisional until official IndicConformer cfg is dumped.
DEFAULT_ASR_DIM = 512


def _try_load_conformer_encoder(device: str):
    """Best-effort frozen IndicConformer encoder. Returns None if unavailable."""
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        logger.info("Loaded frozen IndicConformer-600M (trust_remote_code)")
        return model
    except Exception as exc:
        logger.warning("Could not load IndicConformer (%s); use --encoder none", exc)
        return None


def train_one_epoch(
    model: TemporalFusionModule,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["h_asr"], batch["features"], lengths=batch["lengths"])
        loss, stats = fusion_loss(out, batch, lengths=batch["lengths"])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train ASR TemporalFusion on TSR frame targets (Conformer frozen)."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dir of input_tsr.json / TSR JSON")
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint output dir")
    parser.add_argument(
        "--encoder",
        choices=["none", "conformer"],
        default="none",
        help="none=zero H_asr placeholder; conformer=frozen IndicConformer (if accessible)",
    )
    parser.add_argument("--asr-dim", type=int, default=DEFAULT_ASR_DIM)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    asr_dim = args.asr_dim

    if args.encoder == "conformer":
        conf = _try_load_conformer_encoder(args.device)
        if conf is None:
            logger.error("Conformer unavailable; re-run with --encoder none")
            return 1
        # Full audio→H_asr loop deferred: v1 still trains with zero H_asr + side features
        # once cfg/hook is stable. Checkpoint records encoder=conformer intent.
        logger.warning(
            "Conformer loaded but frame-hook not wired in v1; training with zero H_asr "
            "+ TSR side features (same as none). Weights stay frozen for later hook."
        )
        del conf

    ds = TSRFrameDataset(args.data, frame_ms=args.frame_ms)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_tsr_frames, asr_dim=asr_dim),
    )

    model = TemporalFusionModule(
        asr_dim=asr_dim,
        side_dim=len(FEATURE_NAMES),
        d_model=args.d_model,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optim, args.device)
        history.append({"epoch": epoch, **stats})
        logger.info("epoch %d loss=%.4f pause=%.4f pitch=%.4f", epoch, stats["total"], stats["pause"], stats["pitch"])

    ckpt = {
        "model": model.state_dict(),
        "asr_dim": asr_dim,
        "side_dim": len(FEATURE_NAMES),
        "d_model": args.d_model,
        "encoder": args.encoder,
        "frame_ms": args.frame_ms,
        "history": history,
    }
    path = args.out / "temporal_fusion.pt"
    torch.save(ckpt, path)
    (args.out / "train_asr_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
