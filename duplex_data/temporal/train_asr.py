"""Phase 1: train TemporalFusion on frozen Bellu IndicConformer H_asr.

Input to fusion is only H_asr (and lengths). TSR JSON is loss targets only.
"""

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
from duplex_data.temporal.fusion import TemporalFusionModule, fusion_loss
from duplex_data.temporal.h_asr import (
    ENCODER_DIM,
    encode_wavs_to_h_asr,
    infer_asr_dim,
    load_frozen_encoder,
)

logger = logging.getLogger("duplex_data.temporal.train_asr")

DEFAULT_ASR_DIM = ENCODER_DIM  # Bellu IndicConformer-large encoder_dim


def train_one_epoch(
    model: TemporalFusionModule,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
    *,
    encoder=None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if encoder is not None:
            with torch.no_grad():
                batch["h_asr"] = encode_wavs_to_h_asr(
                    encoder,
                    batch["wav"],
                    batch["wav_lengths"],
                    target_t=int(batch["h_asr"].size(1)),
                    target_lengths=batch["lengths"],
                )
        out = model(batch["h_asr"], lengths=batch["lengths"])
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
        description="Train TemporalFusion: frozen Bellu IndicConformer H_asr → TSR heads."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dir of input_tsr.json / TSR JSON")
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint output dir")
    parser.add_argument(
        "--encoder",
        choices=["none", "conformer"],
        default="conformer",
        help="conformer=Bellu .nemo encoder (required for real H_asr); none=zeros (offline smoke)",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="Bellu IndicConformer pack: hi/te/ta/kn",
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
    encoder = None
    asr_dim = args.asr_dim

    if args.encoder == "conformer":
        encoder = load_frozen_encoder(args.device, language=args.lang)
        if encoder is None:
            logger.error("Bellu IndicConformer .nemo unavailable; re-run with --encoder none")
            return 1
        asr_dim = infer_asr_dim(encoder, args.device, args.asr_dim)
        logger.info("Using H_asr dim=%d from frozen Bellu encoder", asr_dim)

    ds = TSRFrameDataset(args.data, frame_ms=args.frame_ms)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_tsr_frames, asr_dim=asr_dim),
    )

    model = TemporalFusionModule(asr_dim=asr_dim, d_model=args.d_model).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optim, args.device, encoder=encoder)
        history.append({"epoch": epoch, **stats})
        logger.info(
            "epoch %d loss=%.4f pause=%.4f pitch=%.4f",
            epoch,
            stats["total"],
            stats["pause"],
            stats["pitch"],
        )

    ckpt = {
        "model": model.state_dict(),
        "asr_dim": asr_dim,
        "d_model": args.d_model,
        "encoder": args.encoder,
        "lang": args.lang,
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
