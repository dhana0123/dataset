"""Phase 2: train TTS TemporalEncoder only (Parler fully frozen — no LoRA).

v1 trains a proxy objective: TemporalEncoder output should match a deterministic
embedding of the Parler description derived from TSR (via tsr_to_parler_inputs).
Full teacher-forced DAC loss through frozen Parler can be wired later without
changing this module interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from duplex_data.temporal.tts_conditioner import (
    TemporalEncoder,
    conditioner_proxy_loss,
    tsr_description_target,
    tsr_to_conditioner_tensors,
)
from duplex_data.tsr.schema import TemporalSpeechRep

logger = logging.getLogger("duplex_data.temporal.train_tts")


class OutputTSRDataset(Dataset):
    def __init__(self, root: Path, *, frame_ms: int = 40, d_model: int = 1024):
        self.root = Path(root)
        self.frame_ms = frame_ms
        self.d_model = d_model
        self.paths = []
        for p in sorted(self.root.rglob("*.json")):
            name = p.name.lower()
            if name in {"output_tsr.json"} or name.endswith("_out_tsr.json"):
                self.paths.append(p)
                continue
            if name == "input_tsr.json":
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("direction") == "output" and "events" in obj:
                self.paths.append(p)
        # Fallback: any TSR json if no explicit output files
        if not self.paths:
            for p in sorted(self.root.rglob("*.json")):
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(obj, dict) and "events" in obj and "duration_ms" in obj:
                    self.paths.append(p)
        if not self.paths:
            raise FileNotFoundError(f"No TSR JSON under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        rep = TemporalSpeechRep.load(self.paths[idx])
        if rep.direction != "output":
            rep.direction = "output"
        side, word_mask = tsr_to_conditioner_tensors(rep, frame_ms=self.frame_ms)
        target = tsr_description_target(rep, self.d_model)
        return {
            "side": side,
            "word_mask": word_mask,
            "target": target,
            "path": str(self.paths[idx]),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    sides = pad_sequence([b["side"] for b in batch], batch_first=True)
    masks = pad_sequence([b["word_mask"] for b in batch], batch_first=True)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    lengths = torch.tensor([b["side"].size(0) for b in batch], dtype=torch.long)
    return {"side": sides, "word_mask": masks, "target": targets, "lengths": lengths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train TTS TemporalEncoder only (Parler frozen, no LoRA)."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=1024, help="Match Parler decoder hidden if hooking later")
    parser.add_argument("--epochs", type=int, default=3)
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

    ds = OutputTSRDataset(args.data, frame_ms=args.frame_ms, d_model=args.d_model)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = TemporalEncoder(d_model=args.d_model).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            side = batch["side"].to(args.device)
            wm = batch["word_mask"].to(args.device)
            target = batch["target"].to(args.device)
            e = model(side, wm)
            loss = conditioner_proxy_loss(e, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.detach().cpu())
            n += 1
        avg = total / max(n, 1)
        history.append({"epoch": epoch, "loss": avg})
        logger.info("epoch %d loss=%.4f", epoch, avg)

    path = args.out / "tts_temporal_encoder.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "d_model": args.d_model,
            "frame_ms": args.frame_ms,
            "parler_lora": False,
            "history": history,
        },
        path,
    )
    (args.out / "train_tts_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s (Parler untouched, no LoRA)", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
