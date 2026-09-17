"""Unit tests for temporal fusion / feature packer / TTS conditioner."""

from __future__ import annotations

from pathlib import Path

import torch

from duplex_data.temporal.features import (
    FEATURE_NAMES,
    duration_to_num_frames,
    pack_to_torch,
    pack_tsr_features,
)
from duplex_data.temporal.fusion import TemporalFusionModule, fusion_loss
from duplex_data.temporal.tts_conditioner import (
    TemporalEncoder,
    conditioner_proxy_loss,
    tsr_to_conditioner_tensors,
)
from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep


def _sample_tsr() -> TemporalSpeechRep:
    return TemporalSpeechRep(
        direction="input",
        language="en",
        transcript="Yeah I really like it",
        duration_ms=1250,
        events=[
            TSREvent(0, 200, "word", text="Yeah", source="asr"),
            TSREvent(0, 200, "backchannel", text="Yeah", source="auto"),
            TSREvent(200, 500, "pause", source="auto"),
            TSREvent(200, 500, "hesitation", source="auto"),
            TSREvent(500, 600, "word", text="I", source="asr"),
            TSREvent(600, 850, "word", text="really", source="asr"),
            TSREvent(600, 850, "emphasis", text="really", value="strong", source="auto"),
            TSREvent(850, 1050, "word", text="like", source="asr"),
            TSREvent(1050, 1250, "word", text="it", source="asr"),
            TSREvent(1050, 1250, "pitch", text="it", value="falling", source="auto"),
            TSREvent(1250, 1250, "boundary", value="completing", source="auto"),
        ],
    )


def test_pack_tsr_features_shapes():
    rep = _sample_tsr()
    packed = pack_tsr_features(rep, frame_ms=40)
    T = duration_to_num_frames(1250, 40)
    assert packed["num_frames"] == T
    assert packed["features"].shape == (T, len(FEATURE_NAMES))
    assert packed["pause"].sum() > 0
    assert packed["hesitation"].sum() > 0
    assert (packed["pitch_cls"] > 0).any()


def test_fusion_forward_and_loss():
    asr_dim = 32
    B, T = 2, 16
    model = TemporalFusionModule(asr_dim=asr_dim, side_dim=len(FEATURE_NAMES), d_model=64, use_transformer=True)
    h = torch.randn(B, T, asr_dim)
    side = torch.randn(B, T, len(FEATURE_NAMES))
    lengths = torch.tensor([16, 10])
    out = model(h, side, lengths=lengths)
    assert out["E_t"].shape == (B, T, 64)
    assert out["pause_logit"].shape == (B, T)
    assert out["pitch_logit"].shape[2] > 1

    batch = {
        "pause": torch.zeros(B, T),
        "hesitation": torch.zeros(B, T),
        "pitch_cls": torch.zeros(B, T, dtype=torch.long),
        "energy_cls": torch.zeros(B, T, dtype=torch.long),
        "emphasis_cls": torch.zeros(B, T, dtype=torch.long),
        "boundary_cls": torch.zeros(B, T, dtype=torch.long),
        "f0_norm": torch.zeros(B, T),
        "energy_norm": torch.zeros(B, T),
    }
    loss, stats = fusion_loss(out, batch, lengths=lengths)
    assert torch.isfinite(loss)
    assert "total" in stats
    loss.backward()


def test_train_asr_one_batch(tmp_path: Path):
    rep = _sample_tsr()
    (tmp_path / "input_tsr.json").write_text(
        __import__("json").dumps(rep.to_dict(), indent=2), encoding="utf-8"
    )
    from duplex_data.temporal.train_asr import main

    out = tmp_path / "run"
    rc = main(
        [
            "--data",
            str(tmp_path),
            "--out",
            str(out),
            "--encoder",
            "none",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--asr-dim",
            "32",
            "--d-model",
            "64",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    assert (out / "temporal_fusion.pt").is_file()


def test_tts_encoder_shapes_and_proxy_loss():
    rep = _sample_tsr()
    rep.direction = "output"
    side, wm = tsr_to_conditioner_tensors(rep, frame_ms=40)
    assert side.dim() == 2 and wm.shape[-1] == 1
    enc = TemporalEncoder(d_model=128, n_heads=4)
    e = enc(side.unsqueeze(0), wm.unsqueeze(0))
    assert e.shape == (1, side.size(0), 128)
    target = torch.randn(128)
    loss = conditioner_proxy_loss(e, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_pack_to_torch_keys():
    packed = pack_tsr_features(_sample_tsr())
    tensors = pack_to_torch(packed)
    assert "features" in tensors and "pause" in tensors
    assert tensors["pitch_cls"].dtype == torch.int64
