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
    model = TemporalFusionModule(asr_dim=asr_dim, d_model=64, use_transformer=True)
    h = torch.randn(B, T, asr_dim)
    lengths = torch.tensor([16, 10])
    out = model(h, lengths=lengths)
    assert out["E_t"].shape == (B, T, 64)
    assert out["pause_logit"].shape == (B, T)
    assert out["pitch_logit"].shape[2] > 1

    import inspect

    params = list(inspect.signature(TemporalFusionModule.forward).parameters)
    assert params == ["self", "h_asr", "lengths"]

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


class _StubNemoEncoder:
    """NeMo-shaped preprocessor+encoder: [B, D, T], hop 160 then ×4."""

    D = 24

    def preprocessor(self, input_signal, length):
        # 10 ms at 16 kHz = 160 samples.
        hop = 160
        t_mel = torch.clamp(length // hop, min=1)
        t_max = int(t_mel.max().item())
        feat = input_signal.new_zeros(input_signal.size(0), self.D, t_max)
        for b in range(input_signal.size(0)):
            n = int(t_mel[b].item())
            # Encode waveform energy so H_asr is not a constant / not zeros.
            e = input_signal[b, : int(length[b].item())].pow(2).mean().sqrt().clamp_min(1e-4)
            feat[b, :, :n] = e
            feat[b, 0, :n] = torch.linspace(0, 1, n, device=feat.device) * e
        return feat, t_mel

    def encoder(self, audio_signal, length):
        encoded = audio_signal[:, :, ::4].contiguous()
        enc_len = torch.clamp(length // 4, min=1)
        return encoded, enc_len


def test_encode_h_asr_is_real_not_zeros():
    from duplex_data.temporal.h_asr import encode_wavs_to_h_asr

    enc = _StubNemoEncoder()
    wav_a = torch.ones(1, 16000)
    wav_b = torch.full((1, 16000), 2.0)
    lens = torch.tensor([16000])
    tgt = torch.tensor([32])
    h_a = encode_wavs_to_h_asr(enc, wav_a, lens, target_t=32, target_lengths=tgt)
    h_b = encode_wavs_to_h_asr(enc, wav_b, lens, target_t=32, target_lengths=tgt)
    assert h_a.shape == (1, 32, _StubNemoEncoder.D)
    assert h_a.abs().sum() > 0
    assert not torch.allclose(h_a, torch.zeros_like(h_a))
    assert not torch.allclose(h_a, h_b)


def test_train_one_epoch_encoder_frozen_and_no_side(tmp_path: Path):
    import wave

    from duplex_data.temporal.dataset import TSRFrameDataset, collate_tsr_frames
    from duplex_data.temporal.h_asr import encode_wavs_to_h_asr
    from duplex_data.temporal.train_asr import train_one_epoch

    rep = _sample_tsr()
    wav_path = tmp_path / "clip.wav"
    n = 16000 * 1250 // 1000
    pcm = (torch.randn(n).clamp(-1, 1).numpy() * 32767.0).astype("int16")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    d = rep.to_dict()
    d["meta"] = {"audio_path": str(wav_path)}
    (tmp_path / "input_tsr.json").write_text(
        __import__("json").dumps(d, indent=2), encoding="utf-8"
    )
    ds = TSRFrameDataset(tmp_path, frame_ms=40)
    item = ds[0]
    assert "features" not in item
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        collate_fn=lambda b: collate_tsr_frames(b, asr_dim=_StubNemoEncoder.D),
    )
    batch = next(iter(loader))
    assert "features" not in batch
    encoder = _StubNemoEncoder()
    h = encode_wavs_to_h_asr(
        encoder, batch["wav"], batch["wav_lengths"],
        target_t=int(batch["h_asr"].size(1)),
        target_lengths=batch["lengths"],
    )
    assert not torch.allclose(h, torch.zeros_like(h))
    model = TemporalFusionModule(asr_dim=_StubNemoEncoder.D, d_model=32)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = train_one_epoch(model, loader, optim, "cpu", encoder=encoder)
    assert "total" in stats
    out = model(h, lengths=batch["lengths"])
    assert "pause_logit" in out


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
