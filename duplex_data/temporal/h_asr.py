"""Frozen Bellu IndicConformer encoder → H_asr on the TSR 40 ms grid.

Live Bellu ASR (``Bellu-Voice-1/bellu_voice/asr.py``) loads:

    hf_hub_download(<lang repo>/*.nemo)
    EncDecHybridRNNTCTCBPEModel.restore_from(...)
    cur_decoder = "ctc"; freeze(); eval()

then transcribes 16 kHz mono. This module uses the **same** Hugging Face
``.nemo`` repos and ``duplex_data.asr.load_indic_conformer`` (identical
restore/CTC/freeze path). Hidden states come from that model's
``preprocessor`` + ``encoder`` on 16 kHz waveforms (the same frontend
``transcribe()`` uses after loading the wav).

Encoder geometry (IndicConformer **large** hybrid, 120M encoder)
----------------------------------------------------------------
Pinned from the official model card (17 Conformer blocks, dim 512) and a
NeMo-extracted conversion of
``ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large``:

- ``D`` / ``encoder_dim`` = **512**
- Mel hop = **10 ms** (16 kHz, hop 160 samples, window 25 ms / 400)
- Subsampling = **4** (striding) → native encoder stride **40 ms**
- ``T_enc ≈ duration_s / 0.040`` (same grid as TSR ``ceil(duration_ms / 40)``)

If a restore ever emits a different ``T_enc``, linear-resample onto the TSR
grid (near-identity when the hop already matches).

Time alignment
--------------
TSR targets use ``T_tsr = ceil(duration_ms / 40)``.

For each utterance ``b``:

1. Run preprocessor+encoder → ``H_enc[b, :T_enc_b, D]`` with ``D=512``.
2. Linear-interpolate along time to ``T_tsr_b`` frames.
3. Pad to the batch max ``T_tsr``.

So ``H_asr[b, t]`` is the encoder content at TSR frame ``t``
(``t * 40 ms … (t+1) * 40 ms``).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("duplex_data.temporal.h_asr")

# Exact map from Bellu-Voice-1/bellu_voice/lang.py INDIC_CONFORMER.
BELLU_INDIC_CONFORMER = {
    "hi": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
    "te": "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
    "ta": "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
    "kn": "ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large",
}

# Native geometry of the Bellu large .nemo encoder (see module docstring).
ENCODER_DIM = 512
MEL_HOP_MS = 10
SUBSAMPLING_FACTOR = 4
NATIVE_STRIDE_MS = MEL_HOP_MS * SUBSAMPLING_FACTOR  # 40
SAMPLE_RATE = 16_000
MEL_HOP_SAMPLES = SAMPLE_RATE * MEL_HOP_MS // 1000  # 160


def load_frozen_encoder(device: str, language: str = "hi"):
    """Load the same frozen .nemo IndicConformer Bellu uses for ``language``."""
    lang = (language or "hi").lower().split("-")[0]
    repo = BELLU_INDIC_CONFORMER.get(lang)
    if repo is None:
        logger.error(
            "No Bellu IndicConformer pack for lang=%s; supported %s",
            lang,
            sorted(BELLU_INDIC_CONFORMER),
        )
        return None
    try:
        from duplex_data.asr import load_indic_conformer

        model = load_indic_conformer(repo, device=device)
    except Exception as exc:
        logger.warning("Could not load Bellu IndicConformer %s: %s", repo, exc)
        return None
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("Frozen Bellu IndicConformer encoder %s", repo)
    return model


def infer_asr_dim(encoder: Any, device: str, asr_dim_fallback: int) -> int:
    wav = torch.zeros(1, 16000, device=device)
    lengths = torch.tensor([16000], device=device, dtype=torch.long)
    tgt = torch.tensor([8], device=device, dtype=torch.long)
    with torch.no_grad():
        h = encode_wavs_to_h_asr(encoder, wav, lengths, target_t=8, target_lengths=tgt)
    if h is None or h.numel() == 0:
        return asr_dim_fallback
    return int(h.size(-1))


@torch.no_grad()
def encode_wavs_to_h_asr(
    encoder: Any,
    wavs: torch.Tensor,
    wav_lengths: torch.Tensor,
    target_t: int,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[B, target_t, D]`` encoder frames aligned to the TSR grid.

    ``target_t`` is the padded TSR length. ``target_lengths[b]`` is valid
    TSR frames for item ``b`` (``ceil(duration_ms / 40)``).
    """
    device = wavs.device
    bsz = int(wavs.size(0))
    target_t = max(1, int(target_t))
    hidden, enc_lens, layout = _forward_encoder(encoder, wavs, wav_lengths)
    if layout == "bdt":
        hidden = hidden.transpose(1, 2).contiguous()
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected encoder 3D tensor, got {tuple(hidden.shape)}")
    if target_lengths is None:
        target_lengths = torch.full(
            (bsz,), target_t, device=hidden.device, dtype=torch.long
        )
    aligned = _align_to_tsr_grid(hidden, enc_lens, target_t, target_lengths)
    if aligned.size(0) != bsz:
        aligned = aligned.expand(bsz, -1, -1).contiguous()
    return aligned.to(device=device, dtype=torch.float32)


def _forward_encoder(
    encoder: Any, wavs: torch.Tensor, wav_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Same NeMo frontend as live ``model.transcribe``: preprocessor then encoder."""
    if not (hasattr(encoder, "preprocessor") and hasattr(encoder, "encoder")):
        raise RuntimeError(
            "Encoder must be a NeMo ASR model with preprocessor+encoder "
            "(Bellu EncDecHybridRNNTCTCBPEModel)."
        )
    proc, proc_len = encoder.preprocessor(input_signal=wavs, length=wav_lengths)
    encoded, enc_len = encoder.encoder(audio_signal=proc, length=proc_len)
    if not torch.is_tensor(enc_len):
        enc_len = torch.full(
            (encoded.size(0),), encoded.size(-1), device=encoded.device, dtype=torch.long
        )
    return encoded, enc_len.to(dtype=torch.long), "bdt"


def _align_to_tsr_grid(
    hidden_btd: torch.Tensor,
    enc_lens: torch.Tensor,
    target_t: int,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Linear time-resample each utterance from T_enc_b to T_tsr_b, then pad."""
    bsz, t_enc, dim = hidden_btd.shape
    out = hidden_btd.new_zeros(bsz, target_t, dim)
    for b in range(bsz):
        te = max(1, min(int(enc_lens[b].item()), t_enc))
        tt = max(1, min(int(target_lengths[b].item()), target_t))
        src = hidden_btd[b : b + 1, :te, :].transpose(1, 2)
        dst = F.interpolate(src, size=tt, mode="linear", align_corners=False)
        out[b, :tt] = dst.transpose(1, 2).squeeze(0)
    return out
