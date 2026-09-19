"""Temporal fusion / TTS conditioner training (ASR → TTS → LLM later)."""

from .features import pack_tsr_features, FEATURE_NAMES, DEFAULT_FRAME_MS
from .fusion import TemporalFusionModule, fusion_loss
from .h_asr import encode_wavs_to_h_asr, load_frozen_encoder
from .tts_conditioner import TemporalEncoder

__all__ = [
    "pack_tsr_features",
    "FEATURE_NAMES",
    "DEFAULT_FRAME_MS",
    "TemporalFusionModule",
    "fusion_loss",
    "encode_wavs_to_h_asr",
    "load_frozen_encoder",
    "TemporalEncoder",
]
