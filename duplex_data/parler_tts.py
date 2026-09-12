"""Indic Parler-TTS helpers for synthetic duplex turns."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("duplex_data.parler_tts")

# Indian phone-call captions (Indic Parler). Accent + slangy delivery cues.
AGENT_STYLES: dict[str, str] = {
    "leela": (
        "Leela speaks Telugu with a natural Indian phone accent, warm and cheerful, "
        "fast pace, high-pitched, casual call-centre energy, slight Hinglish/Telugu mix feel, "
        "full of energy. The recording is very high quality with no background noise."
    ),
    "happy": (
        "Lalitha speaks Telugu with a clear Indian accent, happy and smiling, "
        "slightly higher pitch, fast conversational pace, friendly Indian customer-care tone, "
        "close microphone, very high quality audio."
    ),
    "phone": (
        "Lalitha speaks Telugu like a real Indian mobile phone call: casual Indian accent, "
        "slightly expressive, fast pace, natural slangy rhythm, close microphone, "
        "very high quality audio, almost no background noise."
    ),
    "formal": (
        "Lalitha speaks Telugu politely with an Indian professional accent, clear diction, "
        "fast pace, Indian customer-service tone (not American), close microphone, "
        "very high quality audio."
    ),
}

USER_STYLES: dict[str, str] = {
    "male": (
        "Prakash speaks Telugu like a real Indian caller on a mobile phone: natural Indian accent, "
        "casual slangy tone, slightly expressive, fast pace, close microphone, "
        "very high quality audio, almost no background noise."
    ),
    "phone": (
        "Prakash speaks Telugu like a real Indian mobile phone call: casual Indian accent, "
        "everyday slang rhythm, slightly expressive, fast pace, close microphone, "
        "very high quality audio, almost no background noise."
    ),
    "soft": (
        "Prakash speaks Telugu softly with a natural Indian accent, caring tone, "
        "fast conversational pace, close microphone, very high quality audio."
    ),
}

_bundle: dict[str, Any] = {}


def load_parler(device: str = "cuda") -> tuple[Any, Any, Any, int]:
    """Return (model, prompt_tokenizer, description_tokenizer, sample_rate)."""
    key = f"parler|{device}"
    if key in _bundle:
        b = _bundle[key]
        return b["model"], b["tok"], b["desc_tok"], b["sr"]

    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    repo = "ai4bharat/indic-parler-tts"
    logger.info("Loading %s on %s …", repo, device)
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(device)
    tok = AutoTokenizer.from_pretrained(repo)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = int(model.config.sampling_rate)
    _bundle[key] = {"model": model, "tok": tok, "desc_tok": desc_tok, "sr": sr}
    return model, tok, desc_tok, sr


def synthesize(
    text: str,
    description: str,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    """Synthesize one utterance. Returns mono float32 + sample rate."""
    import torch

    model, tok, desc_tok, sr = load_parler(device=device)
    desc = desc_tok(description, return_tensors="pt").to(device)
    prompt = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
    audio = gen.cpu().numpy().squeeze().astype(np.float32)
    return audio, sr


def resolve_agent_desc(style: str) -> str:
    key = style.strip().lower()
    if key not in AGENT_STYLES:
        raise SystemExit(
            f"Unknown --agent-style {style!r}. Choose: {', '.join(AGENT_STYLES)}"
        )
    return AGENT_STYLES[key]


def resolve_user_desc(voice: str) -> str:
    key = voice.strip().lower()
    if key not in USER_STYLES:
        raise SystemExit(
            f"Unknown --user-voice {voice!r}. Choose: {', '.join(USER_STYLES)}"
        )
    return USER_STYLES[key]
