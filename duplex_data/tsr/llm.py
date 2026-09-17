"""LLM adapters: Input TSR → Output TSR (same schema)."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep
from duplex_data.tsr.serialize import text_to_tsr, tsr_to_text

logger = logging.getLogger("duplex_data.tsr.llm")

# AI4Bharat 7B — Hindi (+English) only
INDIC_7B_MODEL = "ai4bharat/Airavata"
# Sarvam-30B — all 22 scheduled Indian languages (+ English)
INDIC_22_MODEL = "sarvamai/sarvam-30b"
PARAM2_MODEL = "bharatgenai/Param2-17B-A2.4B-Thinking"


class TSRGenerator(Protocol):
    def generate_tsr(self, input_tsr: TemporalSpeechRep) -> TemporalSpeechRep: ...


def _words_to_even_tsr(
    reply: str,
    *,
    language: str,
    meta: dict,
    ms_per_word: int = 280,
    pause_every: int = 4,
    pause_ms: int = 120,
) -> TemporalSpeechRep:
    """Build a simple time-ordered output TSR from reply text."""
    words = [w for w in re.split(r"\s+", reply.strip()) if w]
    if not words:
        words = ["Okay"]
    events: list[TSREvent] = []
    t = 0
    for i, w in enumerate(words):
        if i > 0 and i % pause_every == 0:
            events.append(
                TSREvent(start_ms=t, end_ms=t + pause_ms, kind="pause", source="llm")
            )
            t += pause_ms
        dur = max(180, min(450, ms_per_word + 10 * len(w)))
        events.append(
            TSREvent(
                start_ms=t,
                end_ms=t + dur,
                kind="word",
                text=w.strip(".,!?;:\"'"),
                source="llm",
            )
        )
        t += dur
    # light emphasis on a mid content word
    content = [e for e in events if e.kind == "word"]
    if len(content) >= 3:
        mid = content[len(content) // 2]
        events.append(
            TSREvent(
                start_ms=mid.start_ms,
                end_ms=mid.end_ms,
                kind="emphasis",
                text=mid.text,
                value="strong",
                source="llm",
            )
        )
    events.append(
        TSREvent(
            start_ms=t,
            end_ms=t,
            kind="boundary",
            value="completing",
            source="llm",
        )
    )
    out = TemporalSpeechRep(
        direction="output",
        language=language,
        transcript=" ".join(w.strip(".,!?;:\"'") for w in words),
        duration_ms=t,
        events=events,
        meta=meta,
    )
    out.sort_events()
    return out


class TemplateLLM:
    """Offline stub: polite English reply with a fixed temporal skeleton."""

    REPLY = "That's great to hear."

    def generate_tsr(self, input_tsr: TemporalSpeechRep) -> TemporalSpeechRep:
        words = [
            ("That's", 0, 250),
            ("great", 250, 500),
            ("to", 600, 750),
            ("hear", 750, 1000),
        ]
        events: list[TSREvent] = []
        for text, s, e in words:
            events.append(
                TSREvent(
                    start_ms=s,
                    end_ms=e,
                    kind="word",
                    text=text,
                    source="llm",
                )
            )
        events.append(
            TSREvent(
                start_ms=250,
                end_ms=500,
                kind="emphasis",
                text="great",
                value="strong",
                source="llm",
            )
        )
        events.append(
            TSREvent(start_ms=500, end_ms=600, kind="pause", source="llm")
        )
        events.append(
            TSREvent(
                start_ms=750,
                end_ms=1000,
                kind="pitch",
                text="hear",
                value="warm/falling",
                source="llm",
            )
        )
        events.append(
            TSREvent(
                start_ms=1000,
                end_ms=1000,
                kind="boundary",
                value="completing",
                source="llm",
            )
        )
        out = TemporalSpeechRep(
            direction="output",
            language=input_tsr.language,
            transcript=self.REPLY,
            duration_ms=1000,
            events=events,
            meta={
                "llm": "template",
                "input_transcript": input_tsr.transcript,
                "input_text_preview": tsr_to_text(input_tsr)[:500],
            },
        )
        out.sort_events()
        return out


class HfCausalLLM:
    """Any HF causal LM via load_llm → reply text → output TSR."""

    def __init__(
        self,
        *,
        model_id: str,
        tag: str = "hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 80,
    ):
        self.model_id = model_id
        self.tag = tag
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens

    def _generate_reply(self, user_text: str, language: str) -> str:
        from duplex_data.sarvam_scripts import LANG_NAME, load_llm

        tok, model, resolved = load_llm(
            self.model_id,
            device=self.device,
            load_in_4bit=self.load_in_4bit,
        )
        self.model_id = resolved
        lang = (language or "en").lower()
        lang_name = LANG_NAME.get(lang, lang)
        prompt = (
            "You are a helpful Indian voice assistant. "
            f"Reply in {lang_name} (language code '{lang}') with ONE short "
            "spoken sentence (no bullets, no quotes, no explanation).\n\n"
            f"User said: {user_text.strip() or '(silence)'}\n"
            "Assistant:"
        )
        import torch

        inputs = tok(prompt, return_tensors="pt")
        # device_map=auto models: move inputs to first param device
        try:
            first = next(model.parameters()).device
            inputs = {k: v.to(first) for k, v in inputs.items()}
        except Exception:
            pass
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=getattr(tok, "pad_token_id", None)
                or getattr(tok, "eos_token_id", None),
            )
        gen = out_ids[0, inputs["input_ids"].shape[-1] :]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        # keep first line / sentence
        text = re.split(r"[\n\r]+", text)[0].strip()
        text = re.split(r"(?<=[.!?।])\s+", text)[0].strip()
        if not text:
            text = "Okay, I understand."
        return text

    def generate_tsr(self, input_tsr: TemporalSpeechRep) -> TemporalSpeechRep:
        reply = self._generate_reply(input_tsr.transcript, input_tsr.language)
        logger.info("%s reply: %s", self.tag, reply)
        return _words_to_even_tsr(
            reply,
            language=input_tsr.language,
            meta={
                "llm": self.tag,
                "model_id": self.model_id,
                "input_transcript": input_tsr.transcript,
            },
        )


class AiravataLLM(HfCausalLLM):
    """7B Hindi (+English) Indian LLM → reply text → output TSR."""

    def __init__(
        self,
        *,
        model_id: str = INDIC_7B_MODEL,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 80,
    ):
        super().__init__(
            model_id=model_id,
            tag="airavata",
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )


class Indic22LLM(HfCausalLLM):
    """Sarvam-30B (22 scheduled Indic langs + English) → output TSR."""

    def __init__(
        self,
        *,
        model_id: str = INDIC_22_MODEL,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 80,
    ):
        super().__init__(
            model_id=model_id,
            tag="indic-22",
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )


def generate_tsr(
    input_tsr: TemporalSpeechRep,
    *,
    backend: str = "template",
    device: str = "cpu",
    load_in_4bit: bool = False,
    model_id: str | None = None,
) -> TemporalSpeechRep:
    """Dispatch LLM: template | parse_echo | indic-22 | airavata | …"""
    backend = (backend or "template").lower().replace("_", "-")
    if backend in {"template", "stub", "offline"}:
        return TemplateLLM().generate_tsr(input_tsr)
    if backend == "parse_echo":
        return text_to_tsr(
            tsr_to_text(input_tsr),
            direction="output",
            language=input_tsr.language,
            meta={"llm": "parse_echo"},
        )
    if backend in {"indic-22", "sarvam-30b", "sarvam-22", "indic22"}:
        mid = model_id or INDIC_22_MODEL
        return Indic22LLM(
            model_id=mid,
            device=device,
            load_in_4bit=load_in_4bit,
        ).generate_tsr(input_tsr)
    if backend in {"param2", "param2-17b"}:
        mid = model_id or PARAM2_MODEL
        return HfCausalLLM(
            model_id=mid,
            tag="param2",
            device=device,
            load_in_4bit=load_in_4bit,
        ).generate_tsr(input_tsr)
    if backend in {"airavata", "indic-7b", "airavata-7b"}:
        mid = model_id or INDIC_7B_MODEL
        return AiravataLLM(
            model_id=mid,
            device=device,
            load_in_4bit=load_in_4bit,
        ).generate_tsr(input_tsr)
    raise ValueError(
        f"Unknown LLM backend {backend!r}; use template, parse_echo, "
        "indic-22/sarvam-30b (22 langs), param2, or airavata/indic-7b (hi)"
    )
