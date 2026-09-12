"""Local open-source Sarvam LLM dialogue scripts (default: sarvam-30b; no Sarvam API)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from .topics import ScenarioDraw

logger = logging.getLogger("duplex_data.sarvam_scripts")

DEFAULT_LLM = "sarvamai/sarvam-30b"

LANG_NAME = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
}


@dataclass
class Turn:
    role: str  # agent | user
    text: str


@dataclass
class DialogueScript:
    domain: str
    scenario: str
    lang: str
    agent_role: str
    user_role: str
    llm: str
    turns: list[Turn]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


_llm_bundle: dict[str, Any] = {}


TRANSFORMERS_MIN_HINT = (
    "sarvamai/sarvam-30b needs a newer `transformers` "
    "(missing ALL_ATTENTION_FUNCTIONS).\n\n"
    "In your TTS/eval venv run:\n"
    '  pip install -U "transformers>=4.57.0" accelerate\n\n'
    "Then retry. If Parler breaks after the upgrade, either:\n"
    "  • use a separate LLM venv for --dry-run-scripts, then --scripts-dir for TTS, or\n"
    "  • fall back: --llm-model sarvamai/sarvam-m\n"
)


def load_llm(
    model_id: str = DEFAULT_LLM,
    *,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load tokenizer + causal LM once (cached)."""
    key = f"{model_id}|{device}|4bit={load_in_4bit}"
    if key in _llm_bundle:
        return _llm_bundle[key]["tok"], _llm_bundle[key]["model"]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # sarvam-30b remote code imports ALL_ATTENTION_FUNCTIONS (newer transformers)
    if "sarvam-30b" in model_id.lower():
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"{TRANSFORMERS_MIN_HINT}\n"
                f"Installed transformers=={getattr(transformers, '__version__', '?')}\n"
                f"Underlying error: {exc}"
            ) from exc

    logger.info(
        "Loading open-source LLM %s (transformers %s) …",
        model_id,
        getattr(transformers, "__version__", "?"),
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as exc:
            raise SystemExit(
                "bitsandbytes required for --llm-load-in-4bit: pip install bitsandbytes"
            ) from exc
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ImportError as exc:
        if "ALL_ATTENTION_FUNCTIONS" in str(exc):
            raise SystemExit(
                f"{TRANSFORMERS_MIN_HINT}\nUnderlying error: {exc}"
            ) from exc
        raise
    model.eval()
    _llm_bundle[key] = {"tok": tok, "model": model}
    return tok, model


def _build_prompt(draw: ScenarioDraw, lang: str, max_turns: int) -> str:
    lang_name = LANG_NAME.get(lang, lang)
    slang_hint = {
        "te": (
            "Telugu phone slang / natural speech: అండి, గా, కదా, అసలు, ఏంటి, చూడు, "
            "సరేలే, అవునండి, మరి, అలాగా, బాగుంది, ప్రాబ్లం, కన్ఫర్మ్, ఆర్డర్, "
            "రిచార్జ్, ఓటిపి — mix English loanwords in Telugu script when people actually say them "
            "(కాంటాక్ట్, అడ్రెస్, నంబరు, పేమెంట్)."
        ),
        "hi": (
            "Hinglish phone slang: हाँ जी, अच्छा, ठीक है, बस, यार, मतलब, अरे, बिलकुल, "
            "ना, सीरियसली, प्रोब्लम, कन्फर्म, ऑर्डर, रिचार्ज, ओटीपी — Devanagari with natural English loans."
        ),
        "ta": (
            "Spoken Tamil phone style with natural English loans people actually say "
            "(order, OTP, recharge, problem) in Tamil script where natural; casual particles."
        ),
        "kn": (
            "Spoken Kannada phone style with natural English loans; casual polite forms."
        ),
    }.get(lang, "Use everyday Indian phone slang and light English loanwords as spoken.")

    return f"""You write REAL Indian phone-call dialogues — not textbook / news / formal letters.

Domain: {draw.domain}
Scenario: {draw.scenario}
Agent role: {draw.agent_role} (support / office side — polite but Indian call-centre tone)
User role: {draw.user_role} (caller — more casual, emotional, slangy)

Language: **{lang_name}** in native script, **spoken India accent vibe** (how people actually talk on calls).

{slang_hint}

Style rules:
- Sound like a real India mobile call: interruptions vibe, short turns, repeats, soft fillers.
- Fillers / backchannels when natural: హ్మ్, అవును, సరే, ఓకే, आच्छा, हाँ, hmm, okay (written in the dialogue language script when possible).
- Light code-mix is GOOD (Indian English words inside the local language), NOT pure English paragraphs.
- Agent: helpful Indian support tone (sir/madam / అండి / जी) — warm, not robotic corporate English.
- User: natural slang, mild frustration or friendliness; not perfect grammar.
- Exactly alternate turns starting with **agent** then **user**.
- Between 6 and {max_turns} turns total.
- No stage directions, no *asterisks*, no URLs, no bullet lists, no “Narrator:”.
- ₹ amounts, OTP, phone numbers in spoken form OK.

Return ONLY valid JSON:
{{"turns":[{{"role":"agent","text":"..."}},{{"role":"user","text":"..."}}]}}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if any
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _normalize_turns(raw: Any, max_turns: int) -> list[Turn]:
    if not isinstance(raw, dict) or "turns" not in raw:
        raise ValueError("JSON missing turns")
    turns_in = raw["turns"]
    if not isinstance(turns_in, list) or not turns_in:
        raise ValueError("turns must be a non-empty list")
    out: list[Turn] = []
    for item in turns_in[:max_turns]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role in ("assistant", "agent", "support", "bot", "hr"):
            role = "agent"
        elif role in ("user", "customer", "caller", "candidate", "patient"):
            role = "user"
        else:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        out.append(Turn(role=role, text=text))
    if len(out) < 4:
        raise ValueError(f"need >=4 turns, got {len(out)}")
    return out[:max_turns]


def generate_dialogue(
    draw: ScenarioDraw,
    *,
    lang: str = "te",
    max_turns: int = 12,
    model_id: str = DEFAULT_LLM,
    device: str = "cuda",
    load_in_4bit: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.85,
) -> DialogueScript:
    import torch

    tok, model = load_llm(model_id, device=device, load_in_4bit=load_in_4bit)
    user_prompt = _build_prompt(draw, lang, max_turns)
    messages = [
        {
            "role": "system",
            "content": (
                "You write naturalistic Indian phone dialogues with local slang and "
                "light code-mixing. Never write formal essays. Output JSON only."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = user_prompt

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=getattr(tok, "eos_token_id", None),
        )
    gen = out_ids[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(gen, skip_special_tokens=True)
    logger.info("LLM raw (trim): %s", text[:200].replace("\n", " "))

    last_err: Exception | None = None
    for _ in range(2):
        try:
            data = _extract_json(text)
            turns = _normalize_turns(data, max_turns)
            # Telugu greeting fix if we wrongly inserted
            if turns and turns[0].role != "agent":
                turns.insert(0, Turn(role="agent", text=_agent_greeting(lang)))
            return DialogueScript(
                domain=draw.domain,
                scenario=draw.scenario,
                lang=lang,
                agent_role=draw.agent_role,
                user_role=draw.user_role,
                llm=model_id,
                turns=turns,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            # one repair pass
            repair = (
                "Fix into valid JSON only with key turns "
                f"(agent/user). Previous output:\n{text[:1500]}"
            )
            messages2 = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": repair},
            ]
            if hasattr(tok, "apply_chat_template"):
                prompt = tok.apply_chat_template(
                    messages2, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = repair
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tok, "eos_token_id", None),
                )
            gen = out_ids[0][inputs["input_ids"].shape[-1] :]
            text = tok.decode(gen, skip_special_tokens=True)

    raise RuntimeError(f"Failed to parse dialogue JSON: {last_err}") from last_err


def _agent_greeting(lang: str) -> str:
    return {
        "te": "నమస్కారం అండి, ఎలా సహాయం చేయాలి చెప్పండి?",
        "hi": "नमस्ते जी, बताइए कैसे मदद करूँ?",
        "ta": "வணக்கம்ங்க, எப்படி உதவலாம் சொல்லுங்க?",
        "kn": "ನಮಸ್ಕಾರ ಅಂಡಿ, ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ ಹೇಳಿ?",
    }.get(lang, "Hello ji, how can I help you?")


def script_from_dict(data: dict) -> DialogueScript:
    turns = [
        Turn(role=str(t["role"]).lower(), text=str(t["text"]).strip())
        for t in data["turns"]
        if str(t.get("text", "")).strip()
    ]
    return DialogueScript(
        domain=str(data.get("domain", "unknown")),
        scenario=str(data.get("scenario", "")),
        lang=str(data.get("lang", "te")),
        agent_role=str(data.get("agent_role", "agent")),
        user_role=str(data.get("user_role", "user")),
        llm=str(data.get("llm", DEFAULT_LLM)),
        turns=turns,
    )
