"""TSR → Indic Parler-TTS conditioning (prompt + description)."""

from __future__ import annotations

from duplex_data.parler_tts import AGENT_STYLES
from duplex_data.tsr.schema import TemporalSpeechRep


def tsr_to_parler_inputs(
    rep: TemporalSpeechRep,
    *,
    speaker_style: str = "phone",
) -> tuple[str, str]:
    """Return (prompt_text, description) for parler_tts.synthesize."""
    words = [e.text for e in rep.words() if e.text]
    prompt = " ".join(words) if words else (rep.transcript or "").strip()
    if not prompt:
        raise ValueError("Output TSR has no words/transcript for TTS prompt")

    base = AGENT_STYLES.get(speaker_style, AGENT_STYLES["phone"])

    cues: list[str] = []
    if any(e.kind == "emphasis" for e in rep.events):
        cues.append("with clear word emphasis")
    if any(e.kind == "pause" for e in rep.events):
        cues.append("with brief natural pauses")
    pitch_vals = {
        str(e.value) for e in rep.events if e.kind == "pitch" and e.value
    }
    if any("falling" in v or "warm" in v for v in pitch_vals):
        cues.append("warm falling intonation at the end")
    elif "rising" in pitch_vals:
        cues.append("slightly rising intonation")
    if any(e.kind == "rate" and e.value == "faster" for e in rep.events):
        cues.append("slightly faster pace")
    elif any(e.kind == "rate" and e.value == "slower" for e in rep.events):
        cues.append("slightly slower pace")
    if any(e.kind == "energy" and e.value == "softer" for e in rep.events):
        cues.append("softer delivery")
    elif any(e.kind == "energy" and e.value == "stronger" for e in rep.events):
        cues.append("stronger energetic delivery")

    if cues:
        description = f"{base} The speech is {', '.join(cues)}."
    else:
        description = base
    return prompt, description
