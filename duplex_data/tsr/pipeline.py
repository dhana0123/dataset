"""End-to-end: Audio → TSR → LLM → TSR → TTS → Audio."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from duplex_data.tsr.from_audio import audio_to_tsr
from duplex_data.tsr.llm import generate_tsr
from duplex_data.tsr.serialize import save_text, tsr_to_text
from duplex_data.tsr.to_tts import tsr_to_parler_inputs

logger = logging.getLogger("duplex_data.tsr.pipeline")


def run(
    audio_path: Path | str,
    out_dir: Path | str,
    *,
    language: str = "en",
    asr_backend: str = "whisper",
    device: str = "cpu",
    asr_model: str | None = None,
    transcript_path: Path | str | None = None,
    align_model: str | None = None,
    llm_backend: str = "template",
    llm_model: str | None = None,
    llm_load_in_4bit: bool = False,
    speaker_style: str = "phone",
    skip_tts: bool = False,
) -> dict:
    """Run smallest E2E pipeline. Returns paths dict."""
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1/4 audio → input TSR")
    input_tsr = audio_to_tsr(
        audio_path,
        language=language,
        asr_backend=asr_backend,
        device=device,
        asr_model=asr_model,
        transcript_path=transcript_path,
        align_model=align_model,
    )
    in_json = out_dir / "input_tsr.json"
    in_txt = out_dir / "input_tsr.txt"
    input_tsr.save(in_json)
    save_text(input_tsr, in_txt)
    logger.info("Input TSR:\n%s", tsr_to_text(input_tsr))

    logger.info("2/4 LLM → output TSR (%s)", llm_backend)
    output_tsr = generate_tsr(
        input_tsr,
        backend=llm_backend,
        device=device,
        load_in_4bit=llm_load_in_4bit,
        model_id=llm_model,
    )
    out_json = out_dir / "output_tsr.json"
    out_txt = out_dir / "output_tsr.txt"
    output_tsr.save(out_json)
    save_text(output_tsr, out_txt)
    logger.info("Output TSR:\n%s", tsr_to_text(output_tsr))

    result = {
        "input_tsr_json": str(in_json),
        "input_tsr_txt": str(in_txt),
        "output_tsr_json": str(out_json),
        "output_tsr_txt": str(out_txt),
        "reply_wav": None,
    }

    if skip_tts:
        logger.info("3–4/4 TTS skipped")
        return result

    logger.info("3/4 TSR → Parler conditioning")
    prompt, description = tsr_to_parler_inputs(output_tsr, speaker_style=speaker_style)
    (out_dir / "tts_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (out_dir / "tts_description.txt").write_text(description + "\n", encoding="utf-8")

    logger.info("4/4 TTS synthesize")
    from duplex_data.parler_tts import synthesize

    audio, sr = synthesize(prompt, description, device=device)
    reply = out_dir / "reply.wav"
    sf.write(str(reply), audio, sr)
    result["reply_wav"] = str(reply)
    logger.info("Wrote %s", reply)
    return result
