"""Unit tests for Temporal Speech Representation (no GPU / no WhisperX required)."""

from __future__ import annotations

from duplex_data.speech_plan.acoustics import AcousticTracks
from duplex_data.tsr.from_audio import words_tracks_to_tsr
from duplex_data.tsr.llm import TemplateLLM, generate_tsr
from duplex_data.tsr.schema import TSREvent, TemporalSpeechRep, sec_to_ms
from duplex_data.tsr.serialize import text_to_tsr, tsr_to_text
from duplex_data.tsr.to_tts import tsr_to_parler_inputs


def test_sec_to_ms_and_json_roundtrip(tmp_path):
    assert sec_to_ms(0.2) == 200
    assert sec_to_ms(1.05) == 1050
    rep = TemporalSpeechRep(
        direction="input",
        language="en",
        transcript="Yeah I like it",
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
    path = tmp_path / "tsr.json"
    rep.save(path)
    loaded = TemporalSpeechRep.load(path)
    assert loaded.duration_ms == 1250
    assert loaded.direction == "input"
    assert len(loaded.events) == len(rep.events)
    assert loaded.words()[0].text == "Yeah"


def test_serialize_parse_yeah_example():
    fixture = """DIRECTION: input
LANGUAGE: en
TRANSCRIPT: Yeah... I really like it.
DURATION_MS: 1250
EVENTS:
0-200ms     Yeah       word|backchannel
200-500ms   PAUSE      pause|hesitation
500-600ms   I          word
600-850ms   really     word|emphasis|strong
850-1050ms  like       word
1050-1250ms it         word|pitch|falling
1250ms      BOUNDARY       boundary|completing
"""
    rep = text_to_tsr(fixture, direction="input", language="en")
    assert rep.transcript.startswith("Yeah")
    assert rep.duration_ms == 1250
    kinds = {e.kind for e in rep.events}
    assert "word" in kinds
    assert "pause" in kinds
    assert "hesitation" in kinds
    assert "backchannel" in kinds
    assert "emphasis" in kinds
    assert "pitch" in kinds
    assert "boundary" in kinds
    words = [e.text for e in rep.words()]
    assert words == ["Yeah", "I", "really", "like", "it"]

    # Round-trip through serializer should remain parseable
    text2 = tsr_to_text(rep)
    rep2 = text_to_tsr(text2, direction="input")
    assert [e.text for e in rep2.words()] == words


def test_words_tracks_to_tsr_ordered():
    words = [
        ("Yeah", (0.0, 0.2)),
        ("I", (0.5, 0.6)),
        ("really", (0.6, 0.85)),
        ("like", (0.85, 1.05)),
        ("it", (1.05, 1.25)),
    ]
    # Empty tracks → still get words + pause + boundary
    tracks = AcousticTracks([], [], [], [])
    rep = words_tracks_to_tsr(
        words, tracks, duration_sec=1.25, language="en", direction="input"
    )
    assert rep.direction == "input"
    assert rep.duration_ms == 1250
    assert [e.text for e in rep.words()] == ["Yeah", "I", "really", "like", "it"]
    assert any(e.kind == "pause" for e in rep.events)
    assert any(e.kind == "hesitation" for e in rep.events)  # 300ms gap
    assert any(e.kind == "backchannel" for e in rep.events)
    assert any(e.kind == "boundary" for e in rep.events)
    # Temporal order
    starts = [e.start_ms for e in rep.events]
    assert starts == sorted(starts)


def test_template_llm_and_to_tts():
    inp = TemporalSpeechRep(
        direction="input",
        language="en",
        transcript="Yeah I like it",
        duration_ms=1250,
        events=[TSREvent(0, 200, "word", text="Yeah", source="asr")],
    )
    out = generate_tsr(inp, backend="template")
    assert out.direction == "output"
    assert "great" in out.transcript.lower()
    prompt, desc = tsr_to_parler_inputs(out)
    assert "great" in prompt.lower() or "That's" in prompt or "That" in prompt
    assert len(desc) > 20
    assert TemplateLLM().REPLY in out.transcript or out.transcript == TemplateLLM().REPLY


def test_indic_7b_alias_and_words_helper():
    from duplex_data.sarvam_scripts import INDIC_7B_LLM, resolve_llm_model
    from duplex_data.tsr.llm import INDIC_7B_MODEL, _words_to_even_tsr

    assert INDIC_7B_MODEL == "ai4bharat/Airavata"
    assert resolve_llm_model("indic-7b") == INDIC_7B_LLM
    assert resolve_llm_model("airavata") == "ai4bharat/Airavata"
    rep = _words_to_even_tsr(
        "ठीक है मैं मदद कर सकता हूँ",
        language="hi",
        meta={"llm": "test"},
    )
    assert rep.direction == "output"
    assert len(rep.words()) >= 3
    assert any(e.kind == "boundary" for e in rep.events)


def test_indic_22_aliases_and_lang_table():
    from duplex_data.sarvam_scripts import (
        INDIC_22_LANGS,
        INDIC_22_LLM,
        LANG_NAME,
        PARAM2_LLM,
        resolve_llm_model,
    )
    from duplex_data.tsr.llm import INDIC_22_MODEL

    assert INDIC_22_MODEL == "sarvamai/sarvam-30b"
    assert resolve_llm_model("indic-22") == INDIC_22_LLM
    assert resolve_llm_model("sarvam-22") == INDIC_22_LLM
    assert resolve_llm_model("param2") == PARAM2_LLM
    assert len(INDIC_22_LANGS) == 22
    for code in INDIC_22_LANGS:
        assert code in LANG_NAME, code
    assert LANG_NAME["te"] == "Telugu"
    assert LANG_NAME["brx"] == "Bodo"
