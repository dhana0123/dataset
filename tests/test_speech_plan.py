"""Unit tests for speech-plan schema + derive + nonverbal vocab."""

from __future__ import annotations

from pathlib import Path

from duplex_data.speech_plan.acoustics import AcousticTracks
from duplex_data.speech_plan.derive import (
    build_lanes,
    derive_pauses,
    derive_rate,
    words_to_content_events,
)
from duplex_data.speech_plan.schema import Event, GlobalState, SpeechPlan
from duplex_data.speech_plan.vocab import (
    indic_forms,
    list_nonverbal_event_ids,
    nonverbal_dropdown_options,
)


def test_speech_plan_roundtrip_keeps_nonverbal(tmp_path: Path):
    plan = SpeechPlan(
        audio_path="clip.wav",
        duration=1.5,
        language="hi",
        global_state=GlobalState(value="reassuring", source="human"),
        lanes={
            "content": [Event("WORD", 0.0, 0.3, "hello", source="asr")],
            "pitch": [Event("PITCH", 0.2, 0.8, "rising", source="auto")],
            "energy": [],
            "rate": [],
            "emphasis": [],
            "pause": [],
            "boundary": [],
            "nonverbal": [
                Event("NONVERBAL", 0.0, 0.1, "हम्म", source="human"),
            ],
        },
        tracks={"f0_hz": [100.0]},
        asr={"transcript": "hello"},
    )
    assert plan.lanes["nonverbal"][0].value == "हम्म"
    assert plan.nonverbal[0]["value"] == "हम्म"

    path = tmp_path / "speech_plan.json"
    plan.save(path)
    loaded = SpeechPlan.load(path)
    assert loaded.audio_path == "clip.wav"
    assert loaded.global_state.value == "reassuring"
    assert loaded.lanes["content"][0].value == "hello"
    assert loaded.lanes["pitch"][0].value == "rising"
    assert loaded.lanes["nonverbal"][0].value == "हम्म"
    assert loaded.nonverbal[0]["value"] == "हम्म"


def test_indic_native_in_vocab():
    from duplex_data.speech_plan.vocab import load_nonverbal_vocab

    load_nonverbal_vocab.cache_clear()
    ids = list_nonverbal_event_ids()
    assert "laugh" in ids
    assert "speech_laugh" in ids
    assert "surprised_gasp" in ids
    assert "huh_question" in ids
    assert "hmm_long" in ids
    assert "guffaw" in ids
    # Direct native-script transcripts (not Latin-only)
    assert "हम्म" in ids
    assert "हाँ" in ids
    assert "हाँ?" in ids
    assert "वाह" in ids
    assert "हाहा" in ids
    assert "హ్మ్" in ids
    assert "అయ్యో" in ids
    assert "ஐயோ" in ids
    forms = indic_forms()
    assert any(f.get("annotation_value") == "ഹ്മ്" for f in forms)
    assert any(f.get("annotation_value") == "उम्म" for f in forms)
    opts = nonverbal_dropdown_options()
    indic_opts = [o for o in opts if o.get("group") == "indic"]
    assert len(indic_opts) >= 40
    assert opts[0]["group"] == "indic"
    assert any(o["id"] == "ம்" for o in indic_opts)
    assert "questioning" in load_nonverbal_vocab()["categories"]



def test_derive_pauses_and_rate():
    words = [
        ("I", (0.0, 0.2)),
        ("really", (0.5, 0.9)),  # 300ms pause
        ("don't", (0.95, 1.05)),
        ("know", (1.1, 1.8)),  # longer → slower
    ]
    pauses = derive_pauses(words)
    assert len(pauses) == 1
    assert pauses[0].value == 300
    assert pauses[0].source == "auto"

    content = words_to_content_events(words)
    assert content[0].source == "asr"
    assert content[0].value == "I"

    rate = derive_rate(words)
    values = {e.value for e in rate}
    assert "slower" in values or "faster" in values


def test_build_lanes_empty_tracks():
    words = [("hi", (0.0, 0.3)), ("there", (0.5, 0.8))]
    tracks = AcousticTracks(times=[], f0_hz=[], intensity_db=[], rms=[])
    lanes = build_lanes(words, tracks, duration=1.0)
    assert len(lanes["content"]) == 2
    assert lanes["nonverbal"] == []
    assert any(e.type == "PAUSE" for e in lanes["pause"])
    assert any(e.type == "BOUNDARY" for e in lanes["boundary"])
