"""Packer unit tests for duplex_data (run from data/: pytest tests)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from duplex_data import asr as asr_mod
from duplex_data.diarize import (
    DiarizationResult,
    DiarizationSkip,
    mono_to_stereo_from_diarization,
    pick_agent_user,
    validate_two_speakers,
)
from duplex_data.prepare import (
    alignments_payload,
    asr_lang_for_indicvoices_config,
    make_sample_dialogues,
    pack_stereo,
    parse_indicvoices_configs,
    write_inspect_samples,
)
from duplex_data.schema import INDICVOICES_FOCUS


def test_pack_stereo_shape():
    a = np.ones(100, dtype=np.float32)
    u = np.zeros(100, dtype=np.float32)
    s = pack_stereo(a, u, 24000)
    assert s.shape == (2, 100)


def test_indicvoices_focus_configs_and_asr_lang():
    assert parse_indicvoices_configs(None) == list(INDICVOICES_FOCUS)
    assert asr_lang_for_indicvoices_config("telugu", None) == "te"


def test_alignments_payload_includes_extraction():
    payload = alignments_payload(
        [("ok", (0.0, 0.2))],
        provenance={
            "transcript_backend": "whisper",
            "transcript_model": "m",
            "timer_backend": "whisper",
            "timer_model": "t",
            "language": "hi",
        },
    )
    assert payload["extraction"]["timer_backend"] == "whisper"
    assert payload["inner_monologue"]["user_text"] is False


def test_diarize_gate_and_skip():
    validate_two_speakers({"A": 1.0, "B": 0.8})
    try:
        validate_two_speakers({"A": 1.0, "B": 0.1})
        assert False, "expected DiarizationSkip"
    except DiarizationSkip:
        pass
    agent, user = pick_agent_user({"SPEAKER_00": 0.5, "SPEAKER_01": 2.0})
    assert agent == "SPEAKER_01"
    sr = 1000
    mono = np.ones(3000, dtype=np.float32)
    result = DiarizationResult(
        turns=[("SPEAKER_01", 0.0, 1.5), ("SPEAKER_00", 1.5, 3.0)],
        agent_id="SPEAKER_01",
        user_id="SPEAKER_00",
        durations={"SPEAKER_01": 1.5, "SPEAKER_00": 1.5},
    )
    stereo = mono_to_stereo_from_diarization(mono, sr, result)
    assert stereo.shape == (2, 3000)


def test_write_three_inspect_samples(tmp_path: Path):
    clips = write_inspect_samples(tmp_path / "samples", text_delay_sec=0.16)
    assert len(clips) == 3
    assert len(make_sample_dialogues()) == 3


def test_whisperx_align_model_map_and_split():
    assert asr_mod.split_words("  namaste  bhai ") == ["namaste", "bhai"]
    assert "hi" in asr_mod.WHISPERX_ALIGN_MODELS


def test_mask_in_diarize():
    from duplex_data.diarize import _mask_token

    assert "hf_***" in _mask_token("token=hf_ABC123xyz")
