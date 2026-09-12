"""Preflight checks for duplex_data.prepare."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("duplex_data.deps")

_REQUIRED: list[tuple[str, str]] = [
    ("datasets", "datasets>=4.4.0"),
    ("pyarrow", "pyarrow>=21.0.0"),
    ("soundfile", "soundfile"),
    ("huggingface_hub", "huggingface-hub"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyannote.audio", "pyannote.audio>=4"),
    ("nemo.collections.asr", "AI4Bharat/NeMo nemo-v2 (for indic-conformer)"),
    ("whisperx", "whisperx"),
    ("transformers", "transformers>=4.48"),
]


def missing_data_packages(*, diarize: bool = True, asr_backend: str = "whisper") -> list[str]:
    problems: list[str] = []
    for mod, pip_name in _REQUIRED:
        if mod == "pyannote.audio" and not diarize:
            continue
        if mod.startswith("nemo") and asr_backend != "indic-conformer":
            continue
        if mod == "whisperx" and asr_backend != "indic-conformer":
            continue
        try:
            importlib.import_module(mod)
        except Exception as exc:
            problems.append(f"{pip_name}  [{type(exc).__name__}: {exc}]")
    return problems


def require_data_deps(*, diarize: bool = True, asr_backend: str = "whisper") -> None:
    problems = missing_data_packages(diarize=diarize, asr_backend=asr_backend)
    if not problems:
        logger.info("Data deps OK.")
        return
    raise ImportError(
        "Missing or broken packer dependencies:\n  - "
        + "\n  - ".join(problems)
        + "\n\nFrom data/: pip install -e .\n"
        "Keep CUDA torch: pip install torch torchaudio "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
