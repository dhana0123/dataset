"""Synthetic Indic duplex: open-source Sarvam-M scripts → Parler → IM pack.

Example:
  python -m duplex_data.synthetic \\
    --out ./moshi_im_synth_te \\
    --lang te \\
    --num-dialogs 3 \\
    --domains recruitment,customer_support \\
    --dry-run-scripts --skip-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from . import asr as asr_mod
from .parler_tts import resolve_agent_desc, resolve_user_desc, synthesize
from .prepare import save_clip
from .push import push_private_dataset, resolve_dataset_repo_id, write_dataset_readme
from .sarvam_scripts import (
    DEFAULT_LLM,
    DialogueScript,
    Turn,
    generate_dialogue,
    script_from_dict,
)
from .schema import SAMPLE_RATE
from .stitch import stitch_turns
from .topics import list_domains, parse_domains, sample_scenario

logger = logging.getLogger("duplex_data.synthetic")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _even_split_words(text: str, duration: float) -> list[tuple[str, tuple[float, float]]]:
    words = [w for w in text.split() if w.strip()]
    if not words or duration <= 0:
        return []
    step = duration / len(words)
    out = []
    for i, w in enumerate(words):
        out.append((w, (i * step, min(duration, (i + 1) * step))))
    return out


def _align_agent(
    agent_mono: np.ndarray,
    sr: int,
    *,
    language: str,
    known_text: str,
    whisper_model: str = "small",
) -> tuple[list[tuple[str, tuple[float, float]]], dict]:
    try:
        result = asr_mod.transcribe_agent_words(
            agent_mono,
            sr,
            language=language,
            backend="whisper",
            asr_model=whisper_model,
        )
        return result.words, result.provenance_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper align failed (%s); even-split fallback", exc)
        dur = float(len(agent_mono)) / float(sr)
        words = _even_split_words(known_text, dur)
        prov = {
            "transcript_backend": "even_split_fallback",
            "timer_backend": "even_split_fallback",
            "language": language,
            "transcript_text": known_text,
        }
        return words, prov


def _agent_transcript(script: DialogueScript) -> str:
    return " ".join(t.text for t in script.turns if t.role == "agent")


def pack_one_dialog(
    script: DialogueScript,
    *,
    out_dir: Path,
    jsonl_path: Path,
    stem: str,
    agent_style: str,
    user_voice: str,
    device: str,
    text_delay_sec: float,
    rng: random.Random,
    whisper_model: str,
) -> dict:
    agent_desc = resolve_agent_desc(agent_style)
    user_desc = resolve_user_desc(user_voice)

    rendered: list[tuple[str, np.ndarray, int]] = []
    for turn in script.turns:
        desc = agent_desc if turn.role == "agent" else user_desc
        logger.info("[%s] TTS %s: %s", stem, turn.role, turn.text[:60])
        wav, sr = synthesize(turn.text, desc, device=device)
        rendered.append((turn.role, wav, sr))

    stitched = stitch_turns(rendered, target_sr=SAMPLE_RATE, rng=rng)
    stereo = stitched.stereo
    words, asr_prov = _align_agent(
        stereo[0],
        stitched.sample_rate,
        language=script.lang,
        known_text=_agent_transcript(script),
        whisper_model=whisper_model,
    )

    provenance = {
        "source": "synthetic_sarvam_30b_parler",
        "domain": script.domain,
        "scenario": script.scenario,
        "llm": script.llm,
        "agent_style": agent_style,
        "user_voice": user_voice,
        "asr": asr_prov,
    }
    clip = save_clip(
        out_dir,
        stem,
        stereo,
        words,
        jsonl_path,
        text_delay_sec=text_delay_sec,
        sample_rate=stitched.sample_rate,
        provenance=provenance,
    )
    return {
        "stem": stem,
        "wav": str(clip.wav_path),
        "duration": clip.duration,
        "n_words": clip.n_agent_words,
        "domain": script.domain,
    }


def load_or_generate_scripts(args: argparse.Namespace) -> list[DialogueScript]:
    scripts_dir = args.out / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    domains = parse_domains(args.domains)

    if args.scripts_dir:
        paths = sorted(Path(args.scripts_dir).glob("*.json"))
        if not paths:
            raise SystemExit(f"No JSON scripts in {args.scripts_dir}")
        return [
            script_from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in paths[: args.num_dialogs]
        ]

    scripts: list[DialogueScript] = []
    for i in range(args.num_dialogs):
        draw = sample_scenario(domains, rng=rng)
        logger.info("Dialog %d/%d domain=%s", i + 1, args.num_dialogs, draw.domain)
        if args.skip_llm:
            script = DialogueScript(
                domain=draw.domain,
                scenario=draw.scenario,
                lang=args.lang,
                agent_role=draw.agent_role,
                user_role=draw.user_role,
                llm="placeholder",
                turns=[
                    Turn(
                        role="agent",
                        text="నమస్కారం అండి, ఎలా సహాయం చేయాలి చెప్పండి?",
                    ),
                    Turn(
                        role="user",
                        text="అవునండి, నాకు ఒక ప్రాబ్లం ఉంది కదా, ఓటిపి రావట్లేదు.",
                    ),
                    Turn(
                        role="agent",
                        text="సరేలే, నంబరు చెప్తే చూస్తాను అండి.",
                    ),
                    Turn(
                        role="user",
                        text="ఓకే బ్రో, థాంక్స్. మళ్లీ ట్రై చేస్తా.",
                    ),
                ],
            )
        else:
            script = generate_dialogue(
                draw,
                lang=args.lang,
                max_turns=args.max_turns,
                model_id=args.llm_model,
                device=args.device,
                load_in_4bit=args.llm_load_in_4bit,
                strict_llm=args.strict_llm,
            )
            if script.turns and script.turns[0].role != "agent":
                from .sarvam_scripts import _agent_greeting

                script.turns.insert(
                    0, Turn(role="agent", text=_agent_greeting(args.lang))
                )
        path = scripts_dir / f"{i:04d}_{script.domain}.json"
        path.write_text(
            json.dumps(script.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        scripts.append(script)
        logger.info("Wrote %s (%d turns)", path, len(script.turns))
    return scripts


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    p = argparse.ArgumentParser(
        description="Sarvam-M (open) + Parler → Moshi IM synthetic duplex pack"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lang", type=str, default="te", help="Dialogue language code (v1: te).")
    p.add_argument("--num-dialogs", type=int, default=3)
    p.add_argument(
        "--domains",
        type=str,
        default=None,
        help=f"Comma domains. Default all. Options: {','.join(list_domains())}",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM,
        help=(
            "HF id or alias: sarvamai/sarvam-30b / indic-22 (default, 22 Indic langs), "
            "sarvam-m, param2 → Param2-17B, "
            "indic-7b / airavata → ai4bharat/Airavata (Hindi 7B only)"
        ),
    )
    p.add_argument("--llm-load-in-4bit", action="store_true")
    p.add_argument(
        "--strict-llm",
        action="store_true",
        help=(
            "Fail if sarvam-30b cannot load (no auto-fallback to sarvam-m when "
            "transformers lacks ALL_ATTENTION_FUNCTIONS)."
        ),
    )
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--agent-style", type=str, default="leela")
    p.add_argument("--user-voice", type=str, default="male")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-delay-sec", type=float, default=0.0)
    p.add_argument("--whisper-model", type=str, default="small")
    p.add_argument(
        "--dry-run-scripts",
        action="store_true",
        help="Only write scripts JSON (runs LLM unless --skip-llm).",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Write placeholder Telugu turns (no GPU LLM).",
    )
    p.add_argument(
        "--scripts-dir",
        type=Path,
        default=None,
        help="Reuse existing script JSON folder (skip LLM).",
    )
    p.add_argument("--hf-dataset", type=str, default="BelluAi/dupxel-indic")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--list-domains", action="store_true")
    args = p.parse_args(argv)

    if args.list_domains:
        print("\n".join(list_domains()))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(exist_ok=True)
    jsonl_path = args.out / "train.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    scripts = load_or_generate_scripts(args)
    if args.dry_run_scripts:
        logger.info("Dry-run scripts only → %s/scripts", args.out)
        return

    rng = random.Random(args.seed + 17)
    meta_rows = []
    for i, script in enumerate(scripts):
        stem = f"synth_{i:04d}_{script.domain}"
        try:
            row = pack_one_dialog(
                script,
                out_dir=args.out,
                jsonl_path=jsonl_path,
                stem=stem,
                agent_style=args.agent_style,
                user_voice=args.user_voice,
                device=args.device,
                text_delay_sec=args.text_delay_sec,
                rng=rng,
                whisper_model=args.whisper_model,
            )
            meta_rows.append(row)
        except Exception:
            logger.exception("Failed %s", stem)

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "synthetic_sarvam_30b_parler",
        "llm": args.llm_model,
        "lang": args.lang,
        "n_clips": len(meta_rows),
        "clips": meta_rows,
    }
    (args.out / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_dataset_readme(args.out)
    logger.info("Packed %d clips → %s", len(meta_rows), args.out)

    if args.no_push or not meta_rows:
        return
    repo = resolve_dataset_repo_id(args.hf_dataset)
    push_private_dataset(args.out, hf_dataset=repo)
    logger.info("Uploaded private dataset %s", repo)


if __name__ == "__main__":
    main()
