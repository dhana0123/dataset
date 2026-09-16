"""Local FastAPI + WaveSurfer 2D-lane speech-plan annotator.

Usage:
  python -m duplex_data.speech_plan.annotator --root ./speech_plans
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Any

from duplex_data.speech_plan.schema import (
    BOUNDARY_VALUES,
    EMPHASIS_VALUES,
    ENERGY_VALUES,
    GLOBAL_STATE_VALUES,
    LANE_KEYS,
    PITCH_VALUES,
    RATE_VALUES,
    Event,
    GlobalState,
    SpeechPlan,
)

logger = logging.getLogger("duplex_data.speech_plan.annotator")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _find_plans(root: Path) -> list[dict[str, str]]:
    """List clip folders that contain speech_plan.json."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    if not root.exists():
        return items
    for p in sorted(root.rglob("speech_plan.json")):
        rel = p.parent.relative_to(root)
        if str(rel) == ".":
            clip_id = p.parent.name
        else:
            clip_id = str(rel).replace("\\", "/")
        if clip_id in seen:
            continue
        seen.add(clip_id)
        items.append({"id": clip_id, "path": str(p)})
    return items


def _resolve_audio(plan: SpeechPlan, plan_path: Path) -> Path | None:
    ap = Path(plan.audio_path)
    if ap.is_file():
        return ap
    cand = plan_path.parent / plan.audio_path
    if cand.is_file():
        return cand
    # Any wav in the same folder
    for ext in (".wav", ".flac", ".ogg", ".mp3"):
        matches = list(plan_path.parent.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def create_app(root: Path):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError(
            "Annotator needs fastapi + uvicorn. Install: pip install -e \".[speech-plan]\""
        ) from exc

    root = root.resolve()
    app = FastAPI(title="Speech Plan Annotator", version="0.1.0")
    app.state.root = root

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(500, f"Missing UI at {index_path}")
        return FileResponse(index_path)

    @app.get("/api/meta")
    def meta():
        return {
            "root": str(root),
            "lane_keys": list(LANE_KEYS),
            "vocab": {
                "global_state": list(GLOBAL_STATE_VALUES),
                "pitch": list(PITCH_VALUES),
                "energy": list(ENERGY_VALUES),
                "rate": list(RATE_VALUES),
                "emphasis": list(EMPHASIS_VALUES),
                "boundary": list(BOUNDARY_VALUES),
                "pause": ["ms"],
                "nonverbal": [],
            },
        }

    @app.get("/api/clips")
    def list_clips():
        return {"clips": _find_plans(root)}

    def _plan_path(clip_id: str) -> Path:
        # clip_id is relative folder under root
        safe = clip_id.replace("\\", "/").lstrip("/")
        if ".." in safe.split("/"):
            raise HTTPException(400, "Invalid clip id")
        p = root / safe / "speech_plan.json"
        if p.is_file():
            return p
        # Fallback: id is just stem matching any plan
        for item in _find_plans(root):
            if item["id"] == clip_id or item["id"].endswith("/" + clip_id):
                return Path(item["path"])
        raise HTTPException(404, f"Clip not found: {clip_id}")

    # Register /audio before the generic {clip_id:path} route so path
    # converters do not swallow ".../audio" as part of clip_id.
    @app.get("/api/clips/{clip_id:path}/audio")
    def get_audio(clip_id: str):
        path = _plan_path(clip_id)
        plan = SpeechPlan.load(path)
        audio = _resolve_audio(plan, path)
        if audio is None or not audio.is_file():
            raise HTTPException(404, "Audio not found for clip")
        media = mimetypes.guess_type(str(audio))[0] or "audio/wav"
        return FileResponse(audio, media_type=media, filename=audio.name)

    @app.put("/api/clips/{clip_id:path}")
    async def save_clip(clip_id: str, body: dict[str, Any]):
        path = _plan_path(clip_id)
        existing = SpeechPlan.load(path)
        incoming = body.get("plan") or body

        # Merge: keep tracks/asr from disk unless explicitly sent
        lanes_raw = incoming.get("lanes") or {}
        lanes: dict[str, list[Event]] = {}
        for key in LANE_KEYS:
            events = []
            for e in lanes_raw.get(key, []):
                events.append(Event.from_dict(e))
            lanes[key] = events
        lanes["nonverbal"] = []

        gs = GlobalState.from_dict(incoming.get("global_state"))
        gs.source = "human"

        plan = SpeechPlan(
            audio_path=incoming.get("audio_path", existing.audio_path),
            duration=float(incoming.get("duration", existing.duration)),
            language=str(incoming.get("language", existing.language)),
            global_state=gs,
            nonverbal=[],
            lanes=lanes,
            tracks=incoming.get("tracks", existing.tracks),
            asr=incoming.get("asr", existing.asr),
        )
        plan.save(path)
        return {"ok": True, "plan_path": str(path), "plan": plan.to_dict()}

    @app.get("/api/clips/{clip_id:path}")
    def get_clip(clip_id: str):
        path = _plan_path(clip_id)
        plan = SpeechPlan.load(path)
        audio = _resolve_audio(plan, path)
        return {
            "id": clip_id,
            "plan_path": str(path),
            "audio_url": f"/api/clips/{clip_id}/audio" if audio else None,
            "plan": plan.to_dict(),
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2D speech-plan annotator UI")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory of speech_plan folders (from extract)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.root.exists():
        logger.error("Root not found: %s", args.root)
        return 1

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Annotator needs uvicorn. Install: pip install -e \".[speech-plan]\""
        ) from exc

    app = create_app(args.root)
    logger.info("Annotator at http://%s:%d  root=%s", args.host, args.port, args.root.resolve())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
