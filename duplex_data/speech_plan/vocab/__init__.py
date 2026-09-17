"""Load nonverbal vocal-event vocabulary from packaged JSON.

Indic forms use native-script ``annotation_value`` as Event.value
(e.g. हम्म, హ్మ్) so labels match direct transcripts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_VOCAB_PATH = Path(__file__).resolve().parent / "nonverbal_events.json"


@lru_cache(maxsize=1)
def load_nonverbal_vocab() -> dict[str, Any]:
    if not _VOCAB_PATH.is_file():
        raise FileNotFoundError(f"Missing nonverbal vocab: {_VOCAB_PATH}")
    return json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))


def list_nonverbal_events() -> list[dict[str, Any]]:
    return list(load_nonverbal_vocab().get("events") or [])


def list_nonverbal_event_ids() -> tuple[str, ...]:
    """English taxonomy ids + Indic native annotation values."""
    ids = [str(e["id"]) for e in list_nonverbal_events()]
    for form in indic_forms():
        val = form.get("annotation_value") or form.get("native")
        if val and str(val) not in ids:
            ids.append(str(val))
    return tuple(ids)


def nonverbal_events_by_category() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for e in list_nonverbal_events():
        cat = str(e.get("category") or "other")
        out.setdefault(cat, []).append(e)
    return out


def indic_forms() -> list[dict[str, Any]]:
    return list(load_nonverbal_vocab().get("indic_forms") or [])


def nonverbal_references() -> list[dict[str, Any]]:
    return list(load_nonverbal_vocab().get("references") or [])


def nonverbal_dropdown_options() -> list[dict[str, str]]:
    """Flat options for the annotator select (Indic native first, then English)."""
    opts: list[dict[str, str]] = []
    # Indic native-script transcripts first — these are the preferred Event.value.
    for form in indic_forms():
        native = str(form.get("annotation_value") or form.get("native") or "")
        if not native:
            continue
        lang = str(form.get("language_name") or form.get("language") or "Indic")
        translit = str(form.get("transliteration") or "")
        opts.append(
            {
                "id": native,
                "label": f"{native}  ·  {translit}  ({lang})",
                "group": "indic",
            }
        )
    for e in list_nonverbal_events():
        opts.append(
            {
                "id": str(e["id"]),
                "label": str(e.get("label") or e["id"]),
                "group": str(e.get("category") or "other"),
            }
        )
    return opts


def nonverbal_meta_payload() -> dict[str, Any]:
    """Shape returned under /api/meta vocab.nonverbal."""
    by_cat = nonverbal_events_by_category()
    indic_opts = [
        {
            "id": str(f.get("annotation_value") or f.get("native")),
            "label": (
                f"{f.get('annotation_value') or f.get('native')}  ·  "
                f"{f.get('transliteration')}  ({f.get('language_name')})"
            ),
        }
        for f in indic_forms()
        if f.get("annotation_value") or f.get("native")
    ]
    by_cat_ui = {
        "indic": indic_opts,
    }
    for cat, events in by_cat.items():
        by_cat_ui[cat] = [{"id": e["id"], "label": e.get("label", e["id"])} for e in events]
    return {
        "event_ids": list(list_nonverbal_event_ids()),
        "options": nonverbal_dropdown_options(),
        "categories": ["indic"] + list(load_nonverbal_vocab().get("categories") or []),
        "by_category": by_cat_ui,
        "indic_forms": indic_forms(),
        "references": nonverbal_references(),
        "annotation_policy": load_nonverbal_vocab().get("annotation_policy") or {},
    }


__all__ = [
    "load_nonverbal_vocab",
    "list_nonverbal_events",
    "list_nonverbal_event_ids",
    "nonverbal_events_by_category",
    "indic_forms",
    "nonverbal_references",
    "nonverbal_dropdown_options",
    "nonverbal_meta_payload",
]
