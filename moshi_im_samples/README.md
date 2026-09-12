# Moshi Inner Monologue — 3 inspect samples

Synthetic stereo tones (not real speech) so you can check **layout** without ASR.

| File | Scenario |
|------|----------|
| `wav/01_overlap.wav` + `.json` | Agent speaks, user overlaps mid-turn (no control tokens). |
| `wav/02_pause.wav` + `.json` | Agent pauses ~1.1s then continues; user channel is silence. |
| `wav/03_backchannel.wav` + `.json` | User talks continuously; agent short haan (backchannel). |

## Layout

- **Left channel** = agent (Moshi)
- **Right channel** = user
- `*.json` = agent word alignments only (`SPEAKER_MAIN`)
- `train.jsonl` = `{"path", "duration"}`

Open the wav in any editor that shows stereo; open the sibling JSON for times.

