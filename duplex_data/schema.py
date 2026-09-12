"""Constants for Indic Inner Monologue packing (train-compatible layout)."""

SAMPLE_RATE = 24_000
MIMI_FRAME_RATE = 12.5
AGENT_SPEAKER = "SPEAKER_MAIN"
INDICVOICES_FOCUS = ("hindi", "telugu", "kannada", "tamil")
INDICVOICES_CONFIG_TO_LANG = {
    "hindi": "hi",
    "telugu": "te",
    "kannada": "kn",
    "tamil": "ta",
}
CONVERSATION_MARKERS = (
    "conversation",
    "conversational",
    "roleplay",
    "role-play",
    "role_play",
    "spontaneous",
)
