"""Lightweight keyword-based emotion detector placeholder."""

EMOTION_MAP = {
    "anxious": ["worried", "anxious", "nervous"],
    "distressed": ["upset", "depressed", "sad", "hopeless"],
    "neutral": [],
}


def detect_emotion(text: str) -> str:
    lower = text.lower()
    for label, terms in EMOTION_MAP.items():
        if any(term in lower for term in terms):
            return label
    return "neutral"
