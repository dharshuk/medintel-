"""Simple heuristic risk scoring until the clinical engine is wired up."""

from dataclasses import dataclass


@dataclass
class RiskResult:
    level: str
    confidence: str


KEYWORDS = {
    "Red": ["severe", "chest pain", "suicidal", "stroke", "faint"],
    "Amber": ["pain", "dizzy", "shortness", "elevated", "high"],
}


def score_risk(question: str, answer: str) -> RiskResult:
    text = f"{question} {answer}".lower()
    for level, words in KEYWORDS.items():
        if any(word in text for word in words):
            confidence = "0.78" if level == "Red" else "0.64"
            return RiskResult(level=level, confidence=confidence)
    return RiskResult(level="Green", confidence="0.55")
