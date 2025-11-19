"""OCR helpers (EasyOCR-ready with graceful fallback)."""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None


@lru_cache(maxsize=1)
def _reader() -> Optional["easyocr.Reader"]:
    if not easyocr:
        return None
    return easyocr.Reader(["en"], gpu=False)


def _is_image(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def _is_audio(filename: str) -> bool:
    return filename.lower().endswith((".wav", ".mp3", ".m4a"))


def _is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


async def extract_text(filename: str, data: bytes) -> str:
    reader = _reader()
    if reader and _is_image(filename):
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            array = np.array(image)
        except Exception:  # pragma: no cover - defensive fallback
            return "Unable to decode image. Please retry with a clear scan."
        results = reader.readtext(array, detail=0, paragraph=True)
        return " ".join(results)
    if _is_audio(filename):
        return "Audio transcription pending integration with Whisper/Gemini."  # placeholder
    if _is_pdf(filename):
        return "PDF ingestion placeholder. Use Gemini Vision hook for richer parsing."
    return "Unable to classify file. Provide PDF, image, or audio inputs."
