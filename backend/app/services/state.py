"""In-memory stores for chat history and parsed reports."""

from collections import deque
from typing import Deque, Dict

CHAT_HISTORY: Deque[dict] = deque(maxlen=200)
REPORTS: Dict[str, dict] = {}
