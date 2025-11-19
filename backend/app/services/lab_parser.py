"""Extract rudimentary lab values from free text."""

import re
from typing import List

LAB_PATTERN = re.compile(r"(?P<name>[A-Za-z ]{3,15})[:\s]+(?P<value>[0-9]+(?:\.[0-9]+)?%?|[0-9]+\s?mg\/dL)")


def parse_labs_from_text(text: str) -> List[dict]:
    labs = []
    for match in LAB_PATTERN.finditer(text or ""):
        labs.append(
            {
                "name": match.group("name").strip(),
                "value": match.group("value"),
                "range": "Pending",
                "flag": "Review",
            }
        )
    return labs
