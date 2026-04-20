from __future__ import annotations

import json
import re
from typing import Any


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def keyword_in_text(text: str, keyword: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
    return bool(re.search(pattern, text.lower()))


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
