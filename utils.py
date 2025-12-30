from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional

from src.config.config import global_config
from src.plugin_system.apis import get_logger


logger = get_logger("presence_agent.utils")


def is_bot_message(message_obj) -> bool:
    """Check if a database message is sent by the bot itself."""
    try:
        return (
            message_obj.user_info.user_id == str(global_config.bot.qq_account)
            and message_obj.user_info.platform == global_config.bot.platform
        )
    except Exception:
        return False


def safe_extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def pick_fallback(messages: List[str], default: str) -> str:
    candidates = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
    if not candidates:
        return default
    return random.choice(candidates)


def now_ts() -> float:
    return time.time()


def hour_in_range(hour: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= hour <= end
    return hour >= start or hour <= end
