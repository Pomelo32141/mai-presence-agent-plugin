from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.plugin_system.apis import chat_api, message_api, get_logger
from .utils import is_bot_message


logger = get_logger("presence_agent.state_collector")


@dataclass
class UserState:
    stream_id: str
    platform: str
    user_id: str
    user_name: str
    recent_messages: List[Any]
    last_user_ts: Optional[float]
    last_bot_ts: Optional[float]
    silence_minutes: float
    unread_since_last_proactive: bool


class StateCollector:
    """只收集事实数据，不做决策或写入"""

    def __init__(self, recent_hours: int = 72):
        self.recent_hours = recent_hours

    def _get_recent_messages(self, stream_id: str, limit: int) -> List[Any]:
        try:
            return message_api.get_recent_messages(
                chat_id=stream_id, hours=self.recent_hours, limit=limit, limit_mode="latest"
            )
        except Exception as exc:
            logger.warning(f"failed to load recent messages for {stream_id}: {exc}")
            return []

    def _last_ts(self, messages: List[Any], *, bot: bool) -> Optional[float]:
        ordered = sorted(messages, key=lambda m: m.time or 0)
        for msg in reversed(ordered):
            if is_bot_message(msg) is bot:
                return msg.time
        return None

    def collect(self, stream, recent_limit: int, last_proactive_ts: Optional[float]) -> UserState:
        recent_messages = self._get_recent_messages(stream.stream_id, recent_limit)
        last_user_ts = self._last_ts(recent_messages, bot=False)
        last_bot_ts = self._last_ts(recent_messages, bot=True)
        now = time.time()
        silence_minutes = (now - last_user_ts) / 60.0 if last_user_ts else 1e9
        unread_since_last_proactive = False
        if last_proactive_ts and last_user_ts and last_user_ts < last_proactive_ts:
            unread_since_last_proactive = True

        user_name = getattr(stream.user_info, "user_nickname", None) or "friend"

        return UserState(
            stream_id=stream.stream_id,
            platform=stream.platform,
            user_id=str(stream.user_info.user_id),
            user_name=user_name,
            recent_messages=recent_messages,
            last_user_ts=last_user_ts,
            last_bot_ts=last_bot_ts,
            silence_minutes=silence_minutes,
            unread_since_last_proactive=unread_since_last_proactive,
        )

    def list_private_streams(self, platform: str):
        platform_arg = chat_api.SpecialTypes.ALL_PLATFORMS if platform == "all" else platform
        return chat_api.get_private_streams(platform_arg)
