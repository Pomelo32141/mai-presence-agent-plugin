from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, List

from src.common.database.database_model import ActionRecords
from src.plugin_system.apis import database_api, get_logger


logger = get_logger("presence_agent.memory_store")


class MemoryStore:
    """封装 DB 读写：按 user_id 存状态/记忆/事件"""

    MEMORY_ACTION = "presence_agent_memory"
    STATE_ACTION = "presence_agent_state"
    EVENT_ACTION = "presence_agent_event"

    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name

    async def _save_blob(self, action_id: str, chat_id: str, action_name: str, payload: Dict[str, Any]) -> None:
        data = {
            "action_id": action_id,
            "time": time.time(),
            "action_reasoning": "",
            "action_name": action_name,
            "action_data": json.dumps(payload, ensure_ascii=False),
            "action_done": True,
            "action_build_into_prompt": False,
            "action_prompt_display": self.plugin_name,
            "chat_id": chat_id,
            "chat_info_stream_id": chat_id,
            "chat_info_platform": "",
        }
        await database_api.db_save(ActionRecords, data=data, key_field="action_id", key_value=action_id)

    async def _load_blob(self, action_id: str) -> Dict[str, Any]:
        try:
            record = await database_api.db_get(
                ActionRecords,
                filters={"action_id": action_id},
                limit=1,
                single_result=True,
            )
            if not record:
                return {}
            raw = record.get("action_data") or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"load blob failed for {action_id}: {exc}")
            return {}

    async def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        return await self._load_blob(f"{user_id}_memory")

    async def save_user_memory(self, user_id: str, data: Dict[str, Any]) -> None:
        await self._save_blob(f"{user_id}_memory", chat_id=user_id, action_name=self.MEMORY_ACTION, payload=data)

    async def get_user_state(self, user_id: str) -> Dict[str, Any]:
        return await self._load_blob(f"{user_id}_state")

    async def save_user_state(self, user_id: str, data: Dict[str, Any]) -> None:
        await self._save_blob(f"{user_id}_state", chat_id=user_id, action_name=self.STATE_ACTION, payload=data)

    async def append_event(self, user_id: str, event: Dict[str, Any]) -> None:
        payload = event.copy()
        payload["ts"] = time.time()
        action_id = f"{user_id}_event_{int(payload['ts']*1000)}"
        await self._save_blob(action_id, chat_id=user_id, action_name=self.EVENT_ACTION, payload=payload)

    async def add_memory_point(self, user_id: str, summary: str) -> None:
        memory = await self.get_user_memory(user_id)
        points: List[str] = memory.get("memory_points", [])
        points.append(summary)
        memory["memory_points"] = points[-50:]  # keep recent
        await self.save_user_memory(user_id, memory)
