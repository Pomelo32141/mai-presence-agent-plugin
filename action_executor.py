from __future__ import annotations

from typing import Dict

from src.plugin_system.apis import send_api, database_api, get_logger


logger = get_logger("presence_agent.action_executor")


class ActionExecutor:
    """执行动作：负责发消息与写行动日志"""

    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name

    async def send_message(self, stream_id: str, text: str) -> bool:
        try:
            return await send_api.text_to_stream(text=text, stream_id=stream_id, typing=True, storage_message=True)
        except Exception as exc:
            logger.error(f"send message failed to {stream_id}: {exc}")
            return False

    async def log_action(self, stream, text: str, reason: str, extra: Dict[str, any]) -> None:
        try:
            await database_api.store_action_info(
                chat_stream=stream,
                action_build_into_prompt=False,
                action_prompt_display=text,
                action_done=True,
                action_data={"reason": reason, **extra},
                action_name=f"{self.plugin_name}_{reason}",
            )
        except Exception as exc:
            logger.warning(f"log action failed: {exc}")
