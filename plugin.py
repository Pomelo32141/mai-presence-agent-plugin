from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from src.plugin_system import (
    BasePlugin,
    ComponentInfo,
    ConfigField,
    EventType,
    BaseEventHandler,
    register_plugin,
)
from src.plugin_system.base.config_types import ConfigLayout, ConfigTab, section_meta
from src.manager.async_task_manager import AsyncTaskManager
from src.plugin_system.apis import get_logger

from .presence_service import PresenceAgentTask

logger = get_logger("presence_agent_plugin")

PLUGIN_NAME = "presence_agent_plugin"


class PresenceAgentStartHandler(BaseEventHandler):
    event_type = EventType.ON_START
    handler_name = "presence_agent_start"
    handler_description = "Start proactive presence loop"
    weight = 50

    async def execute(self, message):
        try:
            if not self.get_config("plugin.enabled", True):
                return True, False, None, None, None
            await _presence_task_manager.add_task(_presence_task)
            return True, False, "PresenceAgent task started", None, None
        except Exception as exc:
            logger.error(f"Failed to start PresenceAgent task: {exc}")
            return False, False, str(exc), None, None


class PresenceAgentStopHandler(BaseEventHandler):
    event_type = EventType.ON_STOP
    handler_name = "presence_agent_stop"
    handler_description = "Stop proactive presence loop"
    weight = 50

    async def execute(self, message):
        try:
            await _presence_task_manager.stop_and_wait_all_tasks()
            return True, False, "PresenceAgent task stopped", None, None
        except Exception as exc:
            logger.error(f"Failed to stop PresenceAgent task: {exc}")
            return False, False, str(exc), None, None


@register_plugin
class PresenceAgentPlugin(BasePlugin):
    """PresenceAgent plugin - 主动陪伴与记忆总结（入口仅做组装）"""

    plugin_name = PLUGIN_NAME
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name = "config.toml"
    config_section_descriptions = {
        "plugin": section_meta("插件基础设置", icon="settings", order=1),
        "general": section_meta("行为与频率设置", icon="timer", order=2),
        "messages": section_meta("提示词与兜底消息", icon="message-circle", order=3),
    }

    config_schema = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用 PresenceAgent 插件", input_type="switch"),
            "config_version": ConfigField(type=str, default="2.0.0", description="配置结构版本号", disabled=True),
        },
        "general": {
            "scan_interval_seconds": ConfigField(
                type=int, default=300, description="扫描间隔（秒）", min=30, max=3600, step=10
            ),
            "inactivity_threshold_minutes": ConfigField(
                type=int, default=180, description="沉默多少分钟后主动关心", min=1, max=10080
            ),
            "proactive_cooldown_minutes": ConfigField(
                type=int, default=90, description="主动消息冷却时间（分钟）", min=5, max=1440
            ),
            "recent_message_limit": ConfigField(
                type=int, default=12, description="用于决策的最近消息条数", min=1, max=60
            ),
            "end_conversation_gap_minutes": ConfigField(
                type=int, default=30, description="超过多少分钟无新消息视为对话结束", min=5, max=240
            ),
            "model_task": ConfigField(type=str, default="replyer", description="使用的 LLM 任务名"),
            "temperature": ConfigField(type=float, default=0.7, description="LLM 温度", min=0.0, max=2.0, step=0.1),
            "max_tokens": ConfigField(type=int, default=400, description="LLM 最大输出 token 数", min=128, max=8192),
            "platform": ConfigField(type=str, default="qq", description='平台过滤："qq" 或 "all"', choices=["qq", "all"]),
        },
        "messages": {
            "inactive_fallback": ConfigField(
                type=list,
                default=[
                    "在吗？我有点想你啦。",
                    "好久没聊天了，最近还好吗？",
                    "在忙吗？我来看看你～",
                ],
                description="久未互动时的兜底消息",
                item_type="string",
                max_items=20,
            ),
        },
    }

    config_layout = ConfigLayout(
        type="tabs",
        tabs=[
            ConfigTab(id="basic", title="基础", sections=["plugin", "general"], icon="settings", order=1),
            ConfigTab(id="messages", title="文案", sections=["messages"], icon="message-circle", order=2),
        ],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _presence_task
        _presence_task = PresenceAgentTask(self.plugin_name)

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (PresenceAgentStartHandler.get_handler_info(), PresenceAgentStartHandler),
            (PresenceAgentStopHandler.get_handler_info(), PresenceAgentStopHandler),
        ]


_presence_task_manager = AsyncTaskManager()
_presence_task = PresenceAgentTask(PLUGIN_NAME)
