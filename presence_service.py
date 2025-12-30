from __future__ import annotations

import time
from typing import Dict, Any, Optional

from src.manager.async_task_manager import AsyncTask
from src.plugin_system.apis import component_registry, get_logger
from .state_collector import StateCollector
from .memory_store import MemoryStore
from .brain_llm import PresenceBrain
from .action_executor import ActionExecutor
from .utils import now_ts


logger = get_logger("presence_agent.service")


class PresenceAgentTask(AsyncTask):
    """后台循环：状态 -> LLM 决策 -> 执行动作 -> 摘要写记忆"""

    def __init__(self, plugin_name: str):
        super().__init__(task_name="presence_agent_loop", wait_before_start=5, run_interval=60)
        self.plugin_name = plugin_name
        self.collector = StateCollector()
        self.memory_store = MemoryStore(plugin_name)
        self.action_executor = ActionExecutor(plugin_name)
        self.brain: Optional[PresenceBrain] = None

    def _load_config(self) -> Dict[str, Any]:
        return component_registry.get_plugin_config(self.plugin_name) or {}

    def _ensure_brain(self, config: Dict[str, Any]) -> None:
        general = config.get("general", {})
        model_task = str(general.get("model_task", "replyer"))
        temperature = float(general.get("temperature", 0.7))
        max_tokens = int(general.get("max_tokens", 400))
        if not self.brain or self.brain.model_task != model_task:
            self.brain = PresenceBrain(model_task=model_task, temperature=temperature, max_tokens=max_tokens)

    async def _should_send(self, user_state: Dict[str, Any], user_cfg: Dict[str, Any], config: Dict[str, Any]) -> bool:
        general = config.get("general", {})
        inactivity_threshold = int(general.get("inactivity_threshold_minutes", 120))
        cooldown_minutes = int(general.get("proactive_cooldown_minutes", 60))
        last_proactive_ts = user_cfg.get("last_proactive_ts")
        if last_proactive_ts and now_ts() - last_proactive_ts < cooldown_minutes * 60:
            return False
        return user_state["silence_minutes"] >= inactivity_threshold

    def _conversation_ended(self, user_state: Dict[str, Any], user_cfg: Dict[str, Any], config: Dict[str, Any]) -> bool:
        gap = int(config.get("general", {}).get("end_conversation_gap_minutes", 30))
        last_summary_ts = user_cfg.get("last_summary_ts")
        if not user_state["last_user_ts"]:
            return False
        if last_summary_ts and user_state["last_user_ts"] <= last_summary_ts:
            return False
        silence = user_state["silence_minutes"]
        return silence >= gap

    async def run(self):
        config = self._load_config()
        if not config or not config.get("plugin", {}).get("enabled", True):
            return

        general = config.get("general", {})
        self.run_interval = max(30, int(general.get("scan_interval_seconds", 300)))
        platform = str(general.get("platform", "qq"))
        recent_limit = int(general.get("recent_message_limit", 10))
        self._ensure_brain(config)

        streams = self.collector.list_private_streams(platform)
        if not streams:
            return

        for stream in streams:
            if not stream or not getattr(stream, "stream_id", None):
                continue
            user_id = str(stream.user_info.user_id)
            user_state_obj = self.collector.collect(stream, recent_limit=recent_limit, last_proactive_ts=None)
            user_state = user_state_obj.__dict__

            user_cfg = await self.memory_store.get_user_state(user_id)
            user_mem = await self.memory_store.get_user_memory(user_id)
            last_proactive_ts = user_cfg.get("last_proactive_ts")
            user_state["unread_since_last_proactive"] = (
                bool(user_state.get("last_user_ts")) and last_proactive_ts and user_state["last_user_ts"] < last_proactive_ts
            )

            # Proactive message
            if await self._should_send(user_state, user_cfg, config):
                decision = await self.brain.decide_action(user_state, user_mem, config)  # type: ignore
                if decision.get("should_reply"):
                    sent = await self.action_executor.send_message(stream.stream_id, decision["message"])
                    if sent:
                        await self.action_executor.log_action(
                            stream,
                            decision["message"],
                            decision.get("reason", "llm"),
                            extra={"user_id": user_id},
                        )
                        user_cfg["last_proactive_ts"] = time.time()
                        await self.memory_store.save_user_state(user_id, user_cfg)

            # Conversation end summary & affinity adjust
            if self._conversation_ended(user_state, user_cfg, config):
                summary = await self.brain.summarize_for_memory(user_state)  # type: ignore
                if summary:
                    await self.memory_store.add_memory_point(user_id, summary)
                    delta = await self.brain.decide_affinity_delta(
                        user_state, summary, user_mem.get("affinity_bias", 0.0)  # type: ignore
                    )
                    affinity_bias = float(user_mem.get("affinity_bias", 0.0)) + delta
                    user_mem["affinity_bias"] = round(affinity_bias, 2)
                    await self.memory_store.save_user_memory(user_id, user_mem)
                    user_cfg["last_summary_ts"] = user_state["last_user_ts"]
                    await self.memory_store.append_event(
                        user_id, {"type": "summary", "summary": summary, "affinity_delta": delta}
                    )
                    await self.memory_store.save_user_state(user_id, user_cfg)
