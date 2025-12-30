from __future__ import annotations

from typing import Any, Dict, Optional

from src.plugin_system.apis import llm_api, get_logger, message_api
from .utils import safe_extract_json, pick_fallback


logger = get_logger("presence_agent.brain_llm")


class PresenceBrain:
    """所有 LLM 交互集中于此"""

    def __init__(self, model_task: str, temperature: float, max_tokens: int):
        self.model_task = model_task
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pick_model(self):
        available = llm_api.get_available_models()
        model_config = available.get(self.model_task) or available.get("replyer")
        if not model_config and available:
            model_config = list(available.values())[0]
        return model_config

    async def decide_action(self, user_state: Dict[str, Any], user_memory: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """LLM decides whether to speak and what to say."""
        model_config = self._pick_model()
        if not model_config:
            return {"should_reply": False, "reason": "no_model"}

        messages_cfg = config.get("messages", {})
        fallback = messages_cfg.get("inactive_fallback", [])

        history_text = message_api.build_readable_messages_to_str(
            user_state["recent_messages"], replace_bot_name=True, truncate=True
        )
        prompt = (
            "You are an empathetic companion. Decide if you should proactively message the user now.\n"
            "Return JSON with keys: should_reply (bool), message (string), reason (string).\n"
            f"User silence minutes: {user_state['silence_minutes']:.1f}\n"
            f"User name: {user_state['user_name']}\n"
            f"Past memory snippets: {user_memory.get('memory_points', [])[-3:]}\n"
            f"Recent conversation:\n{history_text}\n"
        )

        success, response, _reasoning, _model = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="plugin.presence_agent.decide",
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 512),
        )
        if not success:
            return {
                "should_reply": True,
                "message": pick_fallback(fallback, "嘿，在吗？最近过得怎么样？"),
                "reason": "llm_fail",
            }
        data = safe_extract_json(response)
        message = str(data.get("message", "")).strip()
        if not message:
            message = pick_fallback(fallback, "嘿，我想你啦。")
        return {
            "should_reply": bool(data.get("should_reply", True)),
            "message": message,
            "reason": str(data.get("reason", "llm")),
        }

    async def summarize_for_memory(self, user_state: Dict[str, Any]) -> Optional[str]:
        model_config = self._pick_model()
        if not model_config:
            return None
        history_text = message_api.build_readable_messages_to_str(
            user_state["recent_messages"], replace_bot_name=True, truncate=True
        )
        prompt = (
            "Summarize the recent conversation into one concise sentence that helps future remembering.\n"
            "Return JSON with key 'summary'. Keep it short and friendly.\n"
            f"Conversation:\n{history_text}\n"
        )
        success, response, _reasoning, _model = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="plugin.presence_agent.summary",
            temperature=0.4,
            max_tokens=200,
        )
        if not success:
            return None
        data = safe_extract_json(response)
        summary = str(data.get("summary", "")).strip()
        return summary or None

    async def decide_affinity_delta(self, user_state: Dict[str, Any], summary: str, current_bias: float) -> float:
        """Let LLM suggest an affinity delta based on latest interaction."""
        model_config = self._pick_model()
        if not model_config:
            return 0.0
        history_text = summary or ""
        prompt = (
            "You adjust how warm the bot should be to the user.\n"
            "Given the short summary of last dialogue, output JSON with key 'delta' in range [-2, 2].\n"
            f"Current bias: {current_bias}\nSummary: {history_text}\n"
        )
        success, response, _reasoning, _model = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="plugin.presence_agent.affinity",
            temperature=0.3,
            max_tokens=60,
        )
        if not success:
            return 0.0
        data = safe_extract_json(response)
        try:
            return float(data.get("delta", 0))
        except Exception:
            return 0.0
