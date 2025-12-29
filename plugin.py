import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import global_config
from src.manager.async_task_manager import AsyncTask, AsyncTaskManager
from src.person_info.person_info import Person
from src.plugin_system import (
    BaseEventHandler,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    EventHandlerInfo,
    EventType,
    register_plugin,
)
from src.plugin_system.apis import chat_api, llm_api, message_api, send_api, database_api, get_logger
from src.plugin_system.core.component_registry import component_registry
from src.plugin_system.base.config_types import ConfigLayout, ConfigTab, section_meta

logger = get_logger("presence_agent_plugin")

STATE_FILE_NAME = "presence_agent_state.json"
TASK_NAME = "presence_agent_loop"
PLUGIN_NAME = "presence_agent_plugin"


@dataclass
# 时间段提醒规则数据结构
class SegmentRule:
    """Time segment reminder rule."""

    name: str
    start_hour: int
    end_hour: int
    reminder: str
    enabled: bool = True


# 从磁盘加载状态
def _load_state(file_path: str) -> Dict[str, Any]:
    """Load per-stream state from disk."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to load state file: {exc}")
        return {}


# 保存状态到磁盘
def _save_state(file_path: str, state: Dict[str, Any]) -> None:
    """Save per-stream state to disk."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"Failed to save state file: {exc}")


# 判断是否为机器人自身消息
def _is_bot_message(message_obj) -> bool:
    """Check if a database message is sent by the bot itself."""
    try:
        return (
            message_obj.user_info.user_id == str(global_config.bot.qq_account)
            and message_obj.user_info.platform == global_config.bot.platform
        )
    except Exception:
        return False


# 获取最近一条用户消息时间
def _get_last_user_message_ts(messages: List[Any]) -> Optional[float]:
    """Find the last user message timestamp from recent messages."""
    ordered = sorted(messages, key=lambda m: m.time or 0)
    for msg in reversed(ordered):
        if not _is_bot_message(msg):
            return msg.time
    return None


# 读取某会话的近期消息
def _get_recent_messages(stream_id: str, limit: int) -> List[Any]:
    """Fetch recent messages from the database for a stream."""
    try:
        return message_api.get_recent_messages(chat_id=stream_id, hours=72, limit=limit, limit_mode="latest")
    except Exception as exc:
        logger.warning(f"Failed to get recent messages for {stream_id}: {exc}")
        return []


# 获取今天日期字符串
def _today_str() -> str:
    """Return local date string (YYYY-MM-DD)."""
    return time.strftime("%Y-%m-%d", time.localtime())


# 获取当前小时
def _current_hour() -> int:
    """Return local hour."""
    return time.localtime().tm_hour


# 判断小时是否落在时间段内（支持跨午夜）
def _hour_in_range(hour: int, start: int, end: int) -> bool:
    """Check if hour is within a segment that may cross midnight."""
    if start <= end:
        return start <= hour <= end
    return hour >= start or hour <= end


# 从候选列表挑选兜底消息
def _pick_fallback_message(messages: List[str]) -> str:
    """Pick a fallback message."""
    candidates = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
    if not candidates:
        return "Hey, just checking in. How are you doing?"
    return random.choice(candidates)


# 从文本中安全提取 JSON
def _safe_extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
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


# 获取最近一条用户消息文本
def _get_recent_user_text(messages: List[Any]) -> str:
    """Return last user message text for topic continuation."""
    ordered = sorted(messages, key=lambda m: m.time or 0)
    for msg in reversed(ordered):
        if _is_bot_message(msg):
            continue
        content = (msg.processed_plain_text or "").strip()
        if content:
            return content
    return ""


# 基于关键词估算情绪强度
def _compute_emotion_intensity(messages: List[Any], config: Dict[str, Any]) -> int:
    """Estimate emotion intensity from recent messages using keyword matching."""
    emotion_cfg = config.get("emotion", {})
    if not emotion_cfg.get("enabled", True):
        return 0
    keywords = emotion_cfg.get("keywords", [])
    if not isinstance(keywords, list) or not keywords:
        return 0

    recent_limit = int(emotion_cfg.get("recent_limit", 8))
    hits = 0
    for msg in messages[-recent_limit:]:
        if _is_bot_message(msg):
            continue
        text = (msg.processed_plain_text or "").lower()
        if not text:
            continue
        for kw in keywords:
            if isinstance(kw, str) and kw and kw.lower() in text:
                hits += 1
    cap = int(emotion_cfg.get("hit_cap", 5))
    return max(0, min(10, min(hits, cap) * 2))


# 汇总情绪强度与标签
def _summarize_emotion(messages: List[Any], config: Dict[str, Any]) -> Tuple[int, str]:
    """Summarize emotion intensity and label for prompt guidance."""
    intensity = _compute_emotion_intensity(messages, config)
    if intensity <= 0:
        return 0, "neutral"
    if intensity >= int(config.get("emotion", {}).get("intensity_threshold", 6)):
        return intensity, "sensitive"
    return intensity, "mild"


# 统计近期用户消息数量
def _count_recent_user_messages(messages: List[Any]) -> int:
    """Count recent user messages (exclude bot)."""
    return sum(1 for msg in messages if not _is_bot_message(msg))


# 计算好感度分数
def _compute_affinity_score(
    stream,
    recent_messages: List[Any],
    last_user_ts: Optional[float],
    config: Dict[str, Any],
) -> int:
    """Estimate affinity score based on memory and recent context."""
    affinity_cfg = config.get("affinity", {})
    if not affinity_cfg.get("enabled", True):
        return 0

    score = 0
    try:
        person = Person(platform=stream.platform, user_id=str(stream.user_info.user_id))
        if getattr(person, "is_known", False):
            memory_points = [p for p in (person.memory_points or []) if p is not None]
            memory_weight = int(affinity_cfg.get("memory_point_weight", 2))
            know_weight = int(affinity_cfg.get("know_times_weight", 4))
            memory_cap = int(affinity_cfg.get("memory_point_cap", 30))
            know_cap = int(affinity_cfg.get("know_times_cap", 20))

            score += min(len(memory_points), memory_cap) * memory_weight
            score += min(int(getattr(person, "know_times", 0)), know_cap) * know_weight
    except Exception:
        pass

    recent_count = _count_recent_user_messages(recent_messages)
    recent_weight = int(affinity_cfg.get("recent_message_weight", 2))
    recent_cap = int(affinity_cfg.get("recent_message_cap", 20))
    score += min(recent_count, recent_cap) * recent_weight

    if last_user_ts:
        delta_hours = (time.time() - last_user_ts) / 3600.0
        if delta_hours <= 24:
            score += int(affinity_cfg.get("recency_bonus_1d", 10))
        elif delta_hours <= 72:
            score += int(affinity_cfg.get("recency_bonus_3d", 5))
        elif delta_hours <= 168:
            score += int(affinity_cfg.get("recency_bonus_7d", 2))

    return max(0, min(100, score))


# 根据好感度调整阈值
def _adjust_by_affinity(
    base_value: int,
    affinity_score: int,
    reduce_ratio: float,
    min_value: int,
) -> int:
    """Reduce a base value by affinity score to make the bot more clingy."""
    ratio = max(0.0, min(1.0, affinity_score / 100.0))
    reduced = int(base_value * (1 - ratio * reduce_ratio))
    return max(min_value, reduced)


# 判断是否处于安静时段
def _is_quiet_time(now_ts: float, config: Dict[str, Any]) -> bool:
    """Check if current time is in quiet hours."""
    quiet_cfg = config.get("quiet_hours", {})
    if not quiet_cfg.get("enabled", True):
        return False
    start_hour = int(quiet_cfg.get("start_hour", 23))
    end_hour = int(quiet_cfg.get("end_hour", 7))
    hour = time.localtime(now_ts).tm_hour
    return _hour_in_range(hour, start_hour, end_hour)


# 判断用户是否允许主动触达
def _is_user_allowed(stream, config: Dict[str, Any]) -> bool:
    """Check allowlist/denylist for proactive messages."""
    list_cfg = config.get("lists", {})
    user_id = str(stream.user_info.user_id) if stream and stream.user_info else ""
    allowlist = list_cfg.get("allowlist", [])
    denylist = list_cfg.get("denylist", [])

    if isinstance(denylist, list) and user_id in [str(x) for x in denylist]:
        return False
    if isinstance(allowlist, list) and allowlist:
        return user_id in [str(x) for x in allowlist]
    return True


# 根据好感度选择语气风格
def _decide_tone(affinity_score: int, config: Dict[str, Any]) -> Tuple[str, str]:
    """Pick tone style name and description based on affinity score."""
    style_cfg = config.get("style", {})
    thresholds = style_cfg.get("tone_thresholds", {})
    if isinstance(thresholds, dict) and thresholds:
        formal_threshold = int(thresholds.get("formal_max", style_cfg.get("formal_max", 30)))
        warm_threshold = int(thresholds.get("warm_max", style_cfg.get("warm_max", 70)))
    else:
        formal_threshold = int(style_cfg.get("formal_max", 30))
        warm_threshold = int(style_cfg.get("warm_max", 70))

    if affinity_score <= formal_threshold:
        style_key = "formal"
    elif affinity_score <= warm_threshold:
        style_key = "warm"
    else:
        style_key = "intimate"

    tone_map = style_cfg.get("tone_descriptions", {})
    if not isinstance(tone_map, dict):
        tone_map = {}
    fallback_map = {
        "formal": str(style_cfg.get("formal_description", "语气礼貌克制，表达关心但不过度打扰。")),
        "warm": str(style_cfg.get("warm_description", "语气温柔自然，像朋友般关心。")),
        "intimate": str(style_cfg.get("intimate_description", "语气更亲近、更黏人，表达强烈在意。")),
    }
    description = str(tone_map.get(style_key, fallback_map.get(style_key, "")))
    return style_key, description


# 解析当前人设包配置
def _resolve_persona(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Resolve persona pack name and hints."""
    persona_cfg = config.get("persona", {})
    pack_key = str(persona_cfg.get("active_pack", "warm_companion"))
    packs_raw = persona_cfg.get("packs")

    packs: Dict[str, Dict[str, Any]] = {}
    if isinstance(packs_raw, dict):
        packs = {str(k): v for k, v in packs_raw.items() if isinstance(v, dict)}
    elif isinstance(packs_raw, list):
        for item in packs_raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            packs[key] = item

    pack = packs.get(pack_key, {}) if isinstance(packs, dict) else {}
    name = str(pack.get("name", "温柔陪伴"))
    style_hint = str(pack.get("style_hint", "语气温柔自然，像朋友一样关心对方。"))
    behavior_hint = str(pack.get("behavior_hint", "回复简短、真诚、有生活气息。"))
    return name, style_hint, behavior_hint


# 根据好感度决定是否加入幽默提示
def _should_use_humor(config: Dict[str, Any], affinity_score: int) -> bool:
    """Decide whether to add a light humor hint."""
    style_cfg = config.get("style", {})
    humor_base = float(style_cfg.get("humor_rate", 0.15))
    humor_bonus = float(style_cfg.get("humor_affinity_bonus", 0.25))
    ratio = max(0.0, min(1.0, affinity_score / 100.0))
    chance = min(0.6, humor_base + ratio * humor_bonus)
    return random.random() < chance


# 根据未回复与情绪选择触达策略
def _decide_presence_mode(
    unanswered_count: int,
    affinity_score: int,
    emotion_label: str,
    config: Dict[str, Any],
) -> str:
    """Decide whether to reduce or increase proactive reach."""
    strategy_cfg = config.get("response_strategy", {})
    if not strategy_cfg.get("enabled", True):
        return "balanced"

    reduce_after = int(strategy_cfg.get("reduce_after_unanswered", 2))
    increase_after = int(strategy_cfg.get("increase_after_unanswered", 1))
    affinity_threshold = int(strategy_cfg.get("affinity_to_increase", 70))

    if unanswered_count >= reduce_after:
        if affinity_score >= affinity_threshold or emotion_label == "sensitive":
            return "increase"
        return "reduce"
    if unanswered_count >= increase_after and emotion_label == "sensitive":
        return "increase"
    return "balanced"


# 获取用户称呼
def _resolve_user_name(stream) -> str:
    """Resolve a friendly user name."""
    try:
        return stream.user_info.user_nickname or "friend"
    except Exception:
        return "friend"


# 主动陪伴的后台任务
class PresenceAgentTask(AsyncTask):
    """Background task for proactive chat and reminders."""

    def __init__(self, plugin_name: str, state_file_path: str):
        super().__init__(task_name=TASK_NAME, wait_before_start=5, run_interval=60)
        self.plugin_name = plugin_name
        self.state_file_path = state_file_path

    def _load_config(self) -> Dict[str, Any]:
        """Load plugin config from registry."""
        return component_registry.get_plugin_config(self.plugin_name) or {}

    def _build_segments(self, config: Dict[str, Any]) -> List[SegmentRule]:
        """Build time segment rules from config."""
        raw = config.get("segments", {}).get("definitions", [])
        rules: List[SegmentRule] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                rules.append(
                    SegmentRule(
                        name=str(item.get("name", "")).strip() or "segment",
                        start_hour=int(item.get("start_hour", 6)),
                        end_hour=int(item.get("end_hour", 9)),
                        reminder=str(item.get("reminder", "")).strip() or "Take care.",
                        enabled=bool(item.get("enabled", True)),
                    )
                )
        return rules

    def _should_send_inactivity(
        self,
        now_ts: float,
        last_user_ts: Optional[float],
        last_proactive_ts: Optional[float],
        unanswered_count: int,
        affinity_score: int,
        config: Dict[str, Any],
        presence_mode: str = "balanced",
    ) -> bool:
        """Check inactivity reminder conditions."""
        general = config.get("general", {})
        threshold_minutes = int(general.get("inactivity_threshold_minutes", 120))
        cooldown_minutes = int(general.get("proactive_cooldown_minutes", 60))
        max_unanswered = int(general.get("max_unanswered", 3))
        affinity_cfg = config.get("affinity", {})
        strategy_cfg = config.get("response_strategy", {})

        if strategy_cfg.get("enabled", True):
            if presence_mode == "reduce":
                threshold_minutes += int(strategy_cfg.get("reduce_threshold_bonus_minutes", 30))
                cooldown_minutes += int(strategy_cfg.get("reduce_cooldown_bonus_minutes", 30))
            elif presence_mode == "increase":
                threshold_minutes = max(
                    int(strategy_cfg.get("min_increase_threshold_minutes", 5)),
                    threshold_minutes - int(strategy_cfg.get("increase_threshold_cut_minutes", 20)),
                )

        if affinity_cfg.get("enabled", True):
            threshold_minutes = _adjust_by_affinity(
                threshold_minutes,
                affinity_score,
                float(affinity_cfg.get("inactivity_reduce_ratio", 0.5)),
                int(affinity_cfg.get("min_inactivity_minutes", 10)),
            )
            cooldown_minutes = _adjust_by_affinity(
                cooldown_minutes,
                affinity_score,
                float(affinity_cfg.get("cooldown_reduce_ratio", 0.5)),
                int(affinity_cfg.get("min_cooldown_minutes", 10)),
            )
            max_unanswered += int(affinity_cfg.get("max_unanswered_bonus", 1 if affinity_score >= 60 else 0))

        if last_user_ts is None:
            return False
        if unanswered_count >= max_unanswered:
            return False
        if last_proactive_ts and now_ts - last_proactive_ts < cooldown_minutes * 60:
            return False
        silence_minutes = (now_ts - last_user_ts) / 60.0
        return silence_minutes >= threshold_minutes

    def _should_send_quick_check(
        self,
        now_ts: float,
        last_user_ts: Optional[float],
        last_proactive_ts: Optional[float],
        unanswered_count: int,
        affinity_score: int,
        config: Dict[str, Any],
        presence_mode: str = "balanced",
    ) -> bool:
        """Check quick follow-up when user disappears mid-chat."""
        general = config.get("general", {})
        quick_minutes = int(general.get("quick_check_minutes", 5))
        active_window = int(general.get("recent_active_window_minutes", 30))
        cooldown_minutes = int(general.get("proactive_cooldown_minutes", 60))
        max_unanswered = int(general.get("max_unanswered", 3))
        affinity_cfg = config.get("affinity", {})
        strategy_cfg = config.get("response_strategy", {})

        if strategy_cfg.get("enabled", True):
            if presence_mode == "reduce":
                quick_minutes += int(strategy_cfg.get("reduce_quick_bonus_minutes", 2))
                cooldown_minutes += int(strategy_cfg.get("reduce_cooldown_bonus_minutes", 30))
            elif presence_mode == "increase":
                quick_minutes = max(
                    int(strategy_cfg.get("min_increase_quick_minutes", 1)),
                    quick_minutes - int(strategy_cfg.get("increase_quick_cut_minutes", 2)),
                )

        if affinity_cfg.get("enabled", True):
            quick_minutes = _adjust_by_affinity(
                quick_minutes,
                affinity_score,
                float(affinity_cfg.get("quick_check_reduce_ratio", 0.5)),
                int(affinity_cfg.get("min_quick_check_minutes", 1)),
            )
            cooldown_minutes = _adjust_by_affinity(
                cooldown_minutes,
                affinity_score,
                float(affinity_cfg.get("cooldown_reduce_ratio", 0.5)),
                int(affinity_cfg.get("min_cooldown_minutes", 10)),
            )
            max_unanswered += int(affinity_cfg.get("max_unanswered_bonus", 1 if affinity_score >= 60 else 0))

        if last_user_ts is None:
            return False
        if unanswered_count >= max_unanswered:
            return False
        if last_proactive_ts and now_ts - last_proactive_ts < cooldown_minutes * 60:
            return False

        silence_minutes = (now_ts - last_user_ts) / 60.0
        if silence_minutes < quick_minutes:
            return False
        if silence_minutes > active_window:
            return False
        return True

    def _should_send_segment(
        self,
        now_ts: float,
        last_user_ts: Optional[float],
        segment: SegmentRule,
        state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> bool:
        """Check segment reminder conditions."""
        if not segment.enabled:
            return False
        hour = _current_hour()
        if not _hour_in_range(hour, segment.start_hour, segment.end_hour):
            return False
        last_sent_map = state.get("last_segment_sent", {})
        if last_sent_map.get(segment.name) == _today_str():
            return False

        general = config.get("general", {})
        skip_minutes = int(general.get("segment_skip_if_recent_minutes", 30))
        if last_user_ts and now_ts - last_user_ts < skip_minutes * 60:
            return False
        return True

    async def _generate_message(
        self,
        reason: str,
        stream,
        recent_messages: List[Any],
        segment: Optional[SegmentRule],
        config: Dict[str, Any],
        tone_style: str,
        tone_description: str,
        emotion_label: str,
        humor_hint: bool,
        presence_mode: str,
        persona_name: str,
        persona_style: str,
        persona_behavior: str,
    ) -> str:
        """Generate a proactive message using LLM with fallback."""
        general = config.get("general", {})
        messages_cfg = config.get("messages", {})
        topic_cfg = config.get("topic_continuation", {})

        model_task = str(general.get("model_task", "replyer"))
        temperature = float(general.get("temperature", 0.7))
        max_tokens = int(general.get("max_tokens", 200))
        recent_limit = int(general.get("recent_message_limit", 8))
        fallback_key = f"{reason}_fallback"
        fallback_list = messages_cfg.get(fallback_key)
        if not isinstance(fallback_list, list):
            fallback_list = messages_cfg.get("inactive_fallback", [])

        user_name = _resolve_user_name(stream)
        now_local = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        history = recent_messages[-recent_limit:]
        history_text = message_api.build_readable_messages_to_str(history, replace_bot_name=True, truncate=True)
        topic_hint = ""
        if reason == "topic_followup" and topic_cfg.get("enabled", True):
            last_user_text = _get_recent_user_text(history)
            if last_user_text:
                topic_hint = f"Last user topic: {last_user_text}\n"

        reminder_hint = segment.reminder if segment else ""
        system_hint = messages_cfg.get(
            "system_hint",
            "You are a warm, human-like companion. Keep it short, gentle, and caring.",
        )
        relation_hint = f"Relationship tone: {tone_style} ({tone_description})"
        persona_hint = f"Persona: {persona_name}. {persona_style} {persona_behavior}"
        emotion_hint = f"User emotion: {emotion_label}"
        humor_line = "Add a touch of light humor if natural." if humor_hint else "Keep it sincere and calm."
        presence_hint = f"Proactive strategy: {presence_mode}"

        prompt = (
            f"{system_hint}\n\n"
            f"Current time: {now_local}\n"
            f"User name: {user_name}\n"
            f"Reason: {reason}\n"
            f"{relation_hint}\n"
            f"{persona_hint}\n"
            f"{emotion_hint}\n"
            f"{presence_hint}\n"
            f"{humor_line}\n"
            f"Reminder hint: {reminder_hint}\n\n"
            f"{topic_hint}"
            f"Recent conversation:\n{history_text}\n\n"
            "Reply with JSON only:\n"
            '{ "message": "your proactive message" }'
        )

        available = llm_api.get_available_models()
        model_config = available.get(model_task) or available.get("replyer")
        if not model_config and available:
            model_config = list(available.values())[0]

        if not model_config:
            return _pick_fallback_message(fallback_list)

        success, response, _reasoning, _model = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="plugin.presence_agent",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not success:
            return _pick_fallback_message(fallback_list)

        data = _safe_extract_json(response)
        message = str(data.get("message", "")).strip()
        if not message:
            message = response.strip()
        if not message:
            message = _pick_fallback_message(fallback_list)
        return message

    async def _send_message(self, stream_id: str, text: str) -> bool:
        """Send a proactive message to a chat stream."""
        try:
            return await send_api.text_to_stream(text=text, stream_id=stream_id, typing=True, storage_message=True)
        except Exception as exc:
            logger.error(f"Failed to send message to {stream_id}: {exc}")
            return False

    async def _log_proactive_action(
        self,
        stream,
        text: str,
        reason: str,
        affinity_score: int,
        emotion_label: str,
        tone_style: str,
        persona_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Write proactive action info into database."""
        db_cfg = config.get("database", {})
        if not db_cfg.get("enabled", True):
            return
        if not db_cfg.get("store_action", True):
            return
        try:
            action_data = {
                "reason": reason,
                "message": text,
                "affinity_score": affinity_score,
                "emotion_label": emotion_label,
                "tone_style": tone_style,
                "persona": persona_name,
            }
            await database_api.store_action_info(
                chat_stream=stream,
                action_build_into_prompt=False,
                action_prompt_display=text,
                action_done=True,
                action_data=action_data,
                action_name="presence_agent_proactive",
            )
        except Exception as exc:
            logger.warning(f"Failed to log proactive action: {exc}")

    async def run(self):
        """Main periodic scan loop."""
        config = self._load_config()
        if not config or not config.get("plugin", {}).get("enabled", True):
            return

        general = config.get("general", {})
        scan_interval = int(general.get("scan_interval_seconds", 300))
        self.run_interval = max(30, scan_interval)

        # 仅在私聊生效，避免在群聊中主动打扰。
        private_only = True
        platform = str(general.get("platform", "qq"))
        platform_arg = chat_api.SpecialTypes.ALL_PLATFORMS if platform == "all" else platform

        streams = chat_api.get_private_streams(platform_arg) if private_only else chat_api.get_all_streams(platform_arg)
        if not streams:
            return

        state = _load_state(self.state_file_path)
        now_ts = time.time()

        for stream in streams:
            if not stream or not getattr(stream, "stream_id", None):
                continue
            if not _is_user_allowed(stream, config):
                continue
            if _is_quiet_time(now_ts, config):
                continue

            stream_id = stream.stream_id
            recent_messages = _get_recent_messages(stream_id, int(general.get("recent_message_limit", 8)))
            last_user_ts = _get_last_user_message_ts(recent_messages)
            affinity_score = _compute_affinity_score(stream, recent_messages, last_user_ts, config)
            emotion_intensity, emotion_label = _summarize_emotion(recent_messages, config)
            tone_style, tone_description = _decide_tone(affinity_score, config)
            humor_hint = _should_use_humor(config, affinity_score)
            persona_name, persona_style, persona_behavior = _resolve_persona(config)

            stream_state = state.get(stream_id, {})
            last_proactive_ts = stream_state.get("last_proactive_ts")
            unanswered_count = int(stream_state.get("unanswered_count", 0))

            if last_proactive_ts and last_user_ts and last_user_ts > last_proactive_ts:
                unanswered_count = 0

            if stream_state.get("unanswered_count") != unanswered_count:
                stream_state["unanswered_count"] = unanswered_count
                state[stream_id] = stream_state
                _save_state(self.state_file_path, state)

            presence_mode = _decide_presence_mode(
                unanswered_count,
                affinity_score,
                emotion_label,
                config,
            )

            # Unread strategy: if user ignored the last proactive message.
            unread_cfg = config.get("unread_strategy", {})
            if unread_cfg.get("enabled", True) and last_proactive_ts and last_user_ts:
                if last_user_ts < last_proactive_ts:
                    unread_window = int(unread_cfg.get("unread_window_minutes", 15))
                    if now_ts - last_proactive_ts > unread_window * 60:
                        extra_cooldown = int(unread_cfg.get("extra_cooldown_minutes", 30))
                        stream_state["last_proactive_ts"] = now_ts - max(0, extra_cooldown * 60)
                        state[stream_id] = stream_state
                        _save_state(self.state_file_path, state)

            # Quick check if user disappeared mid-chat.
            should_quick_check = self._should_send_quick_check(
                now_ts,
                last_user_ts,
                last_proactive_ts,
                unanswered_count,
                affinity_score,
                config,
                presence_mode,
            )
            if should_quick_check:
                reason = "quick_check"
                if emotion_intensity >= int(config.get("emotion", {}).get("intensity_threshold", 6)):
                    reason = "gentle_nudge"
                text = await self._generate_message(
                    reason,
                    stream,
                    recent_messages,
                    None,
                    config,
                    tone_style,
                    tone_description,
                    emotion_label,
                    humor_hint,
                    presence_mode,
                    persona_name,
                    persona_style,
                    persona_behavior,
                )
                if await self._send_message(stream_id, text):
                    await self._log_proactive_action(
                        stream,
                        text,
                        reason,
                        affinity_score,
                        emotion_label,
                        tone_style,
                        persona_name,
                        config,
                    )
                    stream_state["last_proactive_ts"] = now_ts
                    stream_state["unanswered_count"] = unanswered_count + 1
                    state[stream_id] = stream_state
                    _save_state(self.state_file_path, state)
                continue

            # Inactivity check.
            should_inactive = self._should_send_inactivity(
                now_ts,
                last_user_ts,
                last_proactive_ts,
                unanswered_count,
                affinity_score,
                config,
                presence_mode,
            )
            if should_inactive:
                reason = "inactivity"
                if config.get("topic_continuation", {}).get("enabled", True):
                    if _get_recent_user_text(recent_messages):
                        reason = "topic_followup"
                text = await self._generate_message(
                    reason,
                    stream,
                    recent_messages,
                    None,
                    config,
                    tone_style,
                    tone_description,
                    emotion_label,
                    humor_hint,
                    presence_mode,
                    persona_name,
                    persona_style,
                    persona_behavior,
                )
                if await self._send_message(stream_id, text):
                    await self._log_proactive_action(
                        stream,
                        text,
                        reason,
                        affinity_score,
                        emotion_label,
                        tone_style,
                        persona_name,
                        config,
                    )
                    stream_state["last_proactive_ts"] = now_ts
                    stream_state["unanswered_count"] = unanswered_count + 1
                    state[stream_id] = stream_state
                    _save_state(self.state_file_path, state)
                continue

            # Segment reminders.
            for segment in self._build_segments(config):
                if not self._should_send_segment(now_ts, last_user_ts, segment, stream_state, config):
                    continue
                text = await self._generate_message(
                    "segment_reminder",
                    stream,
                    recent_messages,
                    segment,
                    config,
                    tone_style,
                    tone_description,
                    emotion_label,
                    humor_hint,
                    presence_mode,
                    persona_name,
                    persona_style,
                    persona_behavior,
                )
                if await self._send_message(stream_id, text):
                    await self._log_proactive_action(
                        stream,
                        text,
                        "segment_reminder",
                        affinity_score,
                        emotion_label,
                        tone_style,
                        persona_name,
                        config,
                    )
                    last_sent_map = stream_state.get("last_segment_sent", {})
                    last_sent_map[segment.name] = _today_str()
                    stream_state["last_segment_sent"] = last_sent_map
                    state[stream_id] = stream_state
                    _save_state(self.state_file_path, state)
                break


# 插件启动事件处理器
class PresenceAgentStartHandler(BaseEventHandler):
    """Start the background proactive chat task."""

    event_type = EventType.ON_START
    handler_name = "presence_agent_start"
    handler_description = "Start proactive chat and reminder loop"
    weight = 50

    async def execute(self, message):
        """Start background loop on bot startup."""
        try:
            if not self.get_config("plugin.enabled", True):
                return True, False, None, None, None
            await _presence_task_manager.add_task(_presence_task)
            return True, False, "PresenceAgent task started", None, None
        except Exception as exc:
            logger.error(f"Failed to start PresenceAgent task: {exc}")
            return False, False, str(exc), None, None


# 插件停止事件处理器
class PresenceAgentStopHandler(BaseEventHandler):
    """Stop the background proactive chat task."""

    event_type = EventType.ON_STOP
    handler_name = "presence_agent_stop"
    handler_description = "Stop proactive chat and reminder loop"
    weight = 50

    async def execute(self, message):
        """Stop background loop on bot shutdown."""
        try:
            await _presence_task_manager.stop_and_wait_all_tasks()
            return True, False, "PresenceAgent task stopped", None, None
        except Exception as exc:
            logger.error(f"Failed to stop PresenceAgent task: {exc}")
            return False, False, str(exc), None, None


# Dedicated task manager for this plugin.
_presence_task_manager = AsyncTaskManager()

# State file path will be resolved at plugin init time.
_state_file_path = os.path.join(os.path.dirname(__file__), STATE_FILE_NAME)
_presence_task = PresenceAgentTask(PLUGIN_NAME, _state_file_path)


@register_plugin
# PresenceAgent 插件主体
class PresenceAgentPlugin(BasePlugin):
    """PresenceAgent plugin - proactive check-in and time reminders."""

    plugin_name = PLUGIN_NAME
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name = "config.toml"
    config_section_descriptions = {
        "plugin": section_meta("插件基础设置", icon="settings", order=1),
        "general": section_meta("行为与频率设置", icon="timer", order=2),
        "segments": section_meta("时间段提醒设置", icon="calendar-clock", order=3),
        "messages": section_meta("提示词与兜底消息", icon="message-circle", order=4),
        "affinity": section_meta("好感度影响设置", icon="heart", order=10),
        "emotion": section_meta("情绪强度判断", icon="smile", order=11),
        "style": section_meta("语气与幽默风格", icon="sparkles", order=12),
        "persona": section_meta("人设包设置", icon="user", order=13),
        "response_strategy": section_meta("未回复反馈策略", icon="repeat", order=14),
        "unread_strategy": section_meta("已读不回策略", icon="mail", order=15),
        "database": section_meta("数据库记录", icon="database", order=16),
        "lists": section_meta("白名单/黑名单", icon="list", order=17),
        "quiet_hours": section_meta("安静时段设置", icon="moon", order=18),
        "topic_continuation": section_meta("话题接续设置", icon="message-square", order=19),
    }

    config_schema = {
        "plugin": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用 PresenceAgent 插件",
                input_type="switch",
                order=1,
            ),
            "config_version": ConfigField(
                type=str,
                default="1.1.0",
                description="配置结构版本号",
                disabled=True,
                order=99,
            ),
        },
        "general": {
            "scan_interval_seconds": ConfigField(
                type=int,
                default=600,
                description="扫描间隔（秒）",
                min=30,
                max=3600,
                step=10,
                input_type="slider",
                hint="建议 >= 60 秒",
                order=1,
            ),
            "inactivity_threshold_minutes": ConfigField(
                type=int,
                default=1440,
                description="沉默多少分钟后主动关心",
                min=1,
                max=10080,
                order=2,
            ),
            "quick_check_minutes": ConfigField(
                type=int,
                default=2,
                description="聊天中断后多久快速询问（分钟）",
                min=1,
                max=120,
                order=3,
            ),
            "recent_active_window_minutes": ConfigField(
                type=int,
                default=30,
                description="视为“刚在聊天”的时间窗口（分钟）",
                min=5,
                max=240,
                order=4,
            ),
            "proactive_cooldown_minutes": ConfigField(
                type=int,
                default=120,
                description="主动消息冷却时间（分钟）",
                min=5,
                max=1440,
                order=5,
            ),
            "max_unanswered": ConfigField(
                type=int,
                default=2,
                description="连续未回应次数上限",
                min=0,
                max=10,
                order=6,
            ),
            "segment_skip_if_recent_minutes": ConfigField(
                type=int,
                default=15,
                description="最近有互动则跳过时间段提醒（分钟）",
                min=0,
                max=240,
                order=7,
            ),
            "recent_message_limit": ConfigField(
                type=int,
                default=30,
                description="用于生成的最近消息条数",
                min=1,
                max=60,
                order=8,
            ),
            "model_task": ConfigField(
                type=str,
                default="replyer",
                description="使用的 LLM 任务名（来自 model_task_config）",
                placeholder="replyer",
                order=9,
            ),
            "temperature": ConfigField(
                type=float,
                default=0.7,
                description="LLM 生成温度",
                min=0.0,
                max=2.0,
                step=0.1,
                input_type="slider",
                order=10,
            ),
            "max_tokens": ConfigField(
                type=int,
                default=4096,
                description="LLM 最大输出 token 数",
                min=512,
                max=8192,
                step=10,
                order=11,
            ),
            "private_only": ConfigField(
                type=bool,
                default=True,
                description="仅对私聊发送",
                input_type="switch",
                disabled=True,
                hint="当前插件固定为私聊模式",
                order=98,
            ),
            "platform": ConfigField(
                type=str,
                default="qq",
                description='平台过滤："qq" 或 "all"',
                choices=["qq", "all"],
                input_type="select",
                order=99,
            ),
        },
        "affinity": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用好感度影响频率",
                input_type="switch",
                order=1,
            ),
            "memory_point_weight": ConfigField(
                type=int,
                default=2,
                description="记忆点权重（每条记忆加分）",
                min=0,
                max=10,
                order=2,
            ),
            "memory_point_cap": ConfigField(
                type=int,
                default=30,
                description="记忆点计分上限",
                min=0,
                max=100,
                order=3,
            ),
            "know_times_weight": ConfigField(
                type=int,
                default=4,
                description="熟悉次数权重（每次加分）",
                min=0,
                max=10,
                order=4,
            ),
            "know_times_cap": ConfigField(
                type=int,
                default=20,
                description="熟悉次数计分上限",
                min=0,
                max=100,
                order=5,
            ),
            "recent_message_weight": ConfigField(
                type=int,
                default=2,
                description="最近消息权重",
                min=0,
                max=10,
                order=6,
            ),
            "recent_message_cap": ConfigField(
                type=int,
                default=20,
                description="最近消息计分上限",
                min=0,
                max=100,
                order=7,
            ),
            "recency_bonus_1d": ConfigField(
                type=int,
                default=10,
                description="1天内互动额外加分",
                min=0,
                max=50,
                order=8,
            ),
            "recency_bonus_3d": ConfigField(
                type=int,
                default=5,
                description="3天内互动额外加分",
                min=0,
                max=50,
                order=9,
            ),
            "recency_bonus_7d": ConfigField(
                type=int,
                default=2,
                description="7天内互动额外加分",
                min=0,
                max=50,
                order=10,
            ),
            "inactivity_reduce_ratio": ConfigField(
                type=float,
                default=0.5,
                description="好感度提高时，减少沉默阈值的比例上限",
                min=0.0,
                max=1.0,
                step=0.05,
                order=11,
            ),
            "cooldown_reduce_ratio": ConfigField(
                type=float,
                default=0.5,
                description="好感度提高时，减少冷却时间的比例上限",
                min=0.0,
                max=1.0,
                step=0.05,
                order=12,
            ),
            "quick_check_reduce_ratio": ConfigField(
                type=float,
                default=0.5,
                description="好感度提高时，减少快速询问时间的比例上限",
                min=0.0,
                max=1.0,
                step=0.05,
                order=13,
            ),
            "min_inactivity_minutes": ConfigField(
                type=int,
                default=10,
                description="沉默阈值最低值（分钟）",
                min=1,
                max=120,
                order=14,
            ),
            "min_cooldown_minutes": ConfigField(
                type=int,
                default=10,
                description="冷却时间最低值（分钟）",
                min=1,
                max=120,
                order=15,
            ),
            "min_quick_check_minutes": ConfigField(
                type=int,
                default=1,
                description="快速询问最低值（分钟）",
                min=1,
                max=30,
                order=16,
            ),
            "max_unanswered_bonus": ConfigField(
                type=int,
                default=1,
                description="好感度高时额外允许未回应次数",
                min=0,
                max=5,
                order=17,
            ),
        },
        "lists": {
            "allowlist": ConfigField(
                type=list,
                default=[],
                description="允许主动触达的用户 ID 列表",
                item_type="string",
                placeholder="输入用户 ID",
                max_items=50,
                order=1,
            ),
            "denylist": ConfigField(
                type=list,
                default=[],
                description="禁止主动触达的用户 ID 列表",
                item_type="string",
                placeholder="输入用户 ID",
                max_items=50,
                order=2,
            ),
        },
        "quiet_hours": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用安静时段",
                input_type="switch",
                order=1,
            ),
            "start_hour": ConfigField(
                type=int,
                default=23,
                description="安静时段开始小时（0-23）",
                min=0,
                max=23,
                order=2,
            ),
            "end_hour": ConfigField(
                type=int,
                default=7,
                description="安静时段结束小时（0-23）",
                min=0,
                max=23,
                order=3,
            ),
        },
        "emotion": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用情绪强度判断",
                input_type="switch",
                order=1,
            ),
            "keywords": ConfigField(
                type=list,
                default=["难过", "焦虑", "生气", "烦", "沮丧", "压力", "疲惫", "委屈"],
                description="情绪关键词列表",
                item_type="string",
                placeholder="输入关键词",
                max_items=30,
                hint="用于检测用户情绪强度",
                order=2,
            ),
            "recent_limit": ConfigField(
                type=int,
                default=8,
                description="情绪分析最近消息条数",
                min=1,
                max=50,
                order=3,
            ),
            "hit_cap": ConfigField(
                type=int,
                default=5,
                description="命中上限（防止过度放大）",
                min=1,
                max=20,
                order=4,
            ),
            "intensity_threshold": ConfigField(
                type=int,
                default=6,
                description="触发温柔关心的阈值",
                min=1,
                max=10,
                order=5,
            ),
        },
        "topic_continuation": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用话题接续",
                input_type="switch",
                order=1,
            ),
        },
        "unread_strategy": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用已读不回策略",
                input_type="switch",
                order=1,
            ),
            "unread_window_minutes": ConfigField(
                type=int,
                default=15,
                description="多久判定未回复（分钟）",
                min=1,
                max=240,
                order=2,
            ),
            "extra_cooldown_minutes": ConfigField(
                type=int,
                default=30,
                description="未回复时延长冷却（分钟）",
                min=0,
                max=240,
                order=3,
            ),
        },
        "database": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用数据库记录",
                input_type="switch",
                order=1,
            ),
            "store_action": ConfigField(
                type=bool,
                default=True,
                description="记录主动消息到 ActionRecords",
                input_type="switch",
                order=2,
            ),
        },
        "response_strategy": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用未回复反馈策略",
                input_type="switch",
                order=1,
            ),
            "reduce_after_unanswered": ConfigField(
                type=int,
                default=2,
                description="未回复几次后减少打扰",
                min=0,
                max=10,
                order=2,
            ),
            "increase_after_unanswered": ConfigField(
                type=int,
                default=1,
                description="未回复几次后可加大关心",
                min=0,
                max=10,
                order=3,
            ),
            "affinity_to_increase": ConfigField(
                type=int,
                default=70,
                description="触发加大关心的好感度阈值",
                min=0,
                max=100,
                order=4,
            ),
            "reduce_threshold_bonus_minutes": ConfigField(
                type=int,
                default=30,
                description="减少打扰时增加沉默阈值（分钟）",
                min=0,
                max=240,
                order=5,
            ),
            "reduce_quick_bonus_minutes": ConfigField(
                type=int,
                default=2,
                description="减少打扰时增加快速询问阈值（分钟）",
                min=0,
                max=30,
                order=6,
            ),
            "reduce_cooldown_bonus_minutes": ConfigField(
                type=int,
                default=30,
                description="减少打扰时增加冷却（分钟）",
                min=0,
                max=240,
                order=7,
            ),
            "increase_threshold_cut_minutes": ConfigField(
                type=int,
                default=20,
                description="加大关心时降低沉默阈值（分钟）",
                min=0,
                max=120,
                order=8,
            ),
            "increase_quick_cut_minutes": ConfigField(
                type=int,
                default=2,
                description="加大关心时降低快速询问阈值（分钟）",
                min=0,
                max=20,
                order=9,
            ),
            "min_increase_threshold_minutes": ConfigField(
                type=int,
                default=5,
                description="加大关心沉默阈值下限（分钟）",
                min=1,
                max=120,
                order=10,
            ),
            "min_increase_quick_minutes": ConfigField(
                type=int,
                default=1,
                description="加大关心快速询问阈值下限（分钟）",
                min=1,
                max=30,
                order=11,
            ),
        },
        "style": {
            "formal_max": ConfigField(
                type=int,
                default=30,
                description="正式语气上限（好感度阈值）",
                min=0,
                max=100,
                order=1,
            ),
            "warm_max": ConfigField(
                type=int,
                default=70,
                description="温柔语气上限（好感度阈值）",
                min=0,
                max=100,
                order=2,
            ),
            "tone_thresholds": ConfigField(
                type=dict,
                default={},
                description="自定义语气阈值映射（可选）",
                placeholder='{"formal_max": 30, "warm_max": 70}',
                hint="为空则使用 formal_max / warm_max",
                order=3,
            ),
            "tone_descriptions": ConfigField(
                type=dict,
                default={},
                description="自定义语气描述映射（可选）",
                placeholder='{"formal": "...", "warm": "...", "intimate": "..."}',
                hint="为空则使用单独的语气描述字段",
                order=4,
            ),
            "formal_description": ConfigField(
                type=str,
                default="语气礼貌克制，表达关心但不过度打扰。",
                description="正式语气描述",
                input_type="textarea",
                rows=2,
                order=5,
            ),
            "warm_description": ConfigField(
                type=str,
                default="语气温柔自然，像朋友般关心。",
                description="温柔语气描述",
                input_type="textarea",
                rows=2,
                order=6,
            ),
            "intimate_description": ConfigField(
                type=str,
                default="语气更亲近、更黏人，表达强烈在意。",
                description="亲近语气描述",
                input_type="textarea",
                rows=2,
                order=7,
            ),
            "humor_rate": ConfigField(
                type=float,
                default=0.15,
                description="幽默提示基础概率",
                min=0.0,
                max=1.0,
                step=0.05,
                order=8,
            ),
            "humor_affinity_bonus": ConfigField(
                type=float,
                default=0.25,
                description="好感度带来的幽默概率加成",
                min=0.0,
                max=1.0,
                step=0.05,
                order=9,
            ),
        },
        "persona": {
            "active_pack": ConfigField(
                type=str,
                default="warm_companion",
                description="当前启用的人设包",
                placeholder="warm_companion",
                hint="内置示例：warm_companion / gentle_caregiver / playful_buddy / calm_listener / tsundere_partner / cheerful_motivator / professional_assistant / late_night_confidant",
                order=1,
            ),
            "packs": ConfigField(
                type=list,
                default=[
                    {
                        "key": "warm_companion",
                        "name": "温柔陪伴",
                        "style_hint": "语气温柔自然，像朋友一样关心对方。",
                        "behavior_hint": "用短句，语气轻柔，适度撒娇。",
                    },
                    {
                        "key": "gentle_caregiver",
                        "name": "细心照顾",
                        "style_hint": "像贴心家人，关注作息与身体感受。",
                        "behavior_hint": "提醒吃饭休息，但不过度催促。",
                    },
                    {
                        "key": "playful_buddy",
                        "name": "俏皮朋友",
                        "style_hint": "轻松俏皮，偶尔开个小玩笑。",
                        "behavior_hint": "多用口语和语气词，保持亲近感。",
                    },
                    {
                        "key": "calm_listener",
                        "name": "安静倾听",
                        "style_hint": "稳重、冷静、支持性强。",
                        "behavior_hint": "少用感叹，多用理解与复述。",
                    },
                    {
                        "key": "tsundere_partner",
                        "name": "傲娇伙伴",
                        "style_hint": "表面嘴硬，实际很关心。",
                        "behavior_hint": "语气不太直白，但会偷偷在意。",
                    },
                    {
                        "key": "cheerful_motivator",
                        "name": "元气鼓励",
                        "style_hint": "积极阳光，擅长鼓励和打气。",
                        "behavior_hint": "多用正向词汇，适度用表情语气。",
                    },
                    {
                        "key": "professional_assistant",
                        "name": "专业助手",
                        "style_hint": "礼貌、克制、明确。",
                        "behavior_hint": "简短直给，不使用过多情绪词。",
                    },
                    {
                        "key": "late_night_confidant",
                        "name": "深夜知己",
                        "style_hint": "温和低声，适合深夜对话。",
                        "behavior_hint": "语速慢感，给人安全感。",
                    },
                ],
                description="人设包集合",
                hint="每个条目包含 key/name/style_hint/behavior_hint",
                item_type="object",
                item_fields={
                    "key": {"type": "string", "label": "标识", "placeholder": "warm_companion"},
                    "name": {"type": "string", "label": "名称", "placeholder": "温柔陪伴"},
                    "style_hint": {"type": "string", "label": "风格提示", "placeholder": "语气温柔自然"},
                    "behavior_hint": {"type": "string", "label": "行为提示", "placeholder": "用短句，语气轻柔"},
                },
                max_items=20,
                order=2,
            ),
        },
        "segments": {
            "definitions": ConfigField(
                type=list,
                default=[
                    {
                        "name": "morning",
                        "start_hour": 6,
                        "end_hour": 9,
                        "enabled": True,
                        "reminder": "早上时间到了，记得吃早餐哦。",
                    },
                    {
                        "name": "noon",
                        "start_hour": 11,
                        "end_hour": 13,
                        "enabled": True,
                        "reminder": "中午了，记得吃午饭、稍微休息一下。",
                    },
                    {
                        "name": "evening",
                        "start_hour": 18,
                        "end_hour": 20,
                        "enabled": True,
                        "reminder": "傍晚了，别忘了吃晚饭呀。",
                    },
                    {
                        "name": "late_night",
                        "start_hour": 22,
                        "end_hour": 1,
                        "enabled": True,
                        "reminder": "夜深了，早点休息吧。",
                    },
                ],
                description="时间段提醒定义",
                item_type="object",
                item_fields={
                    "name": {"type": "string", "label": "名称", "placeholder": "morning"},
                    "start_hour": {"type": "number", "label": "开始小时", "min": 0, "max": 23},
                    "end_hour": {"type": "number", "label": "结束小时", "min": 0, "max": 23},
                    "enabled": {"type": "bool", "label": "是否启用"},
                    "reminder": {"type": "string", "label": "提醒文案", "placeholder": "记得吃饭哦"},
                },
                min_items=0,
                max_items=10,
                order=1,
            )
        },
        "messages": {
            "system_hint": ConfigField(
                type=str,
                default="你是一个温暖、拟人化的陪伴者，语气简短、温柔、关心。",
                description="LLM 系统提示词",
                input_type="textarea",
                rows=4,
                order=1,
            ),
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
                order=2,
            ),
            "quick_check_fallback": ConfigField(
                type=list,
                default=[
                    "嗯？你刚刚还在吗？",
                    "我还在这儿，随时都可以聊。",
                    "要不要继续刚才的话题？",
                ],
                description="聊天中断时的快速关心兜底消息",
                item_type="string",
                max_items=20,
                order=3,
            ),
            "gentle_nudge_fallback": ConfigField(
                type=list,
                default=[
                    "我在呢，有需要就跟我说。",
                    "如果你心情不太好，我一直都在。",
                    "想聊聊也可以的，我会认真听。",
                ],
                description="情绪强度高时的温柔关心兜底消息",
                item_type="string",
                max_items=20,
                order=4,
            ),
            "topic_followup_fallback": ConfigField(
                type=list,
                default=[
                    "刚才说到的那件事，还要继续聊吗？",
                    "你刚提到的话题我还记得呢。",
                    "要不要把刚才的话题聊完？",
                ],
                description="话题接续的兜底消息",
                item_type="string",
                max_items=20,
                order=5,
            ),
            "segment_reminder_fallback": ConfigField(
                type=list,
                default=[
                    "我来提醒一下，记得照顾好自己。",
                    "别太忙啦，记得吃饭和休息。",
                ],
                description="时间段提醒的兜底消息",
                item_type="string",
                max_items=20,
                order=6,
            ),
        },
    }

    config_layout = ConfigLayout(
        type="tabs",
        tabs=[
            ConfigTab(
                id="basic",
                title="基础",
                sections=["plugin", "general", "segments"],
                icon="settings",
                order=1,
            ),
            ConfigTab(
                id="behavior",
                title="行为",
                sections=[
                    "affinity",
                    "emotion",
                    "style",
                    "persona",
                    "response_strategy",
                    "unread_strategy",
                    "database",
                    "lists",
                    "quiet_hours",
                    "topic_continuation",
                ],
                icon="activity",
                order=2,
            ),
            ConfigTab(
                id="messages",
                title="文案",
                sections=["messages"],
                icon="message-circle",
                order=3,
            ),
        ],
    )

    def _generate_and_save_default_config(self, config_file_path: str) -> None:
        """Generate default config with AoT for list-of-dict fields."""
        if not self.config_schema:
            logger.debug(f"{self.log_prefix} 插件未定义config_schema，不生成配置文件")
            return

        toml_str = f"# {self.plugin_name} - 自动生成的配置文件\n"
        plugin_description = self.get_manifest_info("description", "插件配置文件")
        toml_str += f"# {plugin_description}\n\n"

        for section, fields in self.config_schema.items():
            if section in self.config_section_descriptions:
                toml_str += f"# {self.config_section_descriptions[section]}\n"

            toml_str += f"[{section}]\n\n"

            if isinstance(fields, dict):
                for field_name, field in fields.items():
                    if not isinstance(field, ConfigField):
                        continue

                    toml_str += f"# {field.description}"
                    if field.required:
                        toml_str += " (必需)"
                    toml_str += "\n"

                    if field.example:
                        toml_str += f"# 示例: {field.example}\n"

                    if field.choices:
                        choices_str = ", ".join(map(str, field.choices))
                        toml_str += f"# 可选值: {choices_str}\n"

                    value = field.default
                    if (
                        isinstance(value, list)
                        and value
                        and all(isinstance(item, dict) for item in value)
                    ):
                        for item in value:
                            toml_str += f"[[{section}.{field_name}]]\n"
                            for item_key, item_value in item.items():
                                toml_str += f"{item_key} = {self._format_toml_value(item_value)}\n"
                            toml_str += "\n"
                    else:
                        toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"

            toml_str += "\n"

        try:
            with open(config_file_path, "w", encoding="utf-8") as f:
                f.write(toml_str)
            logger.info(f"{self.log_prefix} 已生成默认配置文件: {config_file_path}")
        except IOError as e:
            logger.error(f"{self.log_prefix} 保存默认配置文件失败: {e}", exc_info=True)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Update state file path to plugin directory.
        global _state_file_path, _presence_task
        _state_file_path = os.path.join(self.plugin_dir, STATE_FILE_NAME)
        _presence_task = PresenceAgentTask(self.plugin_name, _state_file_path)

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Any]]:
        """Register event handlers."""
        return [
            (PresenceAgentStartHandler.get_handler_info(), PresenceAgentStartHandler),
            (PresenceAgentStopHandler.get_handler_info(), PresenceAgentStopHandler),
        ]








