# Mai Presence Agent Plugin

MaiBot PresenceAgent 插件：基于 LLM 的拟人化主动关心与定时提醒。支持情绪感知、亲密度语气、人设包、安静时段与未回复策略。

## 功能亮点
- 私聊专用的主动关心：聊天中断快速询问、长时间沉默主动关心。
- 时间段提醒：早/中/晚/深夜可配置提醒内容。
- 情绪感知：根据关键词判断情绪强度，给出更温柔的关怀。
- 亲密度驱动语气：亲密度高更黏人，低则更克制礼貌。
- 人设包：一键切换语气风格（温柔陪伴/俏皮朋友/深夜知己等）。
- 未回复反馈机制：可减少打扰或加大关心。
- 纯配置化：无需改代码即可调节频率、语气、提醒与策略。

## 安装
把本插件目录放到 MaiBot 的 `plugins/` 下：
```
MaiBot-main/plugins/PresenceAgentPlugin
```
重启 MaiBot 后会自动加载。

## 配置
插件首次运行会自动生成 `config.toml`。你可以调整如下配置项：

- `general`: 扫描频率、快速询问/沉默阈值、LLM 生成参数等
- `segments`: 时间段提醒（支持跨午夜）
- `emotion`: 情绪关键词、阈值
- `affinity`: 亲密度权重
- `style`: 语气切换阈值、幽默概率
- `response_strategy`: 未回复后的“减少打扰/加大关心”策略
- `persona`: 人设包与默认人设

### 人设包（内置）
- `warm_companion` 温柔陪伴
- `gentle_caregiver` 细心照顾
- `playful_buddy` 俏皮朋友
- `calm_listener` 安静倾听
- `tsundere_partner` 傲娇伙伴
- `cheerful_motivator` 元气鼓励
- `professional_assistant` 专业助手
- `late_night_confidant` 深夜知己

切换方式：
```
[persona]
active_pack = "playful_buddy"
```

## 建议测试流程
1. 将 `scan_interval_seconds` 调低（例如 30 秒）。
2. 将 `quick_check_minutes` 设为 1，`inactivity_threshold_minutes` 设为 2。
3. 私聊几句后停顿，观察是否触发主动消息。

## 注意
- 默认只在私聊中生效，避免群聊打扰。
- 插件会写入 `presence_agent_state.json` 保存状态（自动生成）。

## WebUI 备用方案
如果 WebUI 打不开配置页面，但你仍想使用图形化编辑，请用本插件目录中的 `plugin_routes.py` 替换 MaiBot 项目根目录的 `src/webui/plugin_routes.py` 文件后再重启。

## 作者
cuisy78, Codex
