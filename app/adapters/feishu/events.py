"""飞书事件解析：把 lark 事件或字典转为渠道 DTO，不引入 SDK 类型。"""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class FeishuTextEvent:
    event_id: str
    message_id: str
    chat_id: str
    chat_type: str
    open_id: str
    text: str
    trace_id: str


def parse_text_event(event) -> FeishuTextEvent | None:
    """从 lark-oapi 事件或等价字典提取文本消息。"""
    if isinstance(event, dict):
        event_id = str(event.get("event_id", "") or "")
        message = event.get("message", event)
        sender = event.get("sender", {})
    else:
        event_id = str(getattr(event, "event_id", "") or "")
        message = getattr(event, "message", event)
        sender = getattr(event, "sender", {})

    def value(obj, name, default=""):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default) or default

    message_id = value(message, "message_id")
    chat_id = value(message, "chat_id")
    chat_type = value(message, "chat_type", "p2p")
    message_type = value(message, "message_type", "text")
    if message_type != "text" or not message_id or not chat_id:
        return None
    raw = value(message, "content", "")
    try:
        text = json.loads(raw).get("text", "") if isinstance(raw, str) else value(raw, "text")
    except (TypeError, ValueError):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    sender_id = value(sender, "sender_id", sender)
    open_id = value(sender_id, "open_id")
    return FeishuTextEvent(
        event_id=event_id, message_id=str(message_id), chat_id=str(chat_id),
        chat_type=str(chat_type), open_id=str(open_id), text=text.strip(),
        trace_id=str(event_id or message_id),
    )
