"""飞书消息处理器：协议转换与 ChatCoordinator 解耦。"""
from uuid import uuid4

from ...application.chat import ChatCoordinator
from ...domain.models import ChatRequest, RequestContext
from .events import parse_text_event
from .ports import FeishuClient


class FeishuMessageHandler:
    def __init__(self, *, client: FeishuClient, coordinator: ChatCoordinator):
        self._client = client
        self._coordinator = coordinator

    def handle(self, event) -> bool:
        parsed = parse_text_event(event)
        if parsed is None:
            return False
        request = ChatRequest(
            question=parsed.text,
            conversation_id=f"feishu:{parsed.chat_id}",
            context=RequestContext(trace_id=parsed.trace_id or str(uuid4())),
        )
        result = self._coordinator.ask(request)
        return self._client.reply_text(parsed.message_id, result.text)
