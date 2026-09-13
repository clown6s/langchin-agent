"""飞书消息处理器：协议转换与 ChatCoordinator 解耦。"""
from uuid import uuid4

from ...application.chat import ChatCoordinator
from ...domain.models import ChatRequest, RequestContext
from .events import parse_text_event
from .ports import FeishuClient, IdentityResolver


class FeishuMessageHandler:
    def __init__(self, *, app_id: str,
                 identity: IdentityResolver, client: FeishuClient,
                 coordinator: ChatCoordinator):
        self._app_id = app_id
        self._identity = identity
        self._client = client
        self._coordinator = coordinator

    def handle(self, event) -> bool:
        parsed = parse_text_event(event)
        if parsed is None:
            return False
        context = self._identity.resolve(
            app_id=self._app_id, open_id=parsed.open_id)
        if context is None:
            self._client.reply_text(parsed.message_id, "当前飞书账号尚未绑定，请联系管理员。")
            return False
        request = ChatRequest(
            question=parsed.text,
            conversation_id=f"feishu:{parsed.chat_id}",
            context=RequestContext(
                user_id=context.user_id,
                roles=context.roles, permissions=context.permissions,
                authenticated=True, trace_id=parsed.trace_id or str(uuid4())),
        )
        result = self._coordinator.ask(request)
        return self._client.reply_text(parsed.message_id, result.text)
