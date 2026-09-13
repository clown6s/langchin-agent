"""轻量内存会话/状态存储（生产可切换 MySQL）。"""
from ..domain.models import ChatMessage, ConversationState, RequestContext
from ..domain.ports import ConversationStore, SessionStateStore


class MemoryConversationStore(ConversationStore):
    def __init__(self):
        self._history: dict[str, list[ChatMessage]] = {}

    def get_or_create(self, conversation_id: str, *, context: RequestContext) -> None:
        self._history.setdefault(conversation_id, [])

    def append(self, conversation_id: str, message: ChatMessage, *,
               context: RequestContext) -> None:
        self._history.setdefault(conversation_id, []).append(message)

    def recent(self, conversation_id: str, *, context: RequestContext) -> list[ChatMessage]:
        return list(self._history.get(conversation_id, []))


class MemorySessionStateStore(SessionStateStore):
    def __init__(self):
        self._states: dict[str, ConversationState] = {}

    def get(self, conversation_id: str, *, context: RequestContext) -> ConversationState | None:
        return self._states.get(conversation_id)

    def save(self, conversation_id: str, state: ConversationState, *,
             context: RequestContext) -> None:
        self._states[conversation_id] = state