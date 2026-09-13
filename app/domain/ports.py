"""领域端口（Port）：框架无关的抽象，由 adapters 层实现。

依赖方向：domain ← application ← agent ← adapters。
所有用例只依赖这里的 Protocol，不依赖具体实现。
"""
from typing import Any, Protocol

from .models import (Citation, ChatMessage, CapabilityResult, ConversationState,
                     DomainResolution, RequestContext, SqlResult)


class Chats(Protocol):
    """可对话的 LLM（结构化出参）。"""
    def chat(self, messages: list[ChatMessage]) -> str: ...
    def close(self) -> None: ...


class SchemaCatalog(Protocol):
    """已登记的数据域/表/列/指标目录——判定的权威来源。"""
    def domains(self, *, context: RequestContext) -> list[str]: ...
    def describe(self, domain: str, *, context: RequestContext) -> str: ...
    def resolve(self, question: str, *, context: RequestContext) -> DomainResolution: ...
    def allowed_tables(self, domain: str, *, context: RequestContext) -> set[str]: ...
    def allowed_columns(self, domain: str, *, context: RequestContext) -> dict[str, set[str]]: ...
    def domain_limits(self, domain: str, *, context: RequestContext) -> tuple[int, int]: ...


class ReadonlySqlExecutor(Protocol):
    """只读 SQL 执行器（带 Guard / 行权）。"""
    def query(self, sql: str, *, domain: str,
              context: RequestContext,
              timeout_s: float = 10.0,
              max_rows: int = 1000) -> SqlResult: ...


class DataQueryService(Protocol):
    """把自然语言问题解析并执行为数据查询的用例入口。"""
    def answer(self, question: str, *, domain: str,
               context: RequestContext,
               state: ConversationState | None = None) -> CapabilityResult: ...


class SopService(Protocol):
    """SOP 流程用例入口。"""
    def handle(self, question: str, *, context: RequestContext,
               state: ConversationState | None = None,
               confirmed: bool = False) -> CapabilityResult: ...


class KnowledgeRetriever(Protocol):
    """企业知识检索端口。返回 (text, document_id, chunk_id, score) 四元组列表。"""
    def retrieve(self, query: str, *, top_k: int = 5,
                 context: RequestContext) -> list[tuple[str, str, str, float]]: ...
    def close(self) -> None: ...


class GeneralAnswerer(Protocol):
    """通用/法规/技术知识兜底作答端口。"""
    def answer(self, question: str, *, context: RequestContext) -> str: ...
    def close(self) -> None: ...


class ConversationStore(Protocol):
    def get_or_create(self, conversation_id: str, *, context: RequestContext) -> None: ...
    def append(self, conversation_id: str, message: ChatMessage, *, context: RequestContext) -> None: ...
    def recent(self, conversation_id: str, *, context: RequestContext) -> list[ChatMessage]: ...


class SessionStateStore(Protocol):
    def get(self, conversation_id: str, *, context: RequestContext) -> ConversationState | None: ...
    def save(self, conversation_id: str, state: ConversationState, *, context: RequestContext) -> None: ...


class AuditLogger(Protocol):
    def record(self, *, stage: str, status: str, context: RequestContext,
               conversation_id: str | None,
               payload: dict[str, Any]) -> None: ...