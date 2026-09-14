"""领域实体与值对象。

这里不导入 LangChain / FastAPI / 数据库，只用 Python 数据类表达业务概念。

契约分层（禁止跨越）：
- ChatRequest：外部归一化输入（网络/飞书/脚本共用）。
- ConversationState：业务会话状态，只由 application 层拥有与写入。
- CapabilityResult：业务能力（SQL/SOP/知识/普通回答）的统一结果。
- Agent 运行状态（本轮消息/tool call/循环计数）属 agent 层，不放在这里。
"""
from dataclasses import dataclass, field
from typing import Any


# ---------- 请求上下文 ----------
@dataclass(frozen=True)
class RequestContext:
    """一次请求的追踪上下文（trace_id）。"""
    trace_id: str = ""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


# ---------- 外部归一化输入 ----------
@dataclass(frozen=True)
class ChatRequest:
    """所有入口（API/飞书/CLI）统一转成此结构再进用例。"""
    question: str
    conversation_id: str = "default"
    confirmed: bool = False
    context: RequestContext = field(default_factory=RequestContext)


# ---------- 业务会话状态 ----------
@dataclass(frozen=True)
class ConversationState:
    """跨轮业务会话状态，用于追问解析与域/流程继承。

    仅由 application 会话层（ChatCoordinator）构造和写入；
    agent 与工具只读传入，不得修改或持久化。
    """
    last_intent: str | None = None
    last_domain: str | None = None
    last_query_id: str | None = None
    current_sop_id: str | None = None
    current_sop_version: str | None = None
    current_step_id: str | None = None
    history: tuple[ChatMessage, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def inheritable_domain(self) -> str | None:
        """返回可继承的上一轮数据域。"""
        return self.last_domain


# ---------- 领域解析 ----------
@dataclass(frozen=True)
class DomainCandidate:
    name: str
    score: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainResolution:
    """数据域解析结果，明确区分未知/歧义/未授权。"""
    decision: str  # resolved | unknown | ambiguous | unauthorized
    domain: str | None = None
    candidates: tuple[DomainCandidate, ...] = field(default_factory=tuple)
    margin: float = 0.0
    confidence: float = 0.0
    reason: str = ""


# ---------- 业务能力结果 ----------
@dataclass(frozen=True)
class CapabilityResult:
    """一次业务能力执行的统一结果（SQL/SOP/知识/普通回答）。"""
    capability: str    # query_data | guide_sop | enterprise_knowledge | answer
    status: str        # answered | no_tables | no_sop | confirmation | llm_error | ...
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def with_text(self, text: str) -> "CapabilityResult":
        return CapabilityResult(self.capability, self.status, text, self.payload)


# ---------- 数据查询 ----------
@dataclass(frozen=True)
class SqlResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    query_id: str
    row_count: int
    truncated: bool = False


# ---------- 引用 ----------
@dataclass(frozen=True)
class Citation:
    source_type: str
    source_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    title: str | None = None


# ---------- 回答结果 ----------
@dataclass(frozen=True)
class AgentAnswer:
    """Agent 对一轮用户请求的最终回答与溯源。"""
    text: str
    status: str  # answered | clarification | no_context | rejected
    used_tools: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    sql: dict[str, Any] | None = None
    sop: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


# ---------- 审计 ----------
@dataclass(frozen=True)
class AuditEvent:
    """一次可审计业务事件；只追加，字段与日志/指标保持脱敏约定。"""
    event_id: str
    trace_id: str
    stage: str
    status: str
    conversation_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, stage: str, status: str, context: RequestContext,
            conversation_id: str | None, payload: dict[str, Any],
            event_id: str | None = None) -> "AuditEvent":
        from uuid import uuid4
        return cls(
            event_id=event_id or str(uuid4()),
            trace_id=context.trace_id or "",
            stage=stage, status=status,
            conversation_id=conversation_id, payload=payload)


# 旧名称仅为迁移期兼容；新代码统一使用 ConversationState。
SessionState = ConversationState