"""端到端编排用例：把会话理解、Agent 执行、状态持久化串起来。

API 层只调用这里；本层依赖 domain 端口与 agent，不依赖框架细节。
"""
from time import perf_counter
import structlog

from ..domain.models import (AgentAnswer, ChatMessage, ChatRequest,
                             ConversationState, RequestContext)
from ..domain.ports import (AuditLogger, ConversationStore, SessionStateStore)
from ..observability import metrics
from ..observability.tracing import span
from .conversation import ConversationUseCase
from ..agent.graph import LangGraphAgent

logger = structlog.get_logger(__name__)


class ChatCoordinator:
    def __init__(self, agent: LangGraphAgent,
                 conversation: ConversationStore,
                 sessions: SessionStateStore,
                 understanding: ConversationUseCase | None = None,
                 audit: AuditLogger | None = None):
        self._agent = agent
        self._conversation = conversation
        self._sessions = sessions
        self._understanding = understanding or ConversationUseCase()
        self._audit_logger = audit

    def _audit(self, *, stage: str, status: str, context: RequestContext,
               conversation_id: str | None, payload: dict) -> None:
        if self._audit_logger is None:
            return
        try:
            self._audit_logger.record(stage=stage, status=status, context=context,
                               conversation_id=conversation_id, payload=payload)
        except Exception:
            logger.exception("chat_audit_record_failed",
                             trace_id=context.trace_id, stage=stage)

    def ask(self, request: ChatRequest) -> AgentAnswer:
        started = perf_counter()
        conversation_id = request.conversation_id
        context = request.context
        question = request.question
        self._audit(stage="request", status="started", context=context,
                    conversation_id=conversation_id, payload={"question": question})
        metrics.agent_requests_total.labels("started").inc()
        try:
            with span("chat.ask", conversation_id=conversation_id,
                      user_id=context.user_id or None):
                return self._ask_impl(request)
        except Exception:
            self._audit(stage="request", status="error", context=context,
                        conversation_id=conversation_id, payload={})
            metrics.agent_requests_total.labels("error").inc()
            raise
        finally:
            duration = perf_counter() - started
            logger.info("request_completed", conversation_id=conversation_id,
                        user_id=context.user_id, duration_ms=int(duration * 1000))

    def _ask_impl(self, request: ChatRequest) -> AgentAnswer:
        conversation_id = request.conversation_id
        context = request.context
        question = request.question
        self._conversation.get_or_create(conversation_id, context=context)
        history = self._conversation.recent(conversation_id, context=context)
        state = self._sessions.get(conversation_id, context=context)

        # 1) 会话理解：改写为自包含问题，判定是否可继承域。
        turn = self._understanding.understand(question, context=context, state=state)
        condensed = turn.question or question
        if turn.needs_clarification:
            self._audit(stage="request", status="clarification", context=context,
                        conversation_id=conversation_id, payload={})
            metrics.agent_requests_total.labels("clarification").inc()
            return AgentAnswer("请补充你的问题。", "clarification")

        # 2) 绑定上下文并运行 Agent（SQL/SOP/RAG 工具决策 + 普通回答兜底）。
        state_ctx = ConversationState(
            last_intent="agent",
            last_domain=state.inheritable_domain() if (state and turn.inheritable) else None,
            history=tuple(history[-8:]),
            extra={**(state.extra if state else {}), "last_question": question,
                   "topic": question[:120]},
        )
        self._agent.bind_context(context, state=state_ctx)
        out = self._agent.chat(condensed)
        answer = AgentAnswer(
            text=out["answer"], status="answered",
            used_tools=tuple(out["used_tools"]),
        )

        # 3) 持久化：从工具结果里提取 SOP 推进态，业务状态由会话层独占写入。
        sop_state = self._extract_sop_state(out.get("tool_results", {}),
                                            state if state else None)
        self._conversation.append(conversation_id,
                                  ChatMessage("user", question), context=context)
        self._conversation.append(conversation_id,
                                  ChatMessage("assistant", out["answer"]), context=context)
        new_state = ConversationState(
            last_intent="agent", last_domain=state_ctx.last_domain,
            current_sop_id=sop_state["sop_id"],
            current_sop_version=sop_state["sop_version"],
            current_step_id=sop_state["step_id"],
            history=tuple(self._conversation.recent(conversation_id, context=context)[-8:]),
            extra={"last_question": question, "topic": condensed[:120],
                   "used_tools": list(out["used_tools"])},
        )
        self._sessions.save(conversation_id, new_state, context=context)
        self._audit(stage="request", status="answered", context=context,
                    conversation_id=conversation_id,
                    payload={"used_tools": list(out["used_tools"])})
        metrics.agent_requests_total.labels("answered").inc()
        return answer

    @staticmethod
    def _extract_sop_state(tool_results: dict, prev: ConversationState | None) -> dict:
        """从 guide_sop 工具结果提取当前 SOP 步骤；未推进则沿用上一轮。"""
        import json
        prev_sop = (prev.current_sop_id if prev else None)
        prev_ver = (prev.current_sop_version if prev else None)
        prev_step = (prev.current_step_id if prev else None)
        item = tool_results.get("guide_sop")
        if not item:
            return {"sop_id": prev_sop, "sop_version": prev_ver, "step_id": prev_step}
        try:
            data = json.loads(item.get("text") or "{}")
        except (ValueError, TypeError):
            data = {}
        payload = data.get("payload") or {}
        return {
            "sop_id": payload.get("sop_id") or prev_sop,
            "sop_version": None,
            "step_id": payload.get("step_id") or prev_step,
        }