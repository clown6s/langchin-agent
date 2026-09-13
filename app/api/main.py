"""FastAPI 入口：把 ChatCoordinator 暴露为 /chat。"""
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..application.chat import ChatCoordinator
from ..domain.models import ChatRequest as DomainChatRequest, RequestContext
from ..observability import metrics
from ..observability.logging import close_logging, setup_logging
from ..observability.trace_context import resolve_trace_id
from ..observability.tracing import setup_tracing, shutdown_tracing, span
from .sse import sse

logger = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    conversation_id: str = "default"
    confirmed: bool = False


class ChatResponse(BaseModel):
    answer: str
    status: str
    used_tools: list[str] = []
    conversation_id: str = ""


def _no_auth_context() -> RequestContext:
    """默认拒绝未配置认证的应用，禁止用请求头伪造身份。"""
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="未配置认证提供方")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    setup_logging()
    setup_tracing()
    try:
        yield
    finally:
        shutdown_tracing()
        close_logging()


def create_app(coordinator: ChatCoordinator,
               context_provider: Callable[[], RequestContext] = _no_auth_context) -> FastAPI:
    """创建 API；context_provider 必须返回已验证身份生成的上下文。"""
    app = FastAPI(title="langchin-agent", lifespan=_lifespan)

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or str(uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id, method=request.method, path=request.url.path)
        try:
            with span("http.request", http_method=request.method,
                      http_route=request.url.path):
                response = await call_next(request)
        except Exception:
            logger.exception("request_failed")
            raise
        response.headers["X-Trace-Id"] = trace_id
        metrics.http_requests_total.labels(str(response.status_code)).inc()
        logger.info("request_completed", status_code=response.status_code)
        return response

    @app.post("/chat")
    async def chat(req: ChatRequest,
                   request: Request,
                   context: RequestContext = Depends(context_provider)):
        trace_id = resolve_trace_id(request)
        if not context.trace_id:
            context = replace(context, trace_id=trace_id)
        domain_request = DomainChatRequest(
            question=req.question, conversation_id=req.conversation_id,
            confirmed=req.confirmed, context=context)

        def event_stream():
            for evt in coordinator.ask_stream(domain_request):
                yield sse(evt)

        return StreamingResponse(event_stream(),
                                 media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def make_agent_app(*, context_provider: Callable[[], RequestContext] = _no_auth_context,
                   enable_rag: bool = False) -> FastAPI:
    """便捷工厂：装配 Agent，并显式注入认证上下文提供方。"""
    from ..adapters.audit import StructlogAuditLogger
    from ..adapters.memory_stores import (MemoryConversationStore,
                                          MemorySessionStateStore)
    from ..application.conversation import ConversationUseCase
    from ..bootstrap import build_agent

    built = build_agent(enable_rag=enable_rag)
    coordinator = ChatCoordinator(
        agent=built["agent"], conversation=MemoryConversationStore(),
        sessions=MemorySessionStateStore(), understanding=ConversationUseCase(),
        audit=StructlogAuditLogger())
    return create_app(coordinator, context_provider)


# 通过 uvicorn 工厂或外部组合根创建 app；模块导入不再偷偷伪造用户身份。
app = make_agent_app()
