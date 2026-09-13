"""Resolve the request trace id used by logs and domain contexts."""
from uuid import uuid4

import structlog
from fastapi import Request


def resolve_trace_id(request: Request) -> str:
    context = structlog.contextvars.get_contextvars()
    current = context.get("trace_id")
    if isinstance(current, str) and current:
        return current
    return request.headers.get("X-Trace-Id") or str(uuid4())
