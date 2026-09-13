"""Structured audit logging adapter.

Audit records are emitted to the configured application log only;
there is no Elasticsearch or database side effect.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from ..domain.models import AuditEvent, RequestContext

logger = structlog.get_logger(__name__)
_SENSITIVE = re.compile(
    r"(?i)(bearer\s+|sk-|api[_-]?key|password|token|authorization)"
    r"\s*[:=]?\s*[^\s,;]+"
)
_DROP_KEYS = {"password", "token", "authorization", "api_key", "prompt",
              "answer", "question", "rows"}


def _safe(value: Any, *, limit: int = 500) -> Any:
    if isinstance(value, str):
        text = _SENSITIVE.sub(r"\1[REDACTED]", value)
        return text[:limit]
    if isinstance(value, dict):
        return {str(k): _safe(v, limit=limit) for k, v in value.items()
                if str(k).lower() not in _DROP_KEYS}
    if isinstance(value, (list, tuple)):
        return [_safe(item, limit=limit) for item in value[:20]]
    return value


class StructlogAuditLogger:
    """AuditLogger implementation that writes sanitized structured events."""

    def record(self, *, stage: str, status: str, context: RequestContext,
               conversation_id: str | None, payload: dict[str, Any]) -> None:
        event = AuditEvent.new(stage=stage, status=status, context=context,
                               conversation_id=conversation_id, payload=payload)
        logger.info("audit_event", event_id=event.event_id,
                    trace_id=event.trace_id, stage=event.stage,
                    status=event.status, user_id=event.user_id,
                    conversation_id=event.conversation_id,
                    payload=_safe(event.payload))

    def record_event(self, event: AuditEvent) -> None:
        self.record(stage=event.stage, status=event.status,
                    context=RequestContext(user_id=event.user_id,
                                           trace_id=event.trace_id),
                    conversation_id=event.conversation_id,
                    payload=event.payload)

    def close(self) -> None:
        pass
