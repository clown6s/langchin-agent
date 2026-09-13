"""Application-port adapter for OpenTelemetry spans."""
from .tracing import span


class OtelTraceProvider:
    def span(self, name: str, **attributes):
        return span(name, **attributes)
