"""OpenTelemetry setup with an optional OTLP HTTP exporter."""
from __future__ import annotations

import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider: TracerProvider | None = None


def setup_tracing() -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(resource=Resource.create({
        SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "langchin-agent"),
    }))
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "").strip()
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            # OTLP HTTP exporter 不会自动补 /v1/traces，必须显式带上，
            # 否则请求打到根路径而被 collector/tempo 以 404 拒绝。
            if not endpoint.rstrip("/").endswith("/v1/traces"):
                endpoint = endpoint.rstrip("/") + "/v1/traces"
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, timeout=5)))
        except Exception:
            pass
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    """Flush the process provider without replacing the global provider.

    OpenTelemetry permits setting the global provider only once. Keeping the
    reference avoids noisy override warnings when test clients or application
    lifespans are created more than once in one process.
    """
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            # Exporter shutdown must never break application shutdown.
            pass


def tracer(name: str = "langchin-agent"):
    return trace.get_tracer(name)


@contextmanager
def span(name: str, **attributes):
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
