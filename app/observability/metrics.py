"""Prometheus metrics with low-cardinality labels only."""
from time import perf_counter

from prometheus_client import Counter, Histogram

from .tracing import span

http_requests_total = Counter("http_requests_total", "HTTP requests", ["status"])
agent_requests_total = Counter("agent_requests_total", "Agent requests", ["status"])
llm_requests_total = Counter("llm_requests_total", "LLM calls", ["kind", "status"])
llm_duration_seconds = Histogram("llm_duration_seconds", "LLM call duration", ["kind"])
tool_calls_total = Counter("tool_calls_total", "Tool calls", ["tool", "status"])
nl2sql_total = Counter("nl2sql_total", "NL2SQL calls", ["status"])
rag_retrieval_total = Counter("rag_retrieval_total", "RAG calls", ["status"])
sop_total = Counter("sop_total", "SOP calls", ["status"])


def instrument_llm(llm, messages, *, kind: str):
    """Invoke ``llm`` once while recording a span and low-cardinality metrics.

    Call at each real LLM call site so metrics are counted exactly once,
    regardless of whether the model is a lazy wrapper or a bound instance.
    """
    started = perf_counter()
    try:
        with span("llm.invoke", kind=kind,
                  model=getattr(llm, "model_name", None) or ""):
            result = llm.invoke(messages)
        llm_requests_total.labels(kind, "success").inc()
        return result
    except Exception:
        llm_requests_total.labels(kind, "error").inc()
        raise
    finally:
        llm_duration_seconds.labels(kind).observe(perf_counter() - started)

