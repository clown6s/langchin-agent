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


def instrument_llm_stream(llm, messages, *, kind: str):
    """Invoke ``llm`` via streaming, accumulating into a full message.

    与 instrument_llm 等价，但用 llm.stream 逐块累加，使 LangGraph
    stream_mode=\"messages\" 能捕获 token 增量。调用点每处只记一次指标。
    """
    started = perf_counter()
    try:
        with span("llm.invoke", kind=kind,
                  model=getattr(llm, "model_name", None) or ""):
            full = None
            for chunk in llm.stream(messages):
                full = chunk if full is None else full + chunk
            result = full
        llm_requests_total.labels(kind, "success").inc()
        return result
    except Exception:
        llm_requests_total.labels(kind, "error").inc()
        raise
    finally:
        llm_duration_seconds.labels(kind).observe(perf_counter() - started)

