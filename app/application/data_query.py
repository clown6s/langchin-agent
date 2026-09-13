"""数据查询用例：自然语言问题 → 结构化执行。

职责边界：
- 用 catalog.resolve 把问题落到已登记数据域（名称 + 权限），解析不出/歧义则返回澄清，不执行。
- 生成 SQL 前把 schema 作为唯一权威注入，禁止自由选择未登记表/列。
- 最终的 SQL 执行一律走 ReadonlySqlExecutor（内含 Guard / 只读 / 白名单）。
本用例只负责“数据查询”这条能力，不判断问题该不该查库（那是 agent 层的事）。
"""
import json

from ..domain.models import (CapabilityResult, ChatMessage, ConversationState,
                             DomainResolution, RequestContext, SqlResult)
from ..domain.ports import Chats, ReadonlySqlExecutor, SchemaCatalog
from ..observability import metrics
from ..observability.tracing import span


class DataQueryUseCase:
    def __init__(self, catalog: SchemaCatalog, executor: ReadonlySqlExecutor,
                 llm: Chats, *, max_rows: int = 1000, timeout_s: float = 10.0):
        self._catalog = catalog
        self._executor = executor
        self._llm = llm
        self._max_rows = max_rows
        self._timeout_s = timeout_s

    def resolve(self, question: str, *, context: RequestContext,
                state: ConversationState | None = None) -> DomainResolution:
        """判定问题所属数据域；解析不出/歧义/无权限时返回对应 decision。"""
        return self._catalog.resolve(question, context=context)

    def answer(self, question: str, *, domain: str,
               context: RequestContext,
               state: ConversationState | None = None) -> CapabilityResult:
        """在已确认域上执行一次数据查询，返回结构化结果。"""
        metrics.nl2sql_total.labels("started").inc()
        with span("query_data.answer", domain=domain,
                  user_id=context.user_id or None):
            result = self._answer(question, domain=domain, context=context,
                                  state=state)
        metrics.nl2sql_total.labels(result.status).inc()
        return result

    def _answer(self, question: str, *, domain: str,
                context: RequestContext,
                state: ConversationState | None = None) -> CapabilityResult:
        schema = self._catalog.describe(domain, context=context)
        allowed_tables = self._catalog.allowed_tables(domain, context=context)
        if not allowed_tables:
            return CapabilityResult(
                "query_data", "no_tables", "该数据域未配置可查询的表。",
                {"domain": domain})
        domain_max_rows, domain_timeout = self._catalog.domain_limits(
            domain, context=context)
        max_rows = min(self._max_rows, domain_max_rows)
        timeout_s = min(self._timeout_s, domain_timeout)

        previous = ""
        if state is not None and state.extra.get("last_question"):
            previous = f"上轮已确认的问题（仅用于理解追问）：{state.extra['last_question']}\n"

        prompt = (
            "只针对给定的 MySQL 数据域生成单条 SELECT 查询，禁止编写非查询语句。\n"
            "只能使用下方 schema 中出现的表和列；未出现的表/列一律不得使用。\n"
            "条件值请直接使用安全的 SQL 字面量；服务端会执行只读与白名单校验。\n"
            f"数据域：{domain}\n{previous}用户问题：{question}\n{schema}\n"
            "只返回 JSON：{\"sql\": \"...\"}"
        )
        raw = self._llm.chat([ChatMessage("user", prompt)])
        sql = self._parse_sql(raw)
        if not sql:
            return CapabilityResult(
                "query_data", "llm_error", "无法生成有效查询。",
                {"domain": domain})

        try:
            result = self._executor.query(
                sql, domain=domain, context=context,
                timeout_s=timeout_s, max_rows=max_rows)
        except Exception as exc:
            # Guard / 权限 / 执行错误统一转为结果，不向上抛成 500。
            from ..adapters.readonly_executor import SqlGuardError
            from ..domain.exceptions import SqlPolicyError
            if isinstance(exc, (SqlPolicyError, SqlGuardError)):
                return CapabilityResult(
                    "query_data", "guard_rejected", f"查询未通过安全校验：{exc}")
            return CapabilityResult(
                "query_data", "execution_error", f"查询执行失败：{type(exc).__name__}")
        return self._format(domain, result)

    @staticmethod
    def _parse_sql(raw: str) -> str | None:
        """从 LLM 输出里抽取 SQL；优先解析 JSON，兼容裸 SQL 围栏。"""
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(line for line in text.splitlines()
                             if not line.strip().startswith("```"))
        stripped = text.strip().rstrip(";")
        if stripped.lower().startswith("select") or stripped.lower().startswith("with"):
            return stripped
        try:
            data = json.loads(raw.strip().strip("`"))
            sql = data.get("sql") if isinstance(data, dict) else None
            if isinstance(sql, str) and sql.strip():
                return sql.strip().rstrip(";")
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _format(domain: str, result: SqlResult) -> CapabilityResult:
        return CapabilityResult(
            "query_data", "answered", "",
            {
                "domain": domain, "sql": None,
                "query_id": result.query_id,
                "columns": list(result.columns),
                "rows": [list(r) for r in result.rows],
                "row_count": result.row_count,
                "truncated": bool(result.truncated),
            })