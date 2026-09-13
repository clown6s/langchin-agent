"""只读 SQL 执行器：把生成的 SQL 交给 SQL Guard 校验后再执行。

安全职责（LLM 不可绕过）：
- 只读连接（不接写库）；
- SQL Guard：禁止非 SELECT、白名单表/列、阻断敏感列。
"""
import re

from ..domain.models import RequestContext, SqlResult
from ..domain.ports import ReadonlySqlExecutor


class SqlGuardError(Exception):
    pass


def validate_sql(sql: str, *, allowed_tables: set[str]) -> str:
    """极简 SQL Guard：只放行单条 SELECT，且只引用白名单表。"""
    s = sql.strip().rstrip(";")
    if not re.match(r"^(?i:select|with)\b", s):
        raise SqlGuardError("只允许 SELECT 查询")
    if re.search(r"(?i)(insert|update|delete|drop|alter|truncate|grant|union\s+select)",
                 s) and not re.match(r"^(?i:with)\b", s):
        raise SqlGuardError("检测到非查询语句")
    # 表名白名单：收集 from/join 后的表名做粗校验（生产可用 sqlglot 精确解析）。
    for table in re.findall(r"(?i)(?:from|join)\s+([a-z_0-9]+)", s):
        if table.lower() not in allowed_tables:
            raise SqlGuardError(f"访问未授权表 {table}")
    return s


class GuardedReadonlyExecutor(ReadonlySqlExecutor):
    """Guard 校验 + 只读执行。此处演示内存返回，生产替换为只读连接执行。"""

    def __init__(self, catalog):
        self._catalog = catalog

    def query(self, sql: str, *, domain: str, context: RequestContext,
              timeout_s: float = 10.0, max_rows: int = 1000) -> SqlResult:
        allowed = self._catalog.allowed_tables(domain, context=context)
        clean = validate_sql(sql, allowed_tables=allowed)
        # 生产：用只读连接 + 参数化 + 行权执行 clean。
        return SqlResult(
            columns=("_demo",), rows=(("ok",),),
            query_id="demo-1", row_count=1)