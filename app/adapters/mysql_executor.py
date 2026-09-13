"""MySQL 只读执行器：Guard、表白名单和只读连接集中在这里。"""
import uuid

import pymysql

from ..domain.models import RequestContext, SqlResult
from ..domain.ports import SchemaCatalog
from .sql_ast import assert_readonly
from .column_guard import validate_columns
from .readonly_executor import SqlGuardError, validate_sql


class MysqlReadonlyExecutor:
    def __init__(self, *, catalog: SchemaCatalog,
                 host: str, port: int, user: str, password: str,
                 database: str, connect_timeout: int = 5):
        self._catalog = catalog
        self._dsn = dict(host=host, port=port, user=user, password=password,
                         database=database, connect_timeout=connect_timeout,
                         cursorclass=pymysql.cursors.Cursor,
                         charset="utf8mb4", autocommit=False,
                         read_timeout=30, write_timeout=5)

    def query(self, sql: str, *, domain: str, context: RequestContext,
              timeout_s: float = 10.0, max_rows: int = 1000) -> SqlResult:
        # 只读校验（AST）+ 表白名单（词法）。
        clean = assert_readonly(sql)
        allowed_tables = self._catalog.allowed_tables(domain, context=context)
        validate_sql(clean, allowed_tables=allowed_tables)
        capability_loader = getattr(self._catalog, "column_capabilities", None)
        if callable(capability_loader):
            capabilities = capability_loader(domain, context=context)
            if capabilities:
                validate_columns(clean, capabilities)

        final_sql, params = clean, ()

        timeout_ms = max(1, int(timeout_s * 1000))
        config = dict(self._dsn, read_timeout=timeout_s, write_timeout=timeout_s)
        with pymysql.connect(**config) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SET SESSION TRANSACTION READ ONLY")
                    cur.execute("SET SESSION MAX_EXECUTION_TIME = %s", (timeout_ms,))
                    if params:
                        cur.execute(final_sql, params)
                    else:
                        cur.execute(final_sql)
                    description = cur.description or ()
                    columns = tuple(item[0] for item in description)
                    fetched = tuple(tuple(row) for row in cur.fetchmany(max_rows + 1))
                    truncated = len(fetched) > max_rows
                    rows = fetched[:max_rows]
            finally:
                conn.rollback()  # 只读：任何情况下不提交。
        return SqlResult(
            columns=columns,
            rows=rows,
            query_id=str(uuid.uuid4()),
            row_count=len(rows),
            truncated=truncated,
        )

    def close(self) -> None:
        pass