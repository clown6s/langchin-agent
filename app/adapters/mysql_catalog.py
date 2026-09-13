"""MySQL SchemaCatalog for the single application dataset."""
import pymysql

from ..domain.models import DomainCandidate, DomainResolution, RequestContext
from ..domain.ports import SchemaCatalog


class MysqlSchemaCatalog(SchemaCatalog):
    def __init__(self, *, host: str, port: int, user: str, password: str,
                 database: str, connect_timeout: int = 5):
        self._dsn = dict(host=host, port=port, user=user, password=password,
                         database=database, connect_timeout=connect_timeout,
                         cursorclass=pymysql.cursors.DictCursor,
                         charset="utf8mb4", autocommit=True)

    def _denied_ids(self, user_id: str) -> set[str]:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT domain_id FROM sql_domain_permissions "
                    "WHERE subject_type='user' AND subject_id=%s AND allowed=0",
                    (user_id,))
                return {row["domain_id"] for row in cur.fetchall()}

    def domains(self, *, context: RequestContext) -> list[str]:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                denied = self._denied_ids(context.user_id)
                cur.execute("SELECT domain_id, domain_name FROM sql_data_domains "
                            "WHERE status='active' ORDER BY domain_name")
                return [row["domain_name"] for row in cur.fetchall()
                        if row["domain_id"] not in denied]

    def describe(self, domain: str, *, context: RequestContext) -> str:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                denied = self._denied_ids(context.user_id)
                cur.execute("SELECT domain_id, description FROM sql_data_domains "
                            "WHERE domain_name=%s AND status='active' LIMIT 1", (domain,))
                item = cur.fetchone()
                if item is None or item["domain_id"] in denied:
                    return ""
                cur.execute("SELECT table_name, column_name, business_name, data_type "
                            "FROM sql_allowed_columns WHERE domain_id=%s "
                            "ORDER BY table_name, column_name", (item["domain_id"],))
                lines = [item["description"] or domain]
                current = None
                for row in cur.fetchall():
                    if row["table_name"] != current:
                        current = row["table_name"]
                        lines.append(f"表 {current}: {row.get('business_name') or ''}")
                    lines.append(f"  {row['column_name']} ({row['data_type']})")
                return "\n".join(lines)

    def resolve(self, question: str, *, context: RequestContext) -> DomainResolution:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                denied = self._denied_ids(context.user_id)
                cur.execute("SELECT d.domain_name, d.domain_id, k.keyword, k.weight "
                            "FROM sql_domain_keywords k JOIN sql_data_domains d "
                            "ON d.domain_id=k.domain_id "
                            "WHERE d.status='active'")
                rows = [r for r in cur.fetchall() if r["domain_id"] not in denied]
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for row in rows:
            kw = row["keyword"]
            if kw and kw in question:
                scores[row["domain_name"]] = scores.get(row["domain_name"], 0.0) + float(row["weight"]) * (1.0 + min(len(kw), 8) * .1)
                evidence.setdefault(row["domain_name"], []).append(kw)
        if not scores:
            return DomainResolution(decision="unknown", reason="未命中任何数据域")
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = tuple(DomainCandidate(n, s, tuple(evidence.get(n, ()))) for n, s in ordered)
        top, score = ordered[0]
        if len(ordered) == 1 or ordered[1][1] * 1.3 < score:
            return DomainResolution(decision="resolved", domain=top, candidates=candidates,
                                    margin=score - (ordered[1][1] if len(ordered) > 1 else 0),
                                    confidence=score / (sum(s for _, s in ordered) or 1),
                                    reason="唯一命中数据域")
        return DomainResolution(decision="ambiguous", candidates=candidates,
                                margin=score - ordered[1][1], reason="多域分差过小")

    def allowed_tables(self, domain: str, *, context: RequestContext) -> set[str]:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                denied = self._denied_ids(context.user_id)
                cur.execute("SELECT t.table_name, t.domain_id FROM sql_allowed_tables t "
                            "JOIN sql_data_domains d ON d.domain_id=t.domain_id "
                            "WHERE d.domain_name=%s AND d.status='active'", (domain,))
                return {r["table_name"] for r in cur.fetchall() if r["domain_id"] not in denied}

    def allowed_columns(self, domain: str, *, context: RequestContext) -> dict[str, set[str]]:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT c.table_name, c.column_name FROM sql_allowed_columns c "
                            "JOIN sql_data_domains d ON d.domain_id=c.domain_id "
                            "WHERE d.domain_name=%s AND d.status='active'", (domain,))
                out: dict[str, set[str]] = {}
                for row in cur.fetchall():
                    out.setdefault(row["table_name"], set()).add(row["column_name"])
                return out

    def column_capabilities(self, domain: str, *, context: RequestContext) -> dict:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                denied = self._denied_ids(context.user_id)
                cur.execute("SELECT d.domain_id FROM sql_data_domains d "
                            "WHERE d.domain_name=%s AND d.status='active' LIMIT 1", (domain,))
                item = cur.fetchone()
                if item is None or item["domain_id"] in denied:
                    return {}
                cur.execute("SELECT table_name, column_name, selectable, filterable, sortable, "
                            "aggregatable, `sensitive` FROM sql_allowed_columns WHERE domain_id=%s",
                            (item["domain_id"],))
                out: dict = {}
                for row in cur.fetchall():
                    out.setdefault(row["table_name"], {})[row["column_name"]] = {
                        key: bool(row[key]) for key in ("selectable", "filterable", "sortable", "aggregatable", "sensitive")}
                return out

    def domain_limits(self, domain: str, *, context: RequestContext) -> tuple[int, int]:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT max_rows, max_execution_seconds FROM sql_data_domains "
                            "WHERE domain_name=%s AND status='active' LIMIT 1", (domain,))
                row = cur.fetchone()
                return ((int(row["max_rows"] or 1000), int(row["max_execution_seconds"] or 10))
                        if row else (100000, 60))
