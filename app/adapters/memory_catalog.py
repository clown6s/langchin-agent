"""已登记 schema catalog 的轻量内存实现（可切换 MySQL 版）。

这里是判定的权威来源：只有注册进 catalog 的域/表/列/指标才会被暴露给 LLM。
新数据主题通过 register 登记即可支持，不需要改路由代码。
"""
from dataclasses import dataclass, field

from ..domain.models import (DomainCandidate, DomainResolution, RequestContext)
from ..domain.ports import SchemaCatalog


@dataclass(frozen=True)
class RegisteredDomain:
    name: str
    description: str
    tables: frozenset[str] = frozenset()
    columns: dict[str, frozenset[str]] = field(default_factory=dict)
    keywords: tuple[str, ...] = ()


class MemorySchemaCatalog(SchemaCatalog):
    def __init__(self, domains: list[RegisteredDomain] | None = None):
        self._domains = {d.name: d for d in (domains or [])}
        self._denied: set[str] = set()

    # -- 管理接口（登记新数据主题） ------------------------------------------
    def register(self, domain: RegisteredDomain) -> None:
        self._domains[domain.name] = domain

    def deny(self, domain: str) -> None:
        self._denied.add(domain)

    # -- SchemaCatalog 端口实现 ---------------------------------------------
    def domains(self, *, context: RequestContext) -> list[str]:
        return [n for n in sorted(self._domains) if n not in self._denied]

    def describe(self, domain: str, *, context: RequestContext) -> str:
        d = self._domains.get(domain)
        if d is None or domain in self._denied:
            return ""
        lines = [d.description]
        for table, cols in d.columns.items():
            lines.append(f"表 {table}: " + ", ".join(sorted(cols)))
        return "\n".join(lines)

    def resolve(self, question: str, *, context: RequestContext) -> DomainResolution:
        """关键词匹配打分，返回唯一域；歧义返回 ambiguous，无匹配返回 unknown。"""
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for name, d in self._domains.items():
            if name in self._denied:
                continue
            hits = [kw for kw in d.keywords if kw and kw in question]
            if hits:
                scores[name] = sum(1.0 + min(len(k), 8) * 0.1 for k in hits)
                evidence[name] = hits
        if not scores:
            return DomainResolution(decision="unknown", reason="未命中任何已登记数据域")
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top, top_score = ordered[0]
        candidates = tuple(DomainCandidate(n, s, tuple(evidence.get(n, ())))
                           for n, s in ordered)
        if len(ordered) == 1 or ordered[1][1] * 1.3 < top_score:
            return DomainResolution(decision="resolved", domain=top,
                                    candidates=candidates, margin=top_score - ordered[1][1] if len(ordered) > 1 else 0,
                                    confidence=top_score / (sum(s for _, s in ordered) or 1),
                                    reason="唯一命中数据域")
        return DomainResolution(decision="ambiguous", candidates=candidates,
                                margin=top_score - ordered[1][1],
                                reason="多个数据域分差过小")

    def allowed_tables(self, domain: str, *, context: RequestContext) -> set[str]:
        d = self._domains.get(domain)
        return set(d.tables) if d and domain not in self._denied else set()

    def allowed_columns(self, domain: str, *, context: RequestContext) -> dict[str, set[str]]:
        d = self._domains.get(domain)
        if d is None or domain in self._denied:
            return {}
        return {t: set(c) for t, c in d.columns.items()}

    def domain_limits(self, domain: str, *, context: RequestContext) -> tuple[int, int]:
        return 1000, 10

    def column_capabilities(self, domain: str, *, context: RequestContext) -> dict[str, dict[str, dict]]:
        """内存实现：默认所有列都可选可过滤，敏感列由 RegisteredDomain.sensitive 标记。"""
        d = self._domains.get(domain)
        if d is None or domain in self._denied:
            return {}
        sensitive = getattr(d, "sensitive_columns", frozenset())
        out: dict[str, dict] = {}
        for table, cols in d.columns.items():
            out[table] = {
                name: {"selectable": True, "filterable": True, "sortable": True,
                       "aggregatable": True, "sensitive": name in sensitive}
                for name in cols
            }
        return out