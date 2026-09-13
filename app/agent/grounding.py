"""schema 接地：把已登记数据域/表/列/指标渲染成 LLM 可依据的描述。

判定的权威来自 catalog，而不是模型自由选择。这里产出的描述会被注入到
LangGraph 的提示与 SQL 工具的入参约束（domain 只能从 registered 里选）。
"""
from ..domain.models import ConversationState, RequestContext
from ..domain.ports import SchemaCatalog


class SchemaGrounding:
    def __init__(self, catalog: SchemaCatalog):
        self._catalog = catalog

    def registered_domains(self, *, context: RequestContext) -> list[str]:
        """当前用户可访问的已登记数据域。"""
        try:
            return self._catalog.domains(context=context)
        except Exception:
            return []

    def domain_options_enum(self, *, context: RequestContext) -> str:
        """给 LLM 的 domain 参数选项（强制只能选这些）。"""
        domains = self.registered_domains(context=context)
        if not domains:
            return "（当前没有可查询的数据域）"
        return "、".join(domains)

    def domain_catalog_text(self, *, context: RequestContext,
                            state: ConversationState | None = None) -> str:
        """渲染一份数据域总览，供 LLM 判断“这条问题对得上哪个已登记域”。"""
        domains = self.registered_domains(context=context)
        if not domains:
            return "（无已登记数据域）"
        lines = ["已登记的数据域："]
        for name in domains:
            desc = self._catalog.describe(name, context=context)
            lines.append(f"- {name}: {desc or '（无描述）'}")
        if state is not None and state.last_domain in domains:
            lines.append(f"（上一轮数据域：{state.last_domain}，可作为追问的候选）")
        return "\n".join(lines)

    def validate_domain(self, domain: str, *, context: RequestContext) -> bool:
        """SQL 工具调用前校验：domain 必须属于已登记域。"""
        return domain in self.registered_domains(context=context)