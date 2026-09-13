"""LangChain 工具工厂：把用例封装成 LLM 可调用的能力（按请求上下文绑定）。

工具只做三件事：
- 参数受 schema 接地约束（SQL 工具的 domain 限定已登记域，实际校验在 application 层）；
- 真正的执行与安全校验在 use case 内部完成（Guard / 只读 / 白名单）；
- 非业务问题（法规/技术/闲聊）不调用业务工具。

闭包捕获的上下文是每个请求独立的，避免并发串号，也让 LangGraph 提示缓存不因
频繁变化的上下文失效。
"""
from langchain_core.tools import BaseTool, tool

from ..application.data_query import DataQueryUseCase
from ..application.general import GeneralUseCase
from ..application.sop import SopUseCase
from ..domain.models import ConversationState, RequestContext


def build_tools(*, query_uc: DataQueryUseCase,
                sop_uc: SopUseCase,
                general_uc: GeneralUseCase,
                context: RequestContext,
                state: ConversationState | None = None) -> list[BaseTool]:
    """按请求上下文生成一组工具；每个请求调用一次。"""

    @tool
    def query_data(question: str, domain: str | None = None) -> str:
        """查询业务数据。给定问题的完整自然语言与目标数据域，返回只读查询结果。

        仅当用户确实在询问业务数据、且问题能落到某个已登记数据域时才调用；
        法规/技术/百科/常识等非业务问题**不要**调用本工具。
        """
        if not domain:
            resolution = query_uc.resolve(question, context=context, state=state)
            if resolution.decision != "resolved" or not resolution.domain:
                cand = "、".join(c.name for c in resolution.candidates) or "无候选"
                return (f"数据域无法唯一确定（decision={resolution.decision}，候选={cand}）。"
                        "需要用户指定一个具体数据域。")
            domain = resolution.domain
        result = query_uc.answer(question, domain=domain, context=context, state=state)
        if result.status == "answered":
            p = result.payload
            cols = p.get("columns") or []
            rows = p.get("rows") or []
            return (f"查询执行成功，共 {p.get('row_count', len(rows))} 行。\n"
                    f"列：{','.join(cols)}\n行：{rows[:20]}")
        if result.status == "no_tables":
            return "数据域已确认，但未配置可查询的表，无法执行。"
        return f"查询失败（{result.status}）：{result.text}"

    @tool
    def guide_sop(question: str) -> str:
        """企业流程（SOP）：查询或推进已登记的流程步骤。

        仅当用户明确在咨询/办理企业内部的流程、审批或步骤，且系统登记了对应流程时调用；
        如何做某功能、如何实现某技术（无登记流程）等不调用本工具。
        """
        result = sop_uc.handle(question, context=context, state=state)
        import json
        envelope = {
            "capability": result.capability,
            "status": result.status,
            "text": result.text,
            "payload": result.payload,
        }
        return json.dumps(envelope, ensure_ascii=False)

    @tool
    def retrieve_enterprise(question: str) -> str:
        """检索企业知识库，回答基于内部文档/制度/规范的问题。"""
        result = general_uc.answer(question, context=context)
        return result.text

    return [query_data, guide_sop, retrieve_enterprise]