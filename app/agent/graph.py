"""LangGraph 编排：LLM 按已登记 schema 决定是否调用工具，未匹配回退普通回答。

这就是"能力制"的核心：SQL/SOP/RAG 是 LLM 可选的工具；问题对不上任何能力时，
LLM 直接产出文本答案（法规/技术/常识/追问），不再需要意图分类器。
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ..observability import metrics
from .tools import build_tools


def _merge(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict, total=False):
    condensed_question: str
    messages: Annotated[list, _merge]
    used_tools: Annotated[list, _merge]
    tool_results: Annotated[list, _merge]
    iterations: Annotated[int, operator.add]
    final_answer: str
    system_prompt: str


class LangGraphAgent:
    """基于 LangGraph 的工具决策 Agent。"""

    def __init__(self, *, llm, grounding_text: str = "",
                 grounding=None, query_uc, sop_uc, general_uc,
                 max_iterations: int = 6):
        self._llm = llm
        self._grounding_text = grounding_text
        self._grounding = grounding  # SchemaGrounding；有则按请求上下文动态生成
        self._query_uc = query_uc
        self._sop_uc = sop_uc
        self._general_uc = general_uc
        self._max_iterations = max_iterations
        self._context = None
        self._state = None

    def _tools(self):
        from ..domain.models import RequestContext
        ctx = self._context or RequestContext()
        return build_tools(
            query_uc=self._query_uc, sop_uc=self._sop_uc,
            general_uc=self._general_uc, context=ctx, state=self._state)

    def _graph(self):
        g = StateGraph(AgentState)

        def decide(state: AgentState):
            if state.get("iterations", 0) >= self._max_iterations:
                return {"messages": [AIMessage(content="无法在安全限制内完成查询。")],
                        "iterations": 1}
            llm = self._llm.bind_tools([t for t in self._tools()])
            resp = metrics.instrument_llm(llm, state["messages"], kind="agent")
            return {"messages": [resp], "iterations": 1}

        def run_tool(state: AgentState):
            last = state["messages"][-1]
            tool_map = {t.name: t for t in self._tools()}
            out_messages = []
            used = list(state.get("used_tools", []))
            results = list(state.get("tool_results", []))
            for call in (last.tool_calls or []):
                name = call.get("name")
                tool_fn = tool_map.get(name)
                try:
                    if tool_fn is None:
                        out = f"未知工具：{name}"
                    else:
                        out = tool_fn.invoke(call.get("args", {}))
                        used.append(name)
                    metrics.tool_calls_total.labels(name or "unknown", "success").inc()
                except Exception:
                    metrics.tool_calls_total.labels(name or "unknown", "error").inc()
                    raise
                out_messages.append(ToolMessage(content=str(out),
                                                tool_call_id=call.get("id", "")))
                # 供编排层持久化业务状态用（工具只读，写状态由 coordinator 负责）。
                results.append({"name": name, "text": str(out)})
            return {"messages": out_messages, "used_tools": used,
                    "tool_results": results}

        def route(state: AgentState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            # 普通回答必须先经过企业知识检索；检索过或已有业务工具结果后才结束。
            if (not state.get("used_tools")
                    and getattr(self._general_uc, "_retriever", None) is not None):
                return "retrieve"
            return "finish"

        def retrieve(state: AgentState):
            tool_map = {t.name: t for t in self._tools()}
            tool_fn = tool_map["retrieve_enterprise"]
            question = state.get("condensed_question", "")
            general = self._general_uc
            if getattr(general, "_llm", None) is None:
                out = "当前未配置知识回答模型。"
            else:
                out = tool_fn.invoke({"question": question})
            call = {"name": "retrieve_enterprise", "args": {"question": question},
                    "id": "auto_retrieve"}
            return {
                "messages": [
                    AIMessage(content="", tool_calls=[call]),
                    ToolMessage(content=str(out), tool_call_id="auto_retrieve"),
                ],
                "used_tools": ["retrieve_enterprise"],
                "tool_results": [{"name": "retrieve_enterprise", "text": str(out)}],
            }

        def finish(state: AgentState):
            last = state["messages"][-1]
            if isinstance(last, ToolMessage):
                text = last.content or ""
            elif isinstance(last, AIMessage):
                text = getattr(last, "content", "") or ""
            else:
                text = str(last)
            return {"final_answer": text or "（无回答）"}

        g.add_node("decide", decide)
        g.add_node("tools", run_tool)
        g.add_node("retrieve", retrieve)
        g.add_node("finish", finish)
        g.set_entry_point("decide")
        g.add_conditional_edges("decide", route,
                                {"tools": "tools", "retrieve": "retrieve", "finish": "finish"})
        g.add_edge("tools", "decide")
        g.add_edge("retrieve", "finish")
        g.add_edge("finish", END)
        return g

    def chat(self, condensed_question: str) -> dict:
        """运行一次 Agent，返回 {answer, used_tools}。"""
        system = SystemMessage(self._system_prompt())
        human = HumanMessage(f"用户问题：\n{condensed_question}")
        app = self._graph().compile()
        result = app.invoke({
            "condensed_question": condensed_question,
            "messages": [system, human],
            "used_tools": [],
            "tool_results": [],
            "iterations": 0,
        })
        answer = (result.get("final_answer") or "").strip()
        if not answer:
            answer = self._general_uc.answer(
                condensed_question, context=self._context).text
        # 合并工具结果：按名称取最后一次调用。
        tool_results: dict[str, dict] = {}
        for item in result.get("tool_results", []):
            tool_results[item.get("name")] = item
        return {"answer": answer, "used_tools": result.get("used_tools", []),
                "tool_results": tool_results}

    def _system_prompt(self) -> str:
        grounding_text = self._grounding_text
        if self._grounding is not None and self._context is not None:
            grounding_text = self._grounding.domain_catalog_text(
                context=self._context, state=self._state)
        return (
            "你是企业数据助手。你可用三类工具：\n"
            "1. query_data：查询企业业务数据（必须落在已登记数据域）。\n"
            "2. guide_sop：企业流程（仅已登记流程）。\n"
            "3. retrieve_enterprise：检索企业知识库。\n"
            "判断原则：\n"
            "- 用户在问业务数据且对得上已登记域 → 调 query_data。\n"
            "- 用户在咨询/办理已登记流程 → 调 guide_sop。\n"
            "- 问题基于企业文档/制度 → 调 retrieve_enterprise。\n"
            "- 法规、技术、常识、闲聊、或对不上任何已登记能力 → 不调用工具，直接给出简洁准确的回答。\n"
            "可参考的已登记域/流程信息：\n" + grounding_text
        )

    def bind_context(self, context, state=None) -> "LangGraphAgent":
        """把一次请求的上下文绑定到 agent（配合组合根使用）。"""
        self._context = context
        self._state = state or None
        return self