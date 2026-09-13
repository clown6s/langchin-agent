"""通用知识用例：法规/技术/常识/闲聊 的兜底作答。

企业 RAG 无召回时，本用例直接以 LLM 作答，避免把非数据问题塞进业务域澄清。
只负责“普通回答”这条能力，不参与是否查库的判定。
"""
from ..domain.models import CapabilityResult, ChatMessage, RequestContext
from ..domain.ports import Chats, KnowledgeRetriever


class GeneralUseCase:
    def __init__(self, llm: Chats, retriever: KnowledgeRetriever | None = None,
                 *, top_k: int = 5):
        self._llm = llm
        self._retriever = retriever
        self._top_k = top_k

    def answer(self, question: str, *, context: RequestContext) -> CapabilityResult:
        # 装配了知识库：企业知识是唯一依据，无命中不得用模型常识补答。
        if self._retriever is not None:
            try:
                hits = self._retriever.retrieve(
                    question, top_k=self._top_k, context=context)
            except Exception:
                hits = []
            if not hits:
                return CapabilityResult(
                    "answer", "no_context", "企业知识库未收录相关内容，无法回答。")
            return self._rag_answer(question, hits, context)
        # 未装配知识库：才允许通用 LLM 直接作答。
        prompt = (
            "请直接回答用户的通用知识或常识问题。这是法规/技术/百科或闲聊类问题，"
            "不涉及企业业务数据。请用简洁、准确的中文作答。\n"
            f"用户问题：{question}"
        )
        raw = self._llm.chat([ChatMessage("user", prompt)])
        return CapabilityResult("answer", "answered", raw)

    def _rag_answer(self, question: str, hits, context: RequestContext) -> CapabilityResult:
        context_text = "\n".join(
            f"[{i}] {text}" for i, (text, _, _) in enumerate(hits, 1))
        prompt = (
            "只能依据下面的参考片段回答，禁止使用任何外部知识或模型常识。\n"
            "请围绕用户问题给出完整、清晰、有层次的解释：先给结论，再说明定义、原因、作用、"
            "处理流程或关键注意事项；片段没有支持的内容不要补充或猜测。\n"
            "片段未覆盖的内容一律回答『企业知识库未收录相关内容，无法回答。』\n"
            "不要只复述一个术语或一句话；但也不要为了凑长度扩展片段未支持的事实。\n"
            "只返回 JSON：{\"answer\":\"...\",\"used\":[实际引用到的编号]}。\n"
            f"用户问题：{question}\n{context_text}"
        )
        raw = self._llm.chat([ChatMessage("user", prompt)])
        data = None
        text = raw.strip()
        if "{" in text:
            import json
            try:
                data = json.loads(text.strip("`"))
                text = data.get("answer") or raw
            except (ValueError, TypeError):
                data = None
        # 硬规则：必须引用至少一个片段，否则视为知识库未收录，不允许凭模型知识作答。
        used = data.get("used") if isinstance(data, dict) else None
        if not isinstance(used, list) or not used:
            return CapabilityResult(
                "answer", "no_context", "企业知识库未收录相关内容，无法回答。")
        return CapabilityResult("answer", "answered", text)