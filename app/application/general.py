"""通用知识用例：法规/技术/常识/闲聊 的兜底作答。

企业 RAG 无召回时，本用例直接以 LLM 作答，避免把非数据问题塞进业务域澄清。
只负责“普通回答”这条能力，不参与是否查库的判定。
"""
from ..domain.models import CapabilityResult, ChatMessage, RequestContext
from ..domain.ports import Chats, KnowledgeRetriever
import structlog

logger = structlog.get_logger(__name__)


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
                logger.info("rag_answer_no_hits", question=question,
                            sent_chunk_ids=[])
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
        # hits: (text, document_id, chunk_id, score)
        sent_chunk_ids = [cid for (_t, _d, cid, _s) in hits]
        context_text = "\n".join(
            f"[{i}] {text}" for i, (text, _, _, _) in enumerate(hits, 1))
        prompt = self._rag_prompt(question, context_text)
        raw = self._llm.chat([ChatMessage("user", prompt)])
        data = self._parse_json(raw)
        text = (data.get("answer") if isinstance(data, dict) and data.get("answer")
                else raw.strip())
        used = self._extract_used(data, raw)

        # 兜底：used 未命中时重试一次并强调必须引用；仍为空则按未收录处理。
        if not used:
            retry = self._rag_prompt(
                question, context_text,
                extra="务必把实际引用到的片段编号填入 used，只要答案基于片段就至少引用一个，不能为空。")
            raw2 = self._llm.chat([ChatMessage("user", retry)])
            data2 = self._parse_json(raw2)
            text2 = (data2.get("answer") if isinstance(data2, dict) and data2.get("answer")
                     else raw2.strip())
            used2 = self._extract_used(data2, raw2)
            used = used2
            text = text2 if used2 else text
            if not used2 and not (text and text.strip()):
                logger.info("rag_answer_no_context", question=question,
                            sent_chunk_ids=sent_chunk_ids, used_chunk_ids=[])
                return CapabilityResult(
                    "answer", "no_context", "企业知识库未收录相关内容，无法回答。")
        self._log_rag_used(question, sent_chunk_ids, used)
        return CapabilityResult("answer", "answered", text)

    @staticmethod
    def _log_rag_used(question: str, sent_chunk_ids: list, used) -> None:
        used_ids = []
        if isinstance(used, list):
            used_ids = [sent_chunk_ids[i - 1] for i in used
                        if 1 <= i <= len(sent_chunk_ids)]
        logger.info("rag_answer_used", question=question,
                    sent_chunk_ids=sent_chunk_ids,
                    used_indices=used if isinstance(used, list) else None,
                    used_chunk_ids=used_ids)

    @staticmethod
    def _rag_prompt(question: str, context_text: str, *, extra: str = "") -> str:
        intro = (
            "只能依据下面的参考片段回答，禁止使用任何外部知识或模型常识。\n"
            "请围绕用户问题给出完整、清晰、有层次的解释：先给结论，再说明定义、原因、作用、"
            "处理流程或关键注意事项；片段没有支持的内容不要补充或猜测。\n"
            "片段未覆盖的内容一律回答『企业知识库未收录相关内容，无法回答。』\n"
            "不要只复述一个术语或一句话；但也不要为了凑长度扩展片段未支持的事实。\n"
            "只返回 JSON：{\"answer\":\"...\",\"used\":[实际引用到的编号]}。\n")
        if extra:
            intro += extra + "\n"
        return f"{intro}用户问题：{question}\n{context_text}"

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        text = raw.strip()
        if "{" in text:
            import json
            try:
                data = json.loads(text.strip("`"))
                return data if isinstance(data, dict) else None
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _extract_used(data, raw) -> list | None:
        """取 used：优先选 JSON 的 used 键；否则从答案文本里提取 [N] 内联引用。"""
        if isinstance(data, dict):
            used = data.get("used")
            if isinstance(used, list) and used:
                return [int(x) for x in used if str(x).isdigit()]
        text = raw.strip()
        if isinstance(data, dict) and data.get("answer"):
            text = str(data["answer"])
        # 兼容正文里的 [1]、[2] 引用标记。
        import re
        inline = [int(m) for m in re.findall(r"\[(\d{1,2})\]", text)]
        return inline if inline else None