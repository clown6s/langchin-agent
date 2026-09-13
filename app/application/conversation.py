"""会话关系用例：判定追问/换题/混合，并做指代改写。

这里不判断问题该走哪条能力；只产出“自包含的问题”和“关系类型”，
供 agent 层做增域的上下文提示与继承依据。
"""
from dataclasses import dataclass

from ..domain.models import ChatMessage, ConversationState, RequestContext


@dataclass(frozen=True)
class ResolvedTurn:
    question: str            # 自包含改写后的问题
    relation: str            # new_topic | follow_up | mixed | clarification
    inheritable: bool        # 是否可继承上一轮数据域/流程
    is_question_like: bool   # 是否像一句完整提问
    needs_clarification: bool


class ConversationUseCase:
    """最简规则版会话理解；生产可扩展为更完备的改写+LLM 消解。"""

    def understand(self, question: str, *, context: RequestContext,
                   state: ConversationState | None = None) -> ResolvedTurn:
        q = (question or "").strip()
        if not q:
            return ResolvedTurn("", "clarification", False, False, True)

        topic = self._last_topic(state)

        # 修正/否定追问：不是换话题，而是在原主题上修正。
        if self._is_correction(q):
            return ResolvedTurn(f"{topic} {q}", "follow_up", True, True, False)

        # 纯续问词：继承主题。
        if q in {"那呢", "然后呢", "下一步呢", "接下来呢", "还有呢", "然后呢"}:
            return ResolvedTurn(f"{topic} {q}", "follow_up", True, True, False)

        # 连接词/指代词开头：通常是追问，改写为主题 + 当前句。
        if self._is_follow_up_start(q):
            return ResolvedTurn(f"{topic} {q}", "follow_up", True, True, False)

        # 代词指代上一轮对象（它/这个/那个/上一条/刚才说的）。
        if self._is_anaphora(q) and topic:
            return ResolvedTurn(f"{topic} {q}", "follow_up", True, True, False)

        if self._looks_like_complete_question(q):
            return ResolvedTurn(q, "new_topic",
                                state is None or state.last_intent is None,
                                True, False)
        return ResolvedTurn(q, "new_topic", False, self._looks_like_complete_question(q), False)

    @staticmethod
    def _is_correction(q: str) -> bool:
        return q.startswith(("不对", "不是", "改成", "换作", "换成", "改为", "纠正"))

    @staticmethod
    def _is_follow_up_start(q: str) -> bool:
        return q.startswith(("其中", "只看", "还有", "跟", "和", "然后"))

    @staticmethod
    def _is_anaphora(q: str) -> bool:
        return bool(
            q.startswith(("它", "这个", "那个", "这些", "那些", "上一条", "上一条的",
                          "刚才说的", "刚才", "前面", "前面说的", "上述", "上面说的",
                          "它的", "这个的", "那个的"))
            and len(q) >= 2)

    @staticmethod
    def _last_topic(state: ConversationState | None) -> str:
        if state is None or not state.extra:
            return ""
        return state.extra.get("topic") or state.extra.get("last_question") or ""

    @staticmethod
    def _looks_like_complete_question(q: str) -> bool:
        # 有主语/对象且不以代词/连接词开头，视为独立完整问题。
        if q.startswith(("它", "上述", "刚才", "其中", "和", "跟", "那", "然后")):
            return False
        return len(q) >= 4 and not q[0].isdigit()


def history_messages(state: ConversationState | None) -> list[ChatMessage]:
    """从会话状态导出历史消息（供 agent/tools 做上下文提示）。"""
    if state is None:
        return []
    return list(state.history)