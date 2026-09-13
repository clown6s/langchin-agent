"""SOP 用例：自然语言 → 已登记流程的查询/推进。

与数据查询同理：SOP 只操作已登记的流程和会话内状态机，不由 LLM 自由编造
流程名、步骤、审批或完成状态。代码负责状态推进与权限校验。
"""
from ..domain.models import CapabilityResult, ConversationState, RequestContext
from ..domain.ports import Chats


class _Step:
    """最简 SOP 步骤描述（生产可替换为 domain.SopDefinition）。"""
    def __init__(self, step_id: str, sequence: int, instruction: str,
                 requires_confirmation: bool = False):
        self.step_id = step_id
        self.sequence = sequence
        self.instruction = instruction
        self.requires_confirmation = requires_confirmation


class _Sop:
    def __init__(self, sop_id: str, title: str, steps: list[_Step]):
        self.sop_id = sop_id
        self.title = title
        self.steps = {s.step_id: s for s in steps}
        self._ordered = sorted(steps, key=lambda s: s.sequence)


class InMemorySopRepository:
    """示例 SOP 仓库；生产替换为 SopStore 端口实现。"""
    def __init__(self, items: list[_Sop] | None = None):
        self._items = {s.sop_id: s for s in (items or [])}

    def register(self, sop: _Sop) -> None:
        self._items[sop.sop_id] = sop

    def match(self, question: str) -> _Sop | None:
        for sop in self._items.values():
            if sop.title and sop.title in question:
                return sop
        return None

    def get(self, sop_id: str) -> _Sop | None:
        return self._items.get(sop_id)


class SopUseCase:
    def __init__(self, repo: InMemorySopRepository, llm: Chats):
        self._repo = repo
        self._llm = llm

    def handle(self, question: str, *, context: RequestContext,
               state: ConversationState | None = None,
               confirmed: bool = False) -> CapabilityResult:
        sop = self._repo.match(question)
        if sop is None:
            return CapabilityResult("guide_sop", "no_sop", "未匹配到已登记的流程。")

        current_id = None
        if state is not None and state.current_sop_id == sop.sop_id:
            current_id = state.current_step_id

        # 读取流程说明 vs 推进状态：无当前步骤则返回起点说明。
        if current_id is None or current_id not in sop.steps:
            first = sop._ordered[0]
            return self._render(sop, first, advancing=False)

        step = sop.steps[current_id]
        if step.requires_confirmation and not confirmed:
            return CapabilityResult(
                "guide_sop", "confirmation", f"即将执行：{step.instruction}",
                self._render(sop, step, advancing=True))
        return self._render(sop, step, advancing=True)

    @staticmethod
    def _render(sop, step: _Step, *, advancing: bool) -> CapabilityResult:
        return CapabilityResult("guide_sop", "running", step.instruction, {
            "sop_id": sop.sop_id,
            "step_id": step.step_id,
            "instruction": step.instruction,
            "advancing": advancing,
        })