"""LLM 适配：ChatOpenAI 经 OpenAI 兼容端点（SiliconFlow/DeepSeek）。

- LangGraphAgent 直接使用返回的 ChatOpenAI（具备工具调用能力）。
- DataQueryUseCase / GeneralUseCase 需要 Chats 端口，这里提供 ChatsWrapper 桥接。
- LazyChatOpenAI 延迟真实模型构造到首次调用，避免 import 即需密钥。
- LLM 调用统一打点指标与 span（低基数 label）。
"""
from langchain_openai import ChatOpenAI

from ..domain.models import ChatMessage
from ..domain.ports import Chats
from ..observability import metrics


class LazyChatOpenAI:
    """延迟到首次使用才构造 ChatOpenAI 的惰性包装。

    目的是：组合根 import 时不触发密钥校验/网络，只有真正发起请求才构造真实客户端。
    """

    def __init__(self, *, base_url: str, api_key: str, model: str,
                 temperature: float = 0.2, max_tokens: int = 1024):
        self._config = {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._real: ChatOpenAI | None = None

    def _get(self) -> ChatOpenAI:
        if self._real is None:
            self._real = ChatOpenAI(**self._config)
        return self._real

    def bind_tools(self, tools):
        return self._get().bind_tools(tools)

    def invoke(self, messages):
        # LLM 调用计指标在调用点（ChatsWrapper/graph）统一处理，避免重复打点。
        return self._get().invoke(messages)

    def __getattr__(self, name):
        # 其余属性委托给真实模型（如 ainvoke 等）。
        return getattr(self._get(), name)


def build_chat_openai(*, base_url: str, api_key: str, model: str,
                      temperature: float = 0.2, max_tokens: int = 1024):
    """返回惰性 LLM，import/装配阶段零副作用。"""
    return LazyChatOpenAI(
        base_url=base_url, api_key=api_key, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


class ChatsWrapper(Chats):
    """把 langchain 模型包成 domain.Chats 端口，供 use case 使用。"""

    def __init__(self, llm):
        self._llm = llm

    def chat(self, messages: list[ChatMessage]) -> str:
        langchain_msgs = [
            {"role": m.role, "content": m.content} for m in messages]
        resp = metrics.instrument_llm(self._llm, langchain_msgs, kind="chat")
        return getattr(resp, "content", "") or ""

    def close(self) -> None:
        pass