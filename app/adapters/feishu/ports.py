"""飞书渠道端口：即时回复，业务层只依赖端口。"""
from typing import Protocol


class FeishuClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> bool: ...
    def close(self) -> None: ...
