"""飞书渠道端口：身份解析与回复，业务层只依赖端口。"""
from typing import Protocol

from ...domain.models import RequestContext


class IdentityResolver(Protocol):
    """把飞书 open_id 映射为可信身份；未绑定返回 None。"""
    def resolve(self, *, app_id: str, open_id: str) -> RequestContext | None: ...


class FeishuClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> bool: ...
    def close(self) -> None: ...
