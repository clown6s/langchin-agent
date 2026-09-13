"""认证端口适配。

新项目不实现 token 校验；组合根注入旧系统或其他认证系统的 verifier。
"""
from collections.abc import Callable

from ..domain.models import RequestContext


class AuthenticationError(Exception):
    """访问令牌缺失、失效或已撤销。"""


class PermissionError(Exception):
    """已认证主体缺少所需权限。"""


def context_from_principal(principal) -> RequestContext:
    """把已验证 Principal 转成不可伪造的业务上下文。"""
    return RequestContext(
        user_id=principal.user_id,
        roles=tuple(getattr(principal, "roles", ()) or ()),
        authenticated=True,
        permissions=frozenset(getattr(principal, "permissions", ()) or ()),
    )


def principal_provider_from_verifier(verifier: Callable[[str], object],
                                     *, required_permission: str = "chat:ask"):
    """返回 FastAPI 依赖：验证 Bearer token 后建立 RequestContext。"""
    from fastapi import HTTPException, Request, status

    def provider(request: Request) -> RequestContext:
        header = request.headers.get("Authorization", "")
        if not header.lower().startswith("bearer ") or not header[7:].strip():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="缺少访问令牌")
        try:
            principal = verifier(header[7:].strip())
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="令牌无效或已撤销") from exc
        can = getattr(principal, "can", None)
        if required_permission and (not callable(can) or not can(required_permission)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="无权执行该操作")
        return context_from_principal(principal)

    return provider
