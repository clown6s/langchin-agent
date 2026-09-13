"""独立 JWT 的 FastAPI Bearer 依赖。"""
from collections.abc import Callable

from fastapi import HTTPException, Request, status

from ..domain.models import RequestContext
from .jwt import JwtAuthError, verify_token


def jwt_context_provider(*, secret: str, required_permission: str = "chat:ask") -> Callable:
    def provider(request: Request) -> RequestContext:
        header = request.headers.get("Authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="缺少访问令牌")
        try:
            context = verify_token(header[7:].strip(), secret=secret)
        except JwtAuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="令牌无效或已过期") from exc
        if required_permission and required_permission not in context.permissions:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="无权执行该操作")
        return context
    return provider
