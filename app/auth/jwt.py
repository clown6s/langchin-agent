"""独立 JWT 认证：签发与验证只依赖本项目配置。"""
from datetime import datetime, timedelta, timezone

import jwt

from ..domain.models import RequestContext


class JwtAuthError(Exception):
    """JWT 缺失、签名错误、过期或声明无效。"""


def issue_token(*, secret: str, user_id: str,
                permissions: tuple[str, ...] = (), roles: tuple[str, ...] = (),
                expires_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "permissions": list(permissions),
        "roles": list(roles),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, *, secret: str) -> RequestContext:
    if not secret:
        raise JwtAuthError("未配置 JWT 密钥")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise JwtAuthError("令牌无效或已过期") from exc
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise JwtAuthError("令牌缺少用户")
    permissions = payload.get("permissions", ())
    roles = payload.get("roles", ())
    if not isinstance(permissions, list) or not all(isinstance(x, str) for x in permissions):
        raise JwtAuthError("令牌权限声明无效")
    if not isinstance(roles, list) or not all(isinstance(x, str) for x in roles):
        raise JwtAuthError("令牌角色声明无效")
    return RequestContext(
        user_id=user_id,
        roles=tuple(roles), permissions=frozenset(permissions),
        authenticated=True,
    )
