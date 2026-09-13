"""认证路由：POST /auth/login（公开，无需令牌）。"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..auth.jwt import issue_token


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str


def create_auth_router(*, user_store, secret: str,
                       expires_minutes: int = 60) -> APIRouter:
    """登录路由；user_store 需提供 verify_credentials(username, password)。"""
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/login", response_model=LoginResponse)
    async def login(req: LoginRequest):
        record = user_store.verify_credentials(req.username, req.password)
        if record is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="用户名或密码错误")
        token = issue_token(
            secret=secret, user_id=record.user_id,
            permissions=record.permissions, roles=record.roles,
            expires_minutes=expires_minutes)
        return LoginResponse(
            access_token=token,
            expires_in=expires_minutes * 60,
            user_id=record.user_id)

    return router