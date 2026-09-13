"""生产部署入口：新 Agent 使用本项目独立 JWT 认证。"""
from ..auth.config import AuthSettings
from ..auth.dependencies import jwt_context_provider
from .main import make_agent_app


settings = AuthSettings()
app = make_agent_app(
    context_provider=jwt_context_provider(
        secret=settings.jwt_secret, required_permission="chat:ask"),
    enable_rag=True)
