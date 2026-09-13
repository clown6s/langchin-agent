"""生产部署入口：新 Agent 使用本项目独立 JWT 认证。"""
from ..auth.config import AuthSettings
from ..auth.dependencies import jwt_context_provider
from ..auth.user_store import MysqlUserStore
from ..bootstrap import Settings
from .auth import create_auth_router
from .main import make_agent_app


settings = AuthSettings()
app = make_agent_app(
    context_provider=jwt_context_provider(
        secret=settings.jwt_secret, required_permission="chat:ask"),
    enable_rag=True)

# 登录接口：复用元数据连接校验 auth_users 里的账密后签发 JWT。
_db = Settings()
app.include_router(create_auth_router(
    user_store=MysqlUserStore(
        host=_db.mysql_host, port=_db.mysql_port,
        user=_db.mysql_user, password=_db.mysql_password,
        database=_db.mysql_database),
    secret=settings.jwt_secret,
    expires_minutes=settings.jwt_expires_minutes))
