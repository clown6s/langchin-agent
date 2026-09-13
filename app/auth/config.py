"""独立认证配置。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LC_", env_file=".env", extra="ignore")

    jwt_secret: str = ""
    jwt_expires_minutes: int = 60
