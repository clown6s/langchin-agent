"""飞书 HTTP webhook：只做事件协议入口，业务由 FeishuMessageHandler 处理。"""
from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from ..adapters.feishu.handler import FeishuMessageHandler


def create_feishu_router(handler: FeishuMessageHandler, *,
                         verification_token: str = "") -> APIRouter:
    router = APIRouter(prefix="/feishu")

    @router.post("/events")
    async def events(payload: dict):
        # 飞书 URL 验证 challenge，不进入业务链路。
        if payload.get("type") == "url_verification":
            if verification_token and payload.get("token") != verification_token:
                raise HTTPException(status_code=403, detail="验证令牌错误")
            return {"challenge": payload.get("challenge", "")}
        handler.handle(payload)
        return {"ok": True}

    return router
