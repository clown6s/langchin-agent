"""飞书 HTTP webhook：只做事件协议入口，业务由 FeishuMessageHandler 处理。"""
from fastapi import APIRouter

from ..adapters.feishu.handler import FeishuMessageHandler


def create_feishu_router(handler: FeishuMessageHandler) -> APIRouter:
    router = APIRouter(prefix="/feishu")

    @router.post("/events")
    async def events(payload: dict):
        # 飞书 URL 验证 challenge，不进入业务链路。
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}
        handler.handle(payload)
        return {"ok": True}

    return router