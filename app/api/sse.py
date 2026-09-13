"""SSE 事件编码小助手。"""
import json


def sse(obj: dict) -> str:
    """把一个 dict 编码成 SSE 的 data 行。"""
    body = json.dumps(obj, ensure_ascii=False, default=str)
    return f"data: {body}\n\n"