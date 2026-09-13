"""OpenAI 兼容 Embedding 适配器。"""
import httpx


class OpenAIEmbeddingAdapter:
    def __init__(self, *, base_url: str, api_key: str, model: str,
                 dimension: int, timeout: float = 30):
        self._url = base_url.rstrip("/") + "/embeddings"
        self._key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout = timeout

    def embed_query(self, text: str) -> list[float]:
        response = httpx.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "input": text, "encoding_format": "float"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data or not isinstance(data[0].get("embedding"), list):
            raise ValueError("Embedding 响应无效")
        vector = data[0]["embedding"]
        if len(vector) != self._dimension:
            raise ValueError(f"Embedding 维度错误: {len(vector)} != {self._dimension}")
        return vector
