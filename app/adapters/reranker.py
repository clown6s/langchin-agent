"""OpenAI 兼容 Rerank 适配器。"""
import httpx


class RerankAdapter:
    def __init__(self, *, base_url: str, api_key: str, model: str,
                 timeout: float = 60):
        self._url = base_url.rstrip("/") + "/rerank"
        self._key = api_key
        self._model = model
        self._timeout = timeout

    def rerank(self, query: str, texts: list[str], *, top_n: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        response = httpx.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "query": query,
                  "documents": texts, "top_n": min(top_n, len(texts)),
                  "return_documents": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        ranked = [(int(item["index"]), float(item["relevance_score"]))
                  for item in results]
        return sorted(ranked, key=lambda item: item[1], reverse=True)
