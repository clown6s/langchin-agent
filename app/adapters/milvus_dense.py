"""Milvus dense 检索适配器；客户端由组合根注入。"""
from .hybrid_retriever import SearchHit


class MilvusDenseAdapter:
    def __init__(self, client, *, collection: str, vector_field: str = "dense",
                 output_fields=("chunk_id", "document_id", "text")):
        self._client = client
        self._collection = collection
        self._vector_field = vector_field
        self._output_fields = list(output_fields)

    def search(self, vector: list[float], limit: int,
               allowed_document_ids=None) -> list[SearchHit]:
        """按 document_id 白名单过滤；不传白名单则全量检索。"""
        kwargs = {
            "collection_name": self._collection,
            "data": [vector], "anns_field": self._vector_field,
            "limit": limit, "output_fields": list(self._output_fields),
        }
        if allowed_document_ids:
            ids = ", ".join(f'"{doc}"' for doc in allowed_document_ids)
            kwargs["filter"] = f"document_id in [{ids}]"
        rows = self._client.search(**kwargs)
        result = []
        for hit in (rows[0] if rows else []):
            entity = getattr(hit, "entity", None) or getattr(hit, "fields", {}) or {}
            result.append(SearchHit(
                text=entity.get("text", ""),
                document_id=entity.get("document_id", ""),
                chunk_id=entity.get("chunk_id", ""),
                score=float(getattr(hit, "distance", 0)), metadata=entity))
        return result
