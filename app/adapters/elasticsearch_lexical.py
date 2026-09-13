"""Elasticsearch lexical 检索适配器；客户端由组合根注入。"""
from .hybrid_retriever import SearchHit


class ElasticsearchLexicalAdapter:
    def __init__(self, client, *, index: str):
        self._client = client
        self._index = index

    def search(self, query: str, limit: int,
               allowed_document_ids=None) -> list[SearchHit]:
        """按 document_id 白名单过滤；不传白名单则全量检索。"""
        filters = []
        if allowed_document_ids:
            filters.append({"terms": {"document_id": list(allowed_document_ids)}})
        response = self._client.search(
            index=self._index, size=limit,
            query={"bool": {
                "must": [{"multi_match": {
                    "query": query, "fields": ["text", "search_text", "title"]
                }}],
                "filter": filters,
            }},
        )
        hits = response.get("hits", {}).get("hits", [])
        return [SearchHit(
            text=hit.get("_source", {}).get("text", ""),
            document_id=hit.get("_source", {}).get("document_id", ""),
            chunk_id=hit.get("_source", {}).get("chunk_id", hit.get("_id", "")),
            score=float(hit.get("_score") or 0),
            metadata=hit.get("_source", {}),
        ) for hit in hits]
