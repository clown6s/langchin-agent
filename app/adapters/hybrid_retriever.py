"""独立 Hybrid RAG 检索：dense + lexical + RRF。

底层存储通过两个 callable 注入；不依赖 LangChain。
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Callable

from ..domain.models import RequestContext


@dataclass(frozen=True)
class SearchHit:
    text: str
    document_id: str
    chunk_id: str
    score: float
    metadata: dict = None


class HybridRetriever:
    def __init__(self, *, dense_search: Callable, lexical_search: Callable,
                 rerank=None, recall_k: int = 100, top_k: int = 5,
                 rrf_k: int = 60, min_text_len: int = 1):
        self._dense = dense_search
        self._lexical = lexical_search
        self._rerank = rerank
        self._recall_k = max(1, recall_k)
        self._top_k = max(1, top_k)
        self._rrf_k = max(1, rrf_k)
        self._min_text_len = max(0, min_text_len)

    def retrieve(self, query: str, *, top_k: int = 5,
                 context: RequestContext | None = None,
                 allowed_document_ids=None) -> list[tuple[str, str, str, float]]:
        """返回 (text, document_id, chunk_id, score) 四元组列表。"""
        if not query.strip():
            return []
        limit = min(max(1, top_k or self._top_k), self._top_k)
        dense, lexical = self._run_parallel(query, limit, allowed_document_ids)
        merged: dict[str, SearchHit] = {}
        ranks: dict[str, float] = {}
        for rank, hit in enumerate(dense, 1):
            hit = self._coerce(hit)
            ranks[hit.chunk_id] = ranks.get(hit.chunk_id, 0) + 1 / (self._rrf_k + rank)
            merged.setdefault(hit.chunk_id, hit)
        for rank, hit in enumerate(lexical, 1):
            hit = self._coerce(hit)
            ranks[hit.chunk_id] = ranks.get(hit.chunk_id, 0) + 1 / (self._rrf_k + rank)
            merged.setdefault(hit.chunk_id, hit)
        ordered = sorted(merged, key=lambda key: ranks[key], reverse=True)[:limit]
        results = [(merged[key], ranks[key]) for key in ordered]
        # 过滤过短噪声片段；再可选地做 rerank。
        results = [item for item in results if len(item[0].text) >= self._min_text_len]
        if self._rerank is not None and len(results) > 1:
            texts = [hit.text for hit, _ in results]
            try:
                ranked = self._rerank(query, texts, top_n=self._top_k)
                by_index = {i: hit for i, (hit, _) in enumerate(results)}
                return [(by_index.get(i, results[min(i, len(results)-1)][0]).text,
                         by_index.get(i, results[min(i, len(results)-1)][0]).document_id,
                         by_index.get(i, results[min(i, len(results)-1)][0]).chunk_id,
                         score)
                        for i, score in ranked]
            except Exception:
                pass
        return [(hit.text, hit.document_id, hit.chunk_id, score)
                for hit, score in results]

    def _run_parallel(self, query, recall_k, allowed_document_ids=None):
        def dense():
            return self._dense(query, recall_k, allowed_document_ids)
        def lexical():
            return self._lexical(query, recall_k, allowed_document_ids)
        with ThreadPoolExecutor(max_workers=2) as pool:
            d_future = pool.submit(dense)
            l_future = pool.submit(lexical)
            try:
                dense_rows = d_future.result(timeout=30)
            except Exception:
                dense_rows = []
            try:
                lexical_rows = l_future.result(timeout=30)
            except Exception:
                lexical_rows = []
        return dense_rows, lexical_rows

    @staticmethod
    def _coerce(hit) -> SearchHit:
        if isinstance(hit, SearchHit):
            return hit
        if isinstance(hit, dict):
            return SearchHit(text=str(hit.get("text", "")),
                             document_id=str(hit.get("document_id", "")),
                             chunk_id=str(hit.get("chunk_id", "")),
                             score=float(hit.get("score", 0)),
                             metadata=hit)
        raise TypeError("检索结果必须是 SearchHit 或 dict")
