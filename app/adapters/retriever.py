"""把旧系统的 hybrid 检索器适配成新项目的 KnowledgeRetriever 端口。"""
from ..domain.models import RequestContext
from ..domain.ports import KnowledgeRetriever


class LegacyRetrieverAdapter(KnowledgeRetriever):
    """包住旧 RetrieveDocuments.execute_hybrid。

    新项目不实现向量检索；复用旧系统的 Milvus dense + ES 全文 + RRF + 重排。
    只做返回结构转换，不引入旧系统类型到新业务层。
    """

    def __init__(self, retrieve_documents, *, top_k: int = 5,
                 filter_fn=None):
        self._inner = retrieve_documents
        self._top_k = top_k
        self._filter_fn = filter_fn

    def retrieve(self, query: str, *, top_k: int = 5,
                 context: RequestContext) -> list[tuple[str, str, str, float]]:
        k = top_k or self._top_k
        # 过滤表达式由部署方提供（旧检索器 filter 语法因存储而异）。
        filter_expr = ""
        if self._filter_fn is not None:
            filter_expr = self._filter_fn(context) or ""
        results = self._inner.execute_hybrid(query, top_k=k, filter=filter_expr)
        return [
            (item.text, item.document_id,
             getattr(item, "chunk_id", "") or item.document_id,
             float(item.score))
            for item in results
        ]

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
