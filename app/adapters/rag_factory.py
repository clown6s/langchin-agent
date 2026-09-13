"""独立 RAG 组合根：构造 embedding、Milvus、Elasticsearch 与 hybrid 检索。"""
from .elasticsearch_lexical import ElasticsearchLexicalAdapter
from .embedding import OpenAIEmbeddingAdapter
from .hybrid_retriever import HybridRetriever
from .milvus_dense import MilvusDenseAdapter
from .rag_config import RagSettings


def build_hybrid_retriever(settings: RagSettings | None = None):
    settings = settings or RagSettings()
    from elasticsearch import Elasticsearch
    from pymilvus import MilvusClient

    embedding = OpenAIEmbeddingAdapter(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
    )
    milvus_kwargs = {
        "uri": f"http://{settings.milvus_host}:{settings.milvus_port}",
        "db_name": settings.milvus_db,
    }
    if settings.milvus_token:
        milvus_kwargs["token"] = settings.milvus_token
    elif settings.milvus_user:
        milvus_kwargs["user"] = settings.milvus_user
        milvus_kwargs["password"] = settings.milvus_password
    milvus = MilvusClient(**milvus_kwargs)
    dense = MilvusDenseAdapter(milvus, collection=settings.milvus_collection)

    es_kwargs = {"hosts": [settings.es_host]}
    if settings.es_username:
        es_kwargs["basic_auth"] = (settings.es_username, settings.es_password)
    if settings.es_cert_fingerprint:
        es_kwargs["ssl_assert_fingerprint"] = settings.es_cert_fingerprint
    elastic = Elasticsearch(**es_kwargs)
    lexical = ElasticsearchLexicalAdapter(elastic, index=settings.es_index)

    reranker = None
    rerank_base = settings.rerank_base_url or settings.embedding_base_url
    rerank_key = settings.rerank_api_key or settings.embedding_api_key
    if settings.rerank_model and rerank_base and rerank_key:
        from .reranker import RerankAdapter
        rerank_adapter = RerankAdapter(base_url=rerank_base, api_key=rerank_key,
                                       model=settings.rerank_model)
        reranker = lambda query, texts, top_n: rerank_adapter.rerank(
            query, texts, top_n=top_n)

    return HybridRetriever(
        dense_search=lambda query, limit, allowed=None: dense.search(
            embedding.embed_query(query), limit, allowed),
        lexical_search=lambda query, limit, allowed=None: lexical.search(
            query, limit, allowed),
        rerank=reranker,
        recall_k=settings.recall_k, top_k=min(settings.top_k, settings.rerank_k),
    )
