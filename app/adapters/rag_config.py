"""独立 RAG 配置。"""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LC_", env_file=".env", extra="ignore")

    # Embedding
    embedding_base_url: str = ''
    embedding_api_key: str = ''
    embedding_model: str = ''
    embedding_dim: int = 4096

    # Rerank
    rerank_base_url: str = ''
    rerank_api_key: str = ''
    rerank_model: str = ''

    # Milvus
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_token: str = ''
    milvus_user: str = ''
    milvus_password: str = ''
    milvus_db: str = ''
    milvus_collection: str = ''

    # Elasticsearch
    es_host: str = ''
    es_username: str = ''
    es_password: str = ''
    es_cert_fingerprint: str = ''
    es_index: str = ''

    recall_k: int = Field(
        default=100, validation_alias=AliasChoices("LC_RECALL_K", "RECALL_K"))
    rerank_k: int = Field(
        default=100, validation_alias=AliasChoices("LC_RERANK_K", "RERANK_K"))
    top_k: int = Field(
        default=5, validation_alias=AliasChoices("LC_TOP_K", "TOP_K"))
