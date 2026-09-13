"""组合根：按配置装配所有依赖，返回 LangGraphAgent 及其依赖。

这里只做“装配”，不含业务逻辑。可通过替换某个 adapter 接入 MySQL/真实检索。
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LC_", env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    max_iterations: int = 6

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = ""

    # 业务查询连接与元数据连接分离；查询账号只能读取已授权业务表。
    query_mysql_host: str = ""
    query_mysql_port: int = 0
    query_mysql_user: str = ""
    query_mysql_password: str = ""
    query_mysql_database: str = ""


_default_domains = None  # 由外部受管 catalog 装配数据域。


def build_agent(settings: Settings | None = None, *, retriever=None,
                enable_rag: bool = False) -> dict:
    """装配并返回运行时对象集合。"""
    from .adapters.llm import ChatsWrapper, build_chat_openai
    from .adapters.memory_catalog import MemorySchemaCatalog
    from .adapters.mysql_catalog import MysqlSchemaCatalog
    from .adapters.mysql_executor import MysqlReadonlyExecutor
    from .adapters.readonly_executor import GuardedReadonlyExecutor
    from .agent.graph import LangGraphAgent
    from .agent.grounding import SchemaGrounding
    from .application.data_query import DataQueryUseCase
    from .application.general import GeneralUseCase
    from .application.sop import InMemorySopRepository, SopUseCase, _Sop, _Step

    settings = settings or Settings()
    llm = build_chat_openai(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key,
        model=settings.llm_model, temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens)
    chats = ChatsWrapper(llm)

    catalog, executor = _build_storage(settings)
    if enable_rag and retriever is None:
        from .adapters.rag_factory import build_hybrid_retriever
        retriever = build_hybrid_retriever()
    grounding = SchemaGrounding(catalog)
    query_uc = DataQueryUseCase(catalog, executor, chats)
    sop_uc = SopUseCase(InMemorySopRepository([
        _Sop("expense_claim", "报销流程", [_Step("s1", 1, "提交发票与单据"),
                                          _Step("s2", 2, "部门审批"),
                                          _Step("s3", 3, "财务打款", requires_confirmation=True)]),
    ]), chats)
    general_uc = GeneralUseCase(chats, retriever=(retriever if retriever is not None else None))

    agent = LangGraphAgent(
        llm=llm, grounding=grounding,
        query_uc=query_uc, sop_uc=sop_uc, general_uc=general_uc,
        max_iterations=settings.max_iterations,
    )
    return {"agent": agent, "catalog": catalog,
            "use_cases": {"query": query_uc, "sop": sop_uc, "general": general_uc}}


def _build_storage(settings: Settings):
    """统一走 MySQL：元数据连接读取已登记 schema，只读连接执行业务查询。"""
    from .adapters.mysql_catalog import MysqlSchemaCatalog
    from .adapters.mysql_executor import MysqlReadonlyExecutor
    catalog = MysqlSchemaCatalog(
        host=settings.mysql_host, port=settings.mysql_port,
        user=settings.mysql_user, password=settings.mysql_password,
        database=settings.mysql_database)
    # 查询执行器用独立只读账号；缺省时回退到元数据账号以兼容旧配置。
    q_host = settings.query_mysql_host or settings.mysql_host
    q_port = settings.query_mysql_port or settings.mysql_port
    q_user = settings.query_mysql_user or settings.mysql_user
    q_password = settings.query_mysql_password or settings.mysql_password
    q_db = settings.query_mysql_database or settings.mysql_database
    return catalog, MysqlReadonlyExecutor(
        catalog=catalog, host=q_host, port=q_port,
        user=q_user, password=q_password, database=q_db)


def _anonymous_context():
    from .domain.models import RequestContext
    return RequestContext()
