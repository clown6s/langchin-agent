"""生产部署入口。"""
from .main import make_agent_app

app = make_agent_app(enable_rag=True)