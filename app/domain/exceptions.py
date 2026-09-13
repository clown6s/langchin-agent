"""领域异常。"""
class AppError(Exception):
    """应用层基类异常。"""


class SqlPolicyError(AppError):
    """SQL 安全策略拒绝（Guard / 行级权限）。"""


class ValidationError(AppError):
    """数据校验失败。"""
