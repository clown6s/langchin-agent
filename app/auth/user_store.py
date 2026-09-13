"""DB 用户查询：登录时校验账密，复用元数据连接（LC_MYSQL_*）。

不引入认证到 agent 层；本模块只负责"从 auth_users 读用户并验密码"。
roles / permissions 存 JSON 列，读取后解析为元组。
"""
import json

import pymysql

from .password import verify_password


class UserRecord:
    """登录用户的只读视图；字段不可由调用方伪造。"""

    def __init__(self, user_id: str, username: str, roles: tuple[str, ...],
                 permissions: tuple[str, ...], status: str):
        self.user_id = user_id
        self.username = username
        self.roles = roles
        self.permissions = permissions
        self.status = status


def _as_tuple(value) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = [s.strip() for s in value.split(",") if s.strip()]
        if not isinstance(parsed, list):
            return ()
        return tuple(str(x) for x in (parsed or []))
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    return ()


class MysqlUserStore:
    """从 MySQL ``auth_users`` 表校验用户名密码。"""

    def __init__(self, *, host: str, port: int, user: str, password: str,
                 database: str, table: str = "auth_users",
                 connect_timeout: int = 5):
        if not table or not table.replace("_", "").isalnum():
            raise ValueError("非法的用户表名")
        self._table = table
        self._dsn = dict(host=host, port=port, user=user, password=password,
                         database=database, connect_timeout=connect_timeout,
                         cursorclass=pymysql.cursors.DictCursor,
                         charset="utf8mb4", autocommit=True)

    def find_by_username(self, username: str) -> UserRecord | None:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT user_id, username, password_hash, roles, permissions, status "
                    f"FROM {self._table} WHERE username=%s LIMIT 1",
                    (username,))
                row = cur.fetchone()
        if not row:
            return None
        return UserRecord(
            user_id=str(row["user_id"]),
            username=str(row.get("username") or ""),
            roles=_as_tuple(row.get("roles")),
            permissions=_as_tuple(row.get("permissions")),
            status=str(row.get("status") or "active"),
        )

    def verify_credentials(self, username: str, password: str) -> UserRecord | None:
        """账密正确且账号 active 才返回记录；否则返回 None。"""
        record = self.find_by_username(username)
        if record is None or record.status != "active":
            return None
        stored = self._password_hash(username)
        if stored is None or not verify_password(password, stored):
            return None
        return record

    def _password_hash(self, username: str) -> str | None:
        with pymysql.connect(**self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT password_hash FROM {self._table} WHERE username=%s LIMIT 1",
                    (username,))
                row = cur.fetchone()
        return str(row["password_hash"]) if row else None