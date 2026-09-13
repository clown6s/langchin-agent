"""生成 auth_users 的 PBKDF2 密码哈希（仅标准库，无需安装依赖）。

用法：
    python db/gen_password_hash.py <你的密码>
输出形如：pbkdf2_sha256$100000$<salt_hex>$<dk_hex>
把输出串填入 db/auth_schema.sql 的 INSERT 中 password_hash 处。
"""
import sys
from pathlib import Path

# 把项目根加入 sys.path，再走 app.auth.password 的标准库实现
# （该模块只依赖 hashlib/hmac/secrets，不会触发未安装的第三方库）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.auth.password import hash_password  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1]:
        print("用法：python db/gen_password_hash.py <密码>", file=sys.stderr)
        return 2
    print(hash_password(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())