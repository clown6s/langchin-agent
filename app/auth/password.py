"""密码哈希：Python 标准库 PBKDF2-HMAC-SHA256，零第三方依赖。

存量串格式：``pbkdf2_sha256$<iterations>$<salt_hex>$<dk_hex>``。
只依赖 hashlib/hmac/secrets，保证在没有项目依赖的环境也能用于生成种子哈希。
"""
import hashlib
import hmac
import secrets

_ITERATIONS = 100_000


def hash_password(password: str, iterations: int = _ITERATIONS) -> str:
    """生成带随机盐的 PBKDF2 哈希串。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验明文密码是否匹配存量哈希串；格式不合法一律 False。"""
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
            int(iterations))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, TypeError):
        return False
