-- ============================================================================
-- langchin-agent 登录用户表（放在元数据库，与 sql_data_domains 同库）
-- 建表：
--   在 LC_MYSQL_DATABASE 指向的库执行本文件。
--
-- 密码哈希格式见 password.py：pbkdf2_sha256$<iterations>$<salt_hex>$<dk_hex>
-- 生成首个账号的哈希：
--   python db/gen_password_hash.py abc123
-- 把输出的哈希串填入下方 INSERT 后执行即可。
-- ============================================================================

CREATE TABLE IF NOT EXISTS auth_users (
  user_id       VARCHAR(64)  NOT NULL COMMENT '业务用户标识（写入 JWT sub / 权限表的 subject_id）',
  username      VARCHAR(128) NOT NULL COMMENT '登录名，全局唯一',
  password_hash VARCHAR(255) NOT NULL COMMENT 'PBKDF2 哈希串',
  roles         JSON         NOT NULL COMMENT '角色数组，如 ["admin"]',
  permissions   JSON         NOT NULL COMMENT '权限数组，如 ["chat:ask"]',
  status        VARCHAR(16)  NOT NULL DEFAULT 'active' COMMENT 'active 可登录；其他值禁用',
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id),
  UNIQUE KEY uq_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='认证用户表';

-- ---- 首个管理员账号（把 <FILL_哈希> 替换成 gen_password_hash.py 的输出）----
INSERT INTO auth_users (user_id, username, password_hash, roles, permissions, status)
VALUES ('admin', 'admin', 'pbkdf2_sha256$100000$e23b1fedfc3745efdb4f2170dd85bbb6$d9b09267b744bef0a1cc25d8177adb284e0a705f668e596969be139d7622c46f', '["admin"]', '["chat:ask"]', 'active');