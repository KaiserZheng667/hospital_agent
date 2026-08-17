"""Local authentication, authorization, and audit storage.

The browser never chooses an actor identity.  A random opaque session token is
stored in an HttpOnly cookie, while only its SHA-256 digest is stored here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_AUTH_DB = Path(".medical-agent-data/auth.sqlite")
SESSION_TTL = timedelta(hours=8)
LOGIN_WINDOW = timedelta(minutes=15)
MAX_LOGIN_FAILURES = 5
PASSWORD_N = 2**14
PASSWORD_R = 8
PASSWORD_P = 1


@dataclass(frozen=True)
class Principal:
    """Trusted identity reconstructed by the server from a login session."""

    user_id: str
    username: str
    display_name: str
    department: str
    roles: frozenset[str]
    permissions: frozenset[str]
    csrf_token: str

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Hash a password with the memory-hard scrypt password KDF."""

    if len(password) < 12:
        raise ValueError("密码至少需要 12 个字符")
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=PASSWORD_N, r=PASSWORD_R, p=PASSWORD_P
    )
    return f"scrypt${PASSWORD_N}${PASSWORD_R}${PASSWORD_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, expected_hex = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected_hex))
    except (ValueError, TypeError):
        return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthRepository:
    """SQLite implementation of the authentication and audit boundary."""

    def __init__(self, database: Path = DEFAULT_AUTH_DB) -> None:
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS permissions (
                    permission_code TEXT PRIMARY KEY,
                    description TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_permissions (
                    user_id TEXT NOT NULL,
                    permission_code TEXT NOT NULL,
                    PRIMARY KEY (user_id, permission_code),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (permission_code) REFERENCES permissions(permission_code)
                );

                CREATE TABLE IF NOT EXISTS roles (
                    role_code TEXT PRIMARY KEY,
                    name TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_roles (
                    user_id TEXT NOT NULL,
                    role_code TEXT NOT NULL,
                    PRIMARY KEY (user_id, role_code),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (role_code) REFERENCES roles(role_code)
                );

                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_code TEXT NOT NULL,
                    permission_code TEXT NOT NULL,
                    PRIMARY KEY (role_code, permission_code),
                    FOREIGN KEY (role_code) REFERENCES roles(role_code),
                    FOREIGN KEY (permission_code) REFERENCES permissions(permission_code)
                );

                CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    csrf_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    actor_user_id TEXT,
                    event_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    target_type TEXT,
                    target_id TEXT,
                    request_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_login_sessions_user
                ON login_sessions(user_id, expires_at);

                CREATE INDEX IF NOT EXISTS idx_audit_actor_time
                ON audit_events(actor_user_id, occurred_at DESC);
                """
            )
            permissions = (
                ("patient:read", "查看被授权患者的资料"),
                ("agent:use", "使用医疗 Agent"),
                ("user:read", "查看账号"),
                ("user:create", "创建账号"),
                ("user:disable", "启用或停用账号"),
                ("role:manage", "分配账号角色"),
                ("patient_access:manage", "分配医生患者范围"),
                ("audit:read", "查看审计事件"),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO permissions(permission_code, description) VALUES (?, ?)",
                permissions,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO roles(role_code, name) VALUES (?, ?)",
                (("doctor", "医生"), ("administrator", "系统管理员")),
            )
            role_permissions = (
                ("doctor", "patient:read"),
                ("doctor", "agent:use"),
                ("administrator", "user:read"),
                ("administrator", "user:create"),
                ("administrator", "user:disable"),
                ("administrator", "role:manage"),
                ("administrator", "patient_access:manage"),
                ("administrator", "audit:read"),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO role_permissions(role_code, permission_code)
                VALUES (?, ?)
                """,
                role_permissions,
            )

        initial_password = os.getenv("MEDICAL_AGENT_INITIAL_PASSWORD", "DemoOnly-2026!")
        self._seed_user(
            "doctor-chen", "chen", "陈医生", "骨质疏松研究组", initial_password, "doctor"
        )
        self._seed_user(
            "doctor-lin", "lin", "林医生", "骨质疏松研究组", initial_password, "doctor"
        )
        admin_password = os.getenv(
            "MEDICAL_AGENT_INITIAL_ADMIN_PASSWORD", "AdminOnly-2026!"
        )
        self._seed_user(
            "admin-local", "admin", "本地管理员", "系统管理", admin_password, "administrator"
        )

    def _seed_user(
        self,
        user_id: str,
        username: str,
        display_name: str,
        department: str,
        password: str,
        role: str,
    ) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not exists:
                connection.execute(
                    """
                    INSERT INTO users(
                        user_id, username, display_name, department, password_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, username, display_name, department, hash_password(password), _now()),
                )
            has_role = connection.execute(
                "SELECT 1 FROM user_roles WHERE user_id = ? LIMIT 1", (user_id,)
            ).fetchone()
            if not has_role:
                connection.execute(
                    "INSERT INTO user_roles(user_id, role_code) VALUES (?, ?)",
                    (user_id, role),
                )

    def authenticate(self, username: str, password: str) -> tuple[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, password_hash FROM users
                WHERE username = ? COLLATE NOCASE AND is_active = 1
                """,
                (username.strip(),),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        session_token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO login_sessions(
                    token_hash, user_id, csrf_hash, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _digest(session_token),
                    row["user_id"],
                    _digest(csrf_token),
                    now.isoformat(),
                    (now + SESSION_TTL).isoformat(),
                ),
            )
        return session_token, csrf_token

    def login_is_rate_limited(self, username: str) -> bool:
        """Limit repeated failures for one normalized account identifier."""

        since = (datetime.now(UTC) - LOGIN_WINDOW).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS failure_count FROM audit_events
                WHERE event_type = 'login' AND outcome = 'failure'
                  AND lower(target_id) = lower(?) AND occurred_at >= ?
                """,
                (username.strip(), since),
            ).fetchone()
        return bool(row and row["failure_count"] >= MAX_LOGIN_FAILURES)

    def get_principal(self, session_token: str | None, csrf_token: str = "") -> Principal | None:
        if not session_token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT u.user_id, u.username, u.display_name, u.department,
                       s.csrf_hash, s.expires_at, s.revoked_at
                FROM login_sessions AS s
                JOIN users AS u ON u.user_id = s.user_id
                WHERE s.token_hash = ? AND u.is_active = 1
                """,
                (_digest(session_token),),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
                return None
            permissions = connection.execute(
                """
                SELECT DISTINCT rp.permission_code
                FROM user_roles AS ur
                JOIN role_permissions AS rp ON rp.role_code = ur.role_code
                WHERE ur.user_id = ?
                """,
                (row["user_id"],),
            ).fetchall()
            roles = connection.execute(
                "SELECT role_code FROM user_roles WHERE user_id = ? ORDER BY role_code",
                (row["user_id"],),
            ).fetchall()
        return Principal(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            department=row["department"],
            roles=frozenset(item["role_code"] for item in roles),
            permissions=frozenset(item["permission_code"] for item in permissions),
            csrf_token=csrf_token,
        )

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT u.user_id, u.username, u.display_name, u.department,
                       u.is_active, u.created_at,
                       COALESCE(group_concat(ur.role_code, ','), '') AS roles_csv
                FROM users AS u
                LEFT JOIN user_roles AS ur ON ur.user_id = u.user_id
                GROUP BY u.user_id
                ORDER BY u.created_at, u.username
                """
            ).fetchall()
        users: list[dict[str, Any]] = []
        for row in rows:
            user = dict(row)
            roles_csv = user.pop("roles_csv")
            user["is_active"] = bool(user["is_active"])
            user["roles"] = sorted(filter(None, roles_csv.split(",")))
            users.append(user)
        return users

    def create_user(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
        department: str,
        password: str,
        roles: list[str],
    ) -> None:
        self._validate_roles(roles)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users(
                    user_id, username, display_name, department, password_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, username.strip(), display_name.strip(), department.strip(),
                    hash_password(password), _now(),
                ),
            )
            connection.executemany(
                "INSERT INTO user_roles(user_id, role_code) VALUES (?, ?)",
                ((user_id, role) for role in sorted(set(roles))),
            )

    def set_user_active(self, user_id: str, is_active: bool) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET is_active = ? WHERE user_id = ?",
                (int(is_active), user_id),
            )
            if cursor.rowcount and not is_active:
                connection.execute(
                    """
                    UPDATE login_sessions SET revoked_at = ?
                    WHERE user_id = ? AND revoked_at IS NULL
                    """,
                    (_now(), user_id),
                )
        return bool(cursor.rowcount)

    def set_user_roles(self, user_id: str, roles: list[str]) -> bool:
        self._validate_roles(roles)
        requested_roles = set(roles)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not exists:
                return False
            current_roles = {
                row["role_code"]
                for row in connection.execute(
                    "SELECT role_code FROM user_roles WHERE user_id = ?", (user_id,)
                ).fetchall()
            }
            if current_roles == requested_roles:
                return True
            connection.execute("DELETE FROM user_roles WHERE user_id = ?", (user_id,))
            connection.executemany(
                "INSERT INTO user_roles(user_id, role_code) VALUES (?, ?)",
                ((user_id, role) for role in sorted(requested_roles)),
            )
            connection.execute(
                """
                UPDATE login_sessions SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (_now(), user_id),
            )
        return True

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, occurred_at, actor_user_id, event_type, outcome,
                       target_type, target_id, request_id, details_json
                FROM audit_events ORDER BY event_id DESC LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def _validate_roles(self, roles: list[str]) -> None:
        if not roles:
            raise ValueError("账号至少需要一个角色")
        with self._connect() as connection:
            known = {
                row["role_code"]
                for row in connection.execute("SELECT role_code FROM roles").fetchall()
            }
        unknown = set(roles) - known
        if unknown:
            raise ValueError(f"未知角色：{', '.join(sorted(unknown))}")

    def csrf_is_valid(self, session_token: str | None, csrf_token: str | None) -> bool:
        if not session_token or not csrf_token:
            return False
        with self._connect() as connection:
            row = connection.execute(
                "SELECT csrf_hash FROM login_sessions WHERE token_hash = ? AND revoked_at IS NULL",
                (_digest(session_token),),
            ).fetchone()
        return row is not None and hmac.compare_digest(row["csrf_hash"], _digest(csrf_token))

    def revoke_session(self, session_token: str | None) -> None:
        if not session_token:
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE login_sessions SET revoked_at = ? WHERE token_hash = ?",
                (_now(), _digest(session_token)),
            )

    def audit(
        self,
        event_type: str,
        outcome: str,
        *,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(
                    occurred_at, actor_user_id, event_type, outcome,
                    target_type, target_id, request_id, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(), actor_user_id, event_type, outcome,
                    target_type, target_id, request_id,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )


def _now() -> str:
    return datetime.now(UTC).isoformat()
