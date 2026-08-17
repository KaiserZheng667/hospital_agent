"""Local web application for the medical AI agent teaching project."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field

from medical_agent.auth import AuthRepository, Principal
from medical_agent.graph import build_graph
from medical_agent.repository import get_medical_repository
from medical_agent.session import checkpoint_thread_id

DATA_DIR = Path(".medical-agent-data")
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"
SESSION_DB = DATA_DIR / "sessions.sqlite"
AUTH_DB = DATA_DIR / "auth.sqlite"
STATIC_DIR = Path(__file__).with_name("static")
SESSION_COOKIE = "medical_agent_session"
CSRF_COOKIE = "medical_agent_csrf"
auth_repository = AuthRepository(AUTH_DB)
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message: str = Field(min_length=1, max_length=2000)


class AdminUserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    display_name: str = Field(min_length=1, max_length=80)
    department: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=256)
    roles: list[str] = Field(min_length=1, max_length=2)


class AdminUserStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_active: bool


class AdminUserRoles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: list[str] = Field(min_length=1, max_length=2)


class AdminPatientAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_ids: list[str] = Field(max_length=1000)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect_sessions() -> sqlite3.Connection:
    connection = sqlite3.connect(SESSION_DB)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect_sessions() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_sessions (
                actor_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (actor_id, thread_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_sessions_actor_updated
            ON ui_sessions(actor_id, updated_at DESC)
            """
        )


def _current_principal(request: Request) -> Principal:
    principal = auth_repository.get_principal(
        request.cookies.get(SESSION_COOKIE), request.cookies.get(CSRF_COOKIE, "")
    )
    if principal is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return principal


def _require_csrf(
    request: Request,
    x_csrf_token: str | None = Header(default=None),
) -> None:
    if not auth_repository.csrf_is_valid(
        request.cookies.get(SESSION_COOKIE), x_csrf_token
    ):
        raise HTTPException(status_code=403, detail="请求安全校验失败，请刷新页面后重试")


def _require_permission(principal: Principal, permission: str) -> None:
    if principal.has_permission(permission):
        return
    auth_repository.audit(
        "authorization", "denied", actor_user_id=principal.user_id,
        target_type="permission", target_id=permission,
    )
    raise HTTPException(status_code=403, detail="当前账号没有执行该操作的权限")


CurrentPrincipal = Annotated[Principal, Depends(_current_principal)]


def _require_session(actor_id: str, thread_id: str) -> None:
    with _connect_sessions() as connection:
        row = connection.execute(
            "SELECT 1 FROM ui_sessions WHERE actor_id = ? AND thread_id = ?",
            (actor_id, thread_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")


def _serialize_message(message: Any) -> dict[str, Any] | None:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": str(message.content)}
    if isinstance(message, ToolMessage):
        return None
    if isinstance(message, AIMessage) and not message.tool_calls and message.content:
        return {"role": "assistant", "content": str(message.content)}
    return None


def _load_messages(actor_id: str, thread_id: str) -> list[dict[str, Any]]:
    storage_id = checkpoint_thread_id(actor_id, thread_id)
    config = {"configurable": {"thread_id": storage_id}}
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(config)
    if not snapshot.values:
        return []
    return [
        serialized
        for message in snapshot.values.get("messages", [])
        if (serialized := _serialize_message(message)) is not None
    ]


def _run_agent(
    request: ChatRequest, actor_id: str, request_id: str
) -> tuple[str, list[dict[str, str]]]:
    storage_id = checkpoint_thread_id(actor_id, request.thread_id)
    trace: list[dict[str, str]] = []
    final_answer = "Agent 未生成回答。"

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        for update in graph.stream(
            {
                "messages": [HumanMessage(content=request.message)],
                "request_id": request_id,
                "actor_id": actor_id,
            },
            config={
                "configurable": {"thread_id": storage_id},
                "recursion_limit": 10,
            },
            stream_mode="updates",
        ):
            node_name, node_update = next(iter(update.items()))
            # Routing/no-op nodes can produce ``None`` in update streams.
            # They change graph control flow but do not contain a user-visible message.
            if not node_update:
                continue
            if node_name == "query_plan":
                plan_trace = {
                    "node": node_name,
                    "detail": (
                        f"意图：{node_update['intent']}；"
                        f"范围：{node_update.get('query_scope', 'none')}；"
                        f"患者：{node_update.get('patient_id', '沿用或未指定')}；"
                        "指标：" + ", ".join(node_update.get("requested_indicator_codes", []))
                    ),
                }
                if node_update.get("patient_id"):
                    plan_trace["patient_id"] = node_update["patient_id"]
                trace.append(plan_trace)
                continue
            if node_name == "authorize_patient":
                decision = "允许访问" if node_update["authorization_allowed"] else "拒绝访问"
                trace.append({"node": node_name, "detail": decision})
                continue
            if node_name == "choose_result_source":
                if node_update["use_cached_result"]:
                    source = "复述已保存结果"
                elif node_update["force_tool_call"]:
                    source = "强制刷新数据源"
                else:
                    source = "交由 Agent 决策"
                trace.append({"node": node_name, "detail": source})
                continue

            messages = node_update.get("messages")
            if not messages:
                continue
            message = messages[-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                call = message.tool_calls[0]
                trace.append({"node": node_name, "detail": f"调用 {call['name']}"})
            elif isinstance(message, ToolMessage):
                trace.append({"node": node_name, "detail": "数据源返回成功"})
            elif isinstance(message, AIMessage):
                final_answer = str(message.content)
                trace.append({"node": node_name, "detail": "生成安全回答"})

    return final_answer, trace


load_dotenv()
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
_initialize_storage()
auth_repository.initialize()

app = FastAPI(title="医疗 AI Agent 工作台", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_index(principal: CurrentPrincipal) -> FileResponse:
    _require_permission(principal, "user:read")
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/bootstrap")
def bootstrap(principal: CurrentPrincipal) -> dict[str, Any]:
    return {
        "user": {
            "id": principal.user_id,
            "username": principal.username,
            "name": principal.display_name,
            "department": principal.department,
            "roles": sorted(principal.roles),
            "permissions": sorted(principal.permissions),
        },
        "csrf_token": principal.csrf_token,
        "qwen_available": bool(os.getenv("DASHSCOPE_API_KEY")),
    }


@app.post("/api/auth/login")
def login(credentials: LoginRequest, response: Response) -> dict[str, Any]:
    if auth_repository.login_is_rate_limited(credentials.username):
        auth_repository.audit(
            "login", "rate_limited", target_type="username",
            target_id=credentials.username.strip(),
        )
        raise HTTPException(status_code=429, detail="登录失败次数过多，请稍后再试")
    authenticated = auth_repository.authenticate(credentials.username, credentials.password)
    if authenticated is None:
        auth_repository.audit(
            "login", "failure", target_type="username", target_id=credentials.username.strip()
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    session_token, csrf_token = authenticated
    principal = auth_repository.get_principal(session_token, csrf_token)
    if principal is None:  # defensive: the new session must always resolve
        raise HTTPException(status_code=500, detail="无法创建登录会话")
    secure = os.getenv("MEDICAL_AGENT_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        SESSION_COOKIE, session_token, httponly=True, secure=secure,
        samesite="strict", max_age=8 * 60 * 60, path="/",
    )
    response.set_cookie(
        CSRF_COOKIE, csrf_token, httponly=False, secure=secure,
        samesite="strict", max_age=8 * 60 * 60, path="/",
    )
    auth_repository.audit("login", "success", actor_user_id=principal.user_id)
    return {"authenticated": True, "csrf_token": csrf_token}


@app.post("/api/auth/logout", dependencies=[Depends(_require_csrf)])
def logout(request: Request, response: Response) -> dict[str, bool]:
    session_token = request.cookies.get(SESSION_COOKIE)
    principal = auth_repository.get_principal(session_token)
    auth_repository.revoke_session(session_token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    auth_repository.audit(
        "logout", "success", actor_user_id=principal.user_id if principal else None
    )
    return {"authenticated": False}


@app.get("/api/admin/users")
def admin_list_users(principal: CurrentPrincipal) -> dict[str, Any]:
    _require_permission(principal, "user:read")
    return {"users": auth_repository.list_users()}


@app.post("/api/admin/users", dependencies=[Depends(_require_csrf)])
def admin_create_user(request: AdminUserCreate, principal: CurrentPrincipal) -> dict[str, str]:
    _require_permission(principal, "user:create")
    _require_permission(principal, "role:manage")
    user_id = f"user-{uuid4().hex[:12]}"
    try:
        auth_repository.create_user(
            user_id=user_id,
            username=request.username,
            display_name=request.display_name,
            department=request.department,
            password=request.password,
            roles=request.roles,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    auth_repository.audit(
        "user_created", "success", actor_user_id=principal.user_id,
        target_type="user", target_id=user_id, details={"roles": sorted(request.roles)},
    )
    return {"user_id": user_id}


@app.patch("/api/admin/users/{user_id}/status", dependencies=[Depends(_require_csrf)])
def admin_set_user_status(
    user_id: str, request: AdminUserStatus, principal: CurrentPrincipal
) -> dict[str, bool]:
    _require_permission(principal, "user:disable")
    if user_id == principal.user_id and not request.is_active:
        raise HTTPException(status_code=400, detail="管理员不能停用自己的当前账号")
    if not auth_repository.set_user_active(user_id, request.is_active):
        raise HTTPException(status_code=404, detail="账号不存在")
    auth_repository.audit(
        "user_status_changed", "success", actor_user_id=principal.user_id,
        target_type="user", target_id=user_id, details={"is_active": request.is_active},
    )
    return {"is_active": request.is_active}


@app.put("/api/admin/users/{user_id}/roles", dependencies=[Depends(_require_csrf)])
def admin_set_user_roles(
    user_id: str, request: AdminUserRoles, principal: CurrentPrincipal
) -> dict[str, list[str]]:
    _require_permission(principal, "role:manage")
    if user_id == principal.user_id and "administrator" not in request.roles:
        raise HTTPException(status_code=400, detail="管理员不能移除自己的管理员角色")
    try:
        updated = auth_repository.set_user_roles(user_id, request.roles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="账号不存在")
    auth_repository.audit(
        "user_roles_changed", "success", actor_user_id=principal.user_id,
        target_type="user", target_id=user_id, details={"roles": sorted(request.roles)},
    )
    return {"roles": sorted(set(request.roles))}


@app.get("/api/admin/patients")
def admin_list_patients(principal: CurrentPrincipal) -> dict[str, Any]:
    _require_permission(principal, "patient_access:manage")
    return {"patients": get_medical_repository().list_patients()}


@app.get("/api/admin/users/{user_id}/patient-access")
def admin_get_patient_access(user_id: str, principal: CurrentPrincipal) -> dict[str, Any]:
    _require_permission(principal, "patient_access:manage")
    patients = get_medical_repository().list_authorized_patients(user_id)
    return {"patient_ids": [patient["patient_id"] for patient in patients]}


@app.put(
    "/api/admin/users/{user_id}/patient-access",
    dependencies=[Depends(_require_csrf)],
)
def admin_set_patient_access(
    user_id: str, request: AdminPatientAccess, principal: CurrentPrincipal
) -> dict[str, list[str]]:
    _require_permission(principal, "patient_access:manage")
    known_users = {user["user_id"] for user in auth_repository.list_users()}
    if user_id not in known_users:
        raise HTTPException(status_code=404, detail="账号不存在")
    try:
        get_medical_repository().replace_patient_access(user_id, request.patient_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    normalized = sorted({patient_id.strip().upper() for patient_id in request.patient_ids})
    auth_repository.audit(
        "patient_access_changed", "success", actor_user_id=principal.user_id,
        target_type="user", target_id=user_id, details={"patient_ids": normalized},
    )
    return {"patient_ids": normalized}


@app.get("/api/admin/audit")
def admin_list_audit(
    principal: CurrentPrincipal, limit: int = 100
) -> dict[str, Any]:
    _require_permission(principal, "audit:read")
    return {"events": auth_repository.list_audit_events(limit)}


@app.get("/api/patients")
def authorized_patients(principal: CurrentPrincipal) -> dict[str, Any]:
    _require_permission(principal, "patient:read")
    return {"patients": get_medical_repository().list_authorized_patients(principal.user_id)}


@app.get("/api/sessions")
def list_sessions(principal: CurrentPrincipal) -> dict[str, Any]:
    with _connect_sessions() as connection:
        rows = connection.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM ui_sessions
            WHERE actor_id = ?
            ORDER BY updated_at DESC
            """,
            (principal.user_id,),
        ).fetchall()
    return {"sessions": [dict(row) for row in rows]}


@app.post("/api/sessions", dependencies=[Depends(_require_csrf)])
def create_session(principal: CurrentPrincipal) -> dict[str, str]:
    _require_permission(principal, "agent:use")
    thread_id = uuid4().hex[:12]
    now = _now()
    with _connect_sessions() as connection:
        connection.execute(
            """
            INSERT INTO ui_sessions(actor_id, thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (principal.user_id, thread_id, "新建患者咨询", now, now),
        )
    return {"thread_id": thread_id, "title": "新建患者咨询"}


@app.get("/api/sessions/{thread_id}/messages")
def session_messages(
    thread_id: str, principal: CurrentPrincipal
) -> dict[str, Any]:
    _require_session(principal.user_id, thread_id)
    return {"messages": _load_messages(principal.user_id, thread_id)}


@app.post("/api/chat", dependencies=[Depends(_require_csrf)])
def chat(request: ChatRequest, principal: CurrentPrincipal) -> dict[str, Any]:
    _require_permission(principal, "agent:use")
    _require_permission(principal, "patient:read")
    _require_session(principal.user_id, request.thread_id)
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise HTTPException(status_code=503, detail="千问 API 尚未配置")

    request_id = str(uuid4())
    try:
        auth_repository.audit(
            "agent_query", "started", actor_user_id=principal.user_id,
            target_type="conversation", target_id=request.thread_id, request_id=request_id,
            details={"model": os.getenv("QWEN_MODEL", "qwen-plus")},
        )
        answer, trace = _run_agent(request, principal.user_id, request_id)
    except Exception as exc:
        logger.exception("Agent execution failed; request_id=%s", request_id)
        auth_repository.audit(
            "agent_query", "failure", actor_user_id=principal.user_id,
            target_type="conversation", target_id=request.thread_id, request_id=request_id,
        )
        raise HTTPException(status_code=500, detail="Agent 执行失败") from exc

    authorization_denied = any(
        step["node"] == "authorize_patient" and step["detail"] == "拒绝访问"
        for step in trace
    )
    patient_step = next((step for step in trace if step["node"] == "query_plan"), None)
    patient_id = patient_step.get("patient_id") if patient_step else None
    auth_repository.audit(
        "agent_query", "denied" if authorization_denied else "success",
        actor_user_id=principal.user_id,
        target_type="patient" if patient_id else "conversation",
        target_id=patient_id or request.thread_id, request_id=request_id,
        details={
            "conversation_id": request.thread_id,
            "model": os.getenv("QWEN_MODEL", "qwen-plus"),
        },
    )

    title = request.message.strip()[:28]
    with _connect_sessions() as connection:
        connection.execute(
            """
            UPDATE ui_sessions
            SET title = CASE WHEN title = '新建患者咨询' THEN ? ELSE title END,
                updated_at = ?
            WHERE actor_id = ? AND thread_id = ?
            """,
            (title, _now(), principal.user_id, request.thread_id),
        )
    return {"answer": answer, "trace": trace}


def main() -> None:
    """Start the local product interface."""

    uvicorn.run("medical_agent.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
