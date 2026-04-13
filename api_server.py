from __future__ import annotations

import os
import re
import sys
import uuid
import json
import asyncio
import random
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from importlib import import_module
from datetime import UTC, datetime, timedelta
from typing import Any, Optional
import jwt
from jwt import PyJWKClient
import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

# Support legacy absolute imports used across backend modules.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
# Legacy modules use relative file paths (e.g., payload_store3.json),
# so pin process cwd to backend directory for consistent resolution.
os.chdir(BACKEND_DIR)
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

PostgresSaver = None
try:
    postgres_module = import_module("langgraph.checkpoint.postgres")
    PostgresSaver = getattr(postgres_module, "PostgresSaver", None)
except Exception:
    PostgresSaver = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    thread_id: str
    assistant_text: str
    sql_query: Optional[str] = None
    result_summary: Optional[str] = None
    relevant_questions: Optional[list[str]] = None
    sql_result: Optional[dict[str, Any]] = None
    visualization_code: Optional[str] = None
    visualization_spec: Optional[str] = None
    visualization_meta: Optional[dict[str, Any]] = None
    visualization_figure: Optional[dict[str, Any]] = None


class VoteRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    rating: int
    user_query: Optional[str] = None
    assistant_response: Optional[str] = None


class DailyPulseUpdateRequest(BaseModel):
    questions: list[str] = Field(default_factory=list)


DEFAULT_DAILY_PULSE_QUESTIONS: tuple[str, ...] = (
    "Are we seeing strong short-term sales momentum?",
    "How are we doing in terms of adding new businesses?",
)


app = FastAPI(title="A360 Backend API", version="0.1.0")


DB_WARMUP_ON_STARTUP = os.getenv("DB_WARMUP_ON_STARTUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class DatabaseUnavailableError(RuntimeError):
    pass


@app.exception_handler(DatabaseUnavailableError)
async def _database_unavailable_handler(
    raw_request: Request, exc: DatabaseUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database is temporarily unavailable. Please retry shortly.",
        },
    )

# Optional testing fallback only. Keep empty in production.
TEST_DB_URI_FALLBACK = os.getenv("TEST_DB_URI_FALLBACK", "").strip()


def _build_db_uri() -> str:
    env_uri = (
        os.getenv("DB_URI", "").strip()
        or os.getenv("POSTGRES_URL", "").strip()
        or os.getenv("POSTGRES_URI", "").strip()
    )
    if env_uri:
        return env_uri

    host = (os.getenv("DB_HOST", "").strip() or os.getenv("PGHOST", "").strip())
    port_raw = (os.getenv("DB_PORT", "").strip() or os.getenv("PGPORT", "").strip() or "5432")
    database = (
        os.getenv("DB_NAME", "").strip()
        or os.getenv("POSTGRES_DB", "").strip()
        or os.getenv("PGDATABASE", "").strip()
        or "postgres"
    )
    user = (
        os.getenv("DB_USER", "").strip()
        or os.getenv("POSTGRES_USER", "").strip()
        or os.getenv("PGUSER", "").strip()
    )
    password = (
        os.getenv("DB_PASSWORD", "").strip()
        or os.getenv("POSTGRES_PASSWORD", "").strip()
        or os.getenv("PGPASSWORD", "").strip()
    )

    try:
        port = int(port_raw)
    except ValueError:
        port = 5432

    if host and user and password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    if TEST_DB_URI_FALLBACK and _env_flag("ALLOW_TEST_DB_FALLBACK", False):
        return TEST_DB_URI_FALLBACK

    raise RuntimeError("PostgreSQL connection is not configured in .env")


DB_URI = _build_db_uri()
logger = logging.getLogger(__name__)


def _get_db_connect_timeout() -> int:
    raw = os.getenv("DB_CONNECT_TIMEOUT", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


DB_CONNECT_TIMEOUT = _get_db_connect_timeout()


def _get_db_unavailable_cooldown_seconds() -> int:
    raw = os.getenv("DB_UNAVAILABLE_COOLDOWN_SECONDS", "20").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


DB_UNAVAILABLE_COOLDOWN_SECONDS = _get_db_unavailable_cooldown_seconds()


def _get_db_statement_timeout_ms() -> int:
    raw = os.getenv("DB_STATEMENT_TIMEOUT_MS", "8000").strip()
    try:
        return max(500, int(raw))
    except ValueError:
        return 8000


DB_STATEMENT_TIMEOUT_MS = _get_db_statement_timeout_ms()


def _get_db_bootstrap_statement_timeout_ms() -> int:
    raw = os.getenv("DB_BOOTSTRAP_STATEMENT_TIMEOUT_MS", "120000").strip()
    try:
        return max(DB_STATEMENT_TIMEOUT_MS, int(raw))
    except ValueError:
        return max(DB_STATEMENT_TIMEOUT_MS, 120000)


DB_BOOTSTRAP_STATEMENT_TIMEOUT_MS = _get_db_bootstrap_statement_timeout_ms()


def _get_history_rebuild_timeout_seconds() -> float:
    raw = os.getenv("HISTORY_REBUILD_TIMEOUT_SECONDS", "2.0").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 2.0


HISTORY_REBUILD_TIMEOUT_SECONDS = _get_history_rebuild_timeout_seconds()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


AUTH_REQUIRED = _env_flag("AUTH_REQUIRED", True)
AZURE_AD_TENANT_ID = (
    os.getenv("AZURE_AD_TENANT_ID", "").strip()
    or os.getenv("MSAL_TENANT_ID", "").strip()
)
AZURE_AD_CLIENT_ID = (
    os.getenv("AZURE_AD_CLIENT_ID", "").strip()
    or os.getenv("MSAL_CLIENT_ID", "").strip()
    or os.getenv("MSAL_AUDIENCE", "").strip()
)
AZURE_AD_ISSUER = (
    os.getenv("AZURE_AD_ISSUER", "").strip()
    or (
        f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0"
        if AZURE_AD_TENANT_ID
        else ""
    )
)
AZURE_AD_JWKS_URL = (
    os.getenv("AZURE_AD_JWKS_URL", "").strip()
    or (
        f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
        if AZURE_AD_TENANT_ID
        else ""
    )
)
_jwks_client: Optional[PyJWKClient] = None


def _extract_bearer_token(raw_request: Request) -> Optional[str]:
    authorization = raw_request.headers.get("Authorization", "").strip()
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None

    normalized = token.strip()
    return normalized or None


def _decode_unverified_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}

    return {}


def _verify_azure_ad_token(token: str) -> dict[str, Any]:
    global _jwks_client

    if not AZURE_AD_TENANT_ID or not AZURE_AD_CLIENT_ID or not AZURE_AD_ISSUER:
        raise HTTPException(
            status_code=500,
            detail=(
                "Azure AD auth is enabled but required env vars are missing. "
                "Set AZURE_AD_TENANT_ID and AZURE_AD_CLIENT_ID."
            ),
        )

    if not AZURE_AD_JWKS_URL:
        raise HTTPException(status_code=500, detail="AZURE_AD_JWKS_URL is not configured")

    if _jwks_client is None:
        _jwks_client = PyJWKClient(AZURE_AD_JWKS_URL)

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=AZURE_AD_CLIENT_ID,
            issuer=AZURE_AD_ISSUER,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Token payload is invalid")

    return payload


def _build_user_context(payload: dict[str, Any], authenticated: bool) -> dict[str, Any]:
    user_id = str(payload.get("oid") or payload.get("sub") or "").strip()
    email = str(
        payload.get("preferred_username")
        or payload.get("upn")
        or payload.get("email")
        or ""
    ).strip()

    return {
        "user_id": user_id or "local-user",
        "email": email or None,
        "authenticated": authenticated,
    }


def _get_request_user(raw_request: Request, require_auth: bool = AUTH_REQUIRED) -> dict[str, Any]:
    token = _extract_bearer_token(raw_request)
    if not token:
        if require_auth:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        return {
            "user_id": "local-user",
            "email": None,
            "authenticated": False,
        }

    # Strict verification path for Azure AD backed idTokens.
    if AZURE_AD_TENANT_ID and AZURE_AD_CLIENT_ID:
        payload = _verify_azure_ad_token(token)
        context = _build_user_context(payload, authenticated=True)
        if require_auth and context["user_id"] == "local-user":
            raise HTTPException(status_code=401, detail="Token does not include user identity")
        return context

    # Compatibility fallback when Azure verification is not configured yet.
    payload = _decode_unverified_token(token)
    context = _build_user_context(payload, authenticated=False)
    if require_auth and context["user_id"] == "local-user":
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload; cannot resolve user identity",
        )
    return context


def _init_feedback_db(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_key TEXT PRIMARY KEY,
            applied_at TEXT
        )
        """
    )

    # Lightweight compatibility guard: always ensure thread_registry has hidden-flag shape
    # even when bootstrap has already run in older deployments.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_registry (
            user_id TEXT,
            thread_id TEXT,
            created_at TEXT,
            title TEXT,
            is_hidden BOOLEAN,
            hidden_at TEXT
        )
        """
    )
    conn.execute("ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS is_hidden BOOLEAN")
    conn.execute("ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS hidden_at TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS thread_registry_user_hidden_created_idx
        ON thread_registry (user_id, is_hidden, created_at)
        """
    )

    bootstrap_migration_key = "schema_bootstrap_v2"
    bootstrap_row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_key=%s",
        (bootstrap_migration_key,),
    ).fetchone()
    if bootstrap_row:
        conn.commit()
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_feedback (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            thread_id TEXT,
            message_id TEXT,
            user_query TEXT,
            assistant_response TEXT,
            rating INTEGER,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hidden_threads (
            user_id TEXT,
            thread_id TEXT,
            hidden_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_registry (
            user_id TEXT,
            thread_id TEXT,
            created_at TEXT,
            title TEXT,
            is_hidden BOOLEAN,
            hidden_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_message_cache (
            user_id TEXT,
            thread_id TEXT,
            messages_json TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )

    migration_key = "user_scope_v1"
    migration_row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_key=%s",
        (migration_key,),
    ).fetchone()

    # Ensure modern columns exist irrespective of older migration markers.
    conn.execute("ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS is_hidden BOOLEAN")
    conn.execute("ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS hidden_at TEXT")

    # One-time legacy backfill/migration. Running this every startup can block history for minutes.
    if not migration_row:
        conn.execute("ALTER TABLE message_feedback ADD COLUMN IF NOT EXISTS user_id TEXT")
        conn.execute("ALTER TABLE hidden_threads ADD COLUMN IF NOT EXISTS user_id TEXT")
        conn.execute("ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS user_id TEXT")
        conn.execute("ALTER TABLE thread_message_cache ADD COLUMN IF NOT EXISTS user_id TEXT")

        conn.execute(
            """
            UPDATE message_feedback
            SET user_id='local-user'
            WHERE user_id IS NULL OR TRIM(user_id)=''
            """
        )
        conn.execute(
            """
            UPDATE hidden_threads
            SET user_id='local-user'
            WHERE user_id IS NULL OR TRIM(user_id)=''
            """
        )
        conn.execute(
            """
            UPDATE thread_registry
            SET user_id='local-user'
            WHERE user_id IS NULL OR TRIM(user_id)=''
            """
        )
        conn.execute(
            """
            UPDATE thread_message_cache
            SET user_id='local-user'
            WHERE user_id IS NULL OR TRIM(user_id)=''
            """
        )

        conn.execute("ALTER TABLE message_feedback ALTER COLUMN user_id SET NOT NULL")
        conn.execute("ALTER TABLE hidden_threads ALTER COLUMN user_id SET NOT NULL")
        conn.execute("ALTER TABLE thread_registry ALTER COLUMN user_id SET NOT NULL")
        conn.execute("ALTER TABLE thread_message_cache ALTER COLUMN user_id SET NOT NULL")

        conn.execute("ALTER TABLE hidden_threads DROP CONSTRAINT IF EXISTS hidden_threads_pkey")
        conn.execute("ALTER TABLE thread_registry DROP CONSTRAINT IF EXISTS thread_registry_pkey")
        conn.execute("ALTER TABLE thread_message_cache DROP CONSTRAINT IF EXISTS thread_message_cache_pkey")

        conn.execute(
            """
            INSERT INTO schema_migrations (migration_key, applied_at)
            VALUES (%s, %s)
            ON CONFLICT (migration_key) DO NOTHING
            """,
            (migration_key, datetime.now(UTC).isoformat()),
        )

    hidden_flag_migration_key = "hidden_flag_v1"
    hidden_flag_migration_row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_key=%s",
        (hidden_flag_migration_key,),
    ).fetchone()
    if not hidden_flag_migration_row:
        conn.execute(
            """
            UPDATE thread_registry
            SET is_hidden=FALSE
            WHERE is_hidden IS NULL
            """
        )
        conn.execute(
            """
            UPDATE thread_registry r
            SET is_hidden=TRUE, hidden_at=COALESCE(r.hidden_at, h.hidden_at)
            FROM hidden_threads h
            WHERE r.user_id=h.user_id AND r.thread_id=h.thread_id
            """
        )
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_key, applied_at)
            VALUES (%s, %s)
            ON CONFLICT (migration_key) DO NOTHING
            """,
            (hidden_flag_migration_key, datetime.now(UTC).isoformat()),
        )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS hidden_threads_user_thread_unique
        ON hidden_threads (user_id, thread_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS thread_registry_user_thread_unique
        ON thread_registry (user_id, thread_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS thread_message_cache_user_thread_unique
        ON thread_message_cache (user_id, thread_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS message_feedback_user_thread_message_unique
        ON message_feedback (user_id, thread_id, message_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS thread_registry_user_created_idx
        ON thread_registry (user_id, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS thread_registry_user_hidden_created_idx
        ON thread_registry (user_id, is_hidden, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS hidden_threads_user_idx
        ON hidden_threads (user_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS message_feedback_user_thread_idx
        ON message_feedback (user_id, thread_id)
        """
    )
    conn.execute(
        """
        INSERT INTO schema_migrations (migration_key, applied_at)
        VALUES (%s, %s)
        ON CONFLICT (migration_key) DO NOTHING
        """,
        (bootstrap_migration_key, datetime.now(UTC).isoformat()),
    )
    conn.commit()


db_pool: Optional[ConnectionPool] = None
checkpointer_conn: Optional[psycopg.Connection] = None
db_unavailable_until: Optional[datetime] = None
db_ready = False
db_last_error: Optional[str] = None
db_last_success_at: Optional[str] = None
db_retry_attempts = 0
db_retry_task: Optional[asyncio.Task[Any]] = None
db_retry_task_running = False

_ENSURED_TABLES: set[str] = set()
_ENSURE_TABLES_LOCK = threading.Lock()


def _log_db_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.log(level, " ".join(parts))


def _mark_db_ready_state(ready: bool, error: Optional[str] = None) -> None:
    global db_ready
    global db_last_error
    global db_last_success_at

    if db_ready != ready:
        _log_db_event("db.ready_state_changed", ready=ready, error=error)

    db_ready = ready
    if ready:
        db_last_error = None
        db_last_success_at = datetime.now(UTC).isoformat()
    elif error:
        db_last_error = error


def _extract_missing_relation(exc: Exception) -> Optional[str]:
    table_name = getattr(getattr(exc, "diag", None), "table_name", None)
    if isinstance(table_name, str) and table_name.strip():
        return table_name.strip()

    message = str(exc)
    match = re.search(r'relation\s+"([^"]+)"\s+does not exist', message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _table_ensure_statements(table_name: str) -> list[str]:
    table_map: dict[str, list[str]] = {
        "thread_registry": [
            """
            CREATE TABLE IF NOT EXISTS thread_registry (
                user_id TEXT,
                thread_id TEXT,
                created_at TEXT,
                title TEXT,
                is_hidden BOOLEAN,
                hidden_at TEXT
            )
            """,
            "ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS is_hidden BOOLEAN",
            "ALTER TABLE thread_registry ADD COLUMN IF NOT EXISTS hidden_at TEXT",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS thread_registry_user_thread_unique
            ON thread_registry (user_id, thread_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS thread_registry_user_hidden_created_idx
            ON thread_registry (user_id, is_hidden, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS thread_registry_user_created_idx
            ON thread_registry (user_id, created_at)
            """,
        ],
        "thread_message_cache": [
            """
            CREATE TABLE IF NOT EXISTS thread_message_cache (
                user_id TEXT,
                thread_id TEXT,
                messages_json TEXT NOT NULL,
                updated_at TEXT
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS thread_message_cache_user_thread_unique
            ON thread_message_cache (user_id, thread_id)
            """,
        ],
        "message_feedback": [
            """
            CREATE TABLE IF NOT EXISTS message_feedback (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                thread_id TEXT,
                message_id TEXT,
                user_query TEXT,
                assistant_response TEXT,
                rating INTEGER,
                created_at TEXT
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS message_feedback_user_thread_message_unique
            ON message_feedback (user_id, thread_id, message_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS message_feedback_user_thread_idx
            ON message_feedback (user_id, thread_id)
            """,
        ],
        "daily_pulse_questions": [
            """
            CREATE TABLE IF NOT EXISTS daily_pulse_questions (
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS daily_pulse_questions_user_idx
            ON daily_pulse_questions (user_id)
            """,
        ],
    }
    return table_map.get(table_name, [])


def _ensure_table_if_needed(table_name: str) -> bool:
    if not table_name:
        return False

    if table_name in _ENSURED_TABLES:
        return True

    statements = _table_ensure_statements(table_name)
    if not statements:
        return False

    with _ENSURE_TABLES_LOCK:
        if table_name in _ENSURED_TABLES:
            return True

        _log_db_event("db.ensure_table_started", table=table_name)
        try:
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                for statement in statements:
                    conn.execute(statement)
            _ENSURED_TABLES.add(table_name)
            _log_db_event("db.ensure_table_success", table=table_name)
            return True
        except Exception as exc:
            _log_db_event(
                "db.ensure_table_failed",
                level=logging.WARNING,
                table=table_name,
                error_class=exc.__class__.__name__,
                error=str(exc),
            )
            return False


def _record_db_failure(exc: Exception) -> None:
    global db_unavailable_until

    db_unavailable_until = datetime.now(UTC) + timedelta(seconds=DB_UNAVAILABLE_COOLDOWN_SECONDS)
    _mark_db_ready_state(False, error=str(exc))
    _log_db_event(
        "db.connection_failure",
        level=logging.WARNING,
        error_class=exc.__class__.__name__,
        error=str(exc),
    )


@app.on_event("startup")
async def _startup_db_warmup() -> None:
    global db_retry_task
    global db_retry_task_running

    if db_retry_task_running:
        return

    db_retry_task_running = True
    db_retry_task = asyncio.create_task(_db_retry_loop(), name="db-retry-loop")
    _log_db_event("db.startup_retry_started", warmup_enabled=DB_WARMUP_ON_STARTUP)


async def _db_retry_loop() -> None:
    global db_retry_attempts

    base_delay = 1.0
    max_delay = 30.0
    delay = base_delay

    while True:
        db_retry_attempts += 1
        attempt = db_retry_attempts
        _log_db_event("db.startup_retry_attempt", attempt=attempt, delay_s=round(delay, 2))
        try:
            _ensure_db_pool()
            _mark_db_ready_state(True)
            _log_db_event("db.startup_retry_success", attempt=attempt)
            delay = base_delay
            await asyncio.sleep(60.0)
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _record_db_failure(exc)
            _log_db_event(
                "db.startup_retry_failed",
                level=logging.WARNING,
                attempt=attempt,
                error_class=exc.__class__.__name__,
                error=str(exc),
            )

        jitter_factor = 1.0 + random.uniform(-0.2, 0.2)
        await asyncio.sleep(max(0.5, delay * jitter_factor))
        delay = min(max_delay, delay * 2)


@app.on_event("shutdown")
async def _shutdown_db_connections() -> None:
    global db_pool
    global checkpointer_conn
    global db_retry_task
    global db_retry_task_running

    db_retry_task_running = False
    if db_retry_task is not None:
        db_retry_task.cancel()
        try:
            await db_retry_task
        except asyncio.CancelledError:
            pass
        finally:
            db_retry_task = None

    if db_pool is not None:
        try:
            db_pool.close()
        finally:
            db_pool = None

    if checkpointer_conn is not None:
        try:
            checkpointer_conn.close()
        finally:
            checkpointer_conn = None


def _configure_runtime_connection(conn: psycopg.Connection) -> None:
    # Pool configure callback must leave connections idle (not INTRANS).
    previous_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        conn.execute(f"SET statement_timeout = {DB_STATEMENT_TIMEOUT_MS}")
    finally:
        conn.autocommit = previous_autocommit


def _ensure_db_pool() -> ConnectionPool:
    global db_pool
    global db_unavailable_until

    if db_pool is not None:
        return db_pool

    try:
        pool = ConnectionPool(
            conninfo=DB_URI,
            min_size=max(1, int(os.getenv("DB_POOL_MIN_SIZE", "1"))),
            max_size=max(1, int(os.getenv("DB_POOL_MAX_SIZE", "10"))),
            timeout=DB_CONNECT_TIMEOUT,
            configure=_configure_runtime_connection,
            kwargs={
                "autocommit": False,
                "connect_timeout": DB_CONNECT_TIMEOUT,
            },
            open=True,
        )
        db_pool = pool
        db_unavailable_until = None
        return pool
    except Exception as exc:
        _record_db_failure(exc)
        raise


def _ensure_checkpointer_connection() -> psycopg.Connection:
    global checkpointer_conn

    if checkpointer_conn is not None:
        return checkpointer_conn

    checkpointer_conn = psycopg.connect(
        DB_URI,
        autocommit=False,
        connect_timeout=DB_CONNECT_TIMEOUT,
    )
    checkpointer_conn.execute(f"SET statement_timeout = {DB_STATEMENT_TIMEOUT_MS}")
    checkpointer_conn.commit()
    return checkpointer_conn


def _db_fetchall(query: str, params: Optional[tuple[Any, ...]] = None) -> list[Any]:
    global db_unavailable_until

    now = datetime.now(UTC)
    if db_unavailable_until and now < db_unavailable_until:
        raise DatabaseUnavailableError("PostgreSQL connection is unavailable")

    ensured_missing_table = False
    for attempt in (1, 2):
        try:
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                db_unavailable_until = None
                cursor = conn.execute(query, params) if params is not None else conn.execute(query)
                _mark_db_ready_state(True)
                return cursor.fetchall()
        except psycopg.errors.UndefinedTable as exc:
            table_name = _extract_missing_relation(exc)
            _log_db_event(
                "db.undefined_table_detected",
                level=logging.WARNING,
                table=table_name,
                query_preview=query.strip().splitlines()[0][:120] if query else "",
            )
            if ensured_missing_table or not table_name or not _ensure_table_if_needed(table_name):
                raise DatabaseUnavailableError("Required database table is missing") from exc
            ensured_missing_table = True
            _log_db_event("db.query_retry_after_ensure", table=table_name)
            continue
        except psycopg.errors.InFailedSqlTransaction as exc:
            if attempt == 1:
                continue
            raise DatabaseUnavailableError("PostgreSQL transaction is in failed state") from exc
        except psycopg.errors.QueryCanceled as exc:
            logger.warning("PostgreSQL query timeout: %s", exc.__class__.__name__)
            raise DatabaseUnavailableError("PostgreSQL query timed out") from exc
        except (PoolTimeout, psycopg.OperationalError, psycopg.InterfaceError, psycopg.errors.ConnectionTimeout) as exc:
            if attempt == 1:
                continue
            _record_db_failure(exc)
            raise DatabaseUnavailableError("PostgreSQL connection is unavailable") from exc

    raise DatabaseUnavailableError("PostgreSQL query execution failed")


def _db_fetchone(query: str, params: Optional[tuple[Any, ...]] = None) -> Any:
    global db_unavailable_until

    now = datetime.now(UTC)
    if db_unavailable_until and now < db_unavailable_until:
        raise DatabaseUnavailableError("PostgreSQL connection is unavailable")

    ensured_missing_table = False
    for attempt in (1, 2):
        try:
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                db_unavailable_until = None
                cursor = conn.execute(query, params) if params is not None else conn.execute(query)
                _mark_db_ready_state(True)
                return cursor.fetchone()
        except psycopg.errors.UndefinedTable as exc:
            table_name = _extract_missing_relation(exc)
            _log_db_event(
                "db.undefined_table_detected",
                level=logging.WARNING,
                table=table_name,
                query_preview=query.strip().splitlines()[0][:120] if query else "",
            )
            if ensured_missing_table or not table_name or not _ensure_table_if_needed(table_name):
                raise DatabaseUnavailableError("Required database table is missing") from exc
            ensured_missing_table = True
            _log_db_event("db.query_retry_after_ensure", table=table_name)
            continue
        except psycopg.errors.InFailedSqlTransaction as exc:
            if attempt == 1:
                continue
            raise DatabaseUnavailableError("PostgreSQL transaction is in failed state") from exc
        except psycopg.errors.QueryCanceled as exc:
            logger.warning("PostgreSQL query timeout: %s", exc.__class__.__name__)
            raise DatabaseUnavailableError("PostgreSQL query timed out") from exc
        except (PoolTimeout, psycopg.OperationalError, psycopg.InterfaceError, psycopg.errors.ConnectionTimeout) as exc:
            if attempt == 1:
                continue
            _record_db_failure(exc)
            raise DatabaseUnavailableError("PostgreSQL connection is unavailable") from exc

    raise DatabaseUnavailableError("PostgreSQL query execution failed")


def _db_execute(query: str, params: Optional[tuple[Any, ...]] = None) -> None:
    global db_unavailable_until

    now = datetime.now(UTC)
    if db_unavailable_until and now < db_unavailable_until:
        raise DatabaseUnavailableError("PostgreSQL connection is unavailable")

    ensured_missing_table = False
    for attempt in (1, 2):
        try:
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                db_unavailable_until = None
                if params is None:
                    conn.execute(query)
                else:
                    conn.execute(query, params)
                _mark_db_ready_state(True)
                return
        except psycopg.errors.UndefinedTable as exc:
            table_name = _extract_missing_relation(exc)
            _log_db_event(
                "db.undefined_table_detected",
                level=logging.WARNING,
                table=table_name,
                query_preview=query.strip().splitlines()[0][:120] if query else "",
            )
            if ensured_missing_table or not table_name or not _ensure_table_if_needed(table_name):
                raise DatabaseUnavailableError("Required database table is missing") from exc
            ensured_missing_table = True
            _log_db_event("db.query_retry_after_ensure", table=table_name)
            continue
        except psycopg.errors.InFailedSqlTransaction as exc:
            if attempt == 1:
                continue
            raise DatabaseUnavailableError("PostgreSQL transaction is in failed state") from exc
        except psycopg.errors.QueryCanceled as exc:
            logger.warning("PostgreSQL query timeout: %s", exc.__class__.__name__)
            raise DatabaseUnavailableError("PostgreSQL query timed out") from exc
        except (PoolTimeout, psycopg.OperationalError, psycopg.InterfaceError, psycopg.errors.ConnectionTimeout) as exc:
            if attempt == 1:
                continue
            _record_db_failure(exc)
            raise DatabaseUnavailableError("PostgreSQL connection is unavailable") from exc

    raise DatabaseUnavailableError("PostgreSQL query execution failed")


def _build_checkpointer() -> Any:
    try:
        if PostgresSaver is None:
            return MemorySaver()
        checkpointer_instance = PostgresSaver(_ensure_checkpointer_connection())
        checkpointer_instance.setup()
        return checkpointer_instance
    except Exception:
        # Fallback keeps development unblocked if postgres saver is unavailable.
        return MemorySaver()


checkpointer = None
chatbot = None
stream_subgraph = None


def _build_rag_examples_for_question(question: str) -> tuple[str, str, list[str]]:
    from chatbot8 import build_rag_examples, get_intent_summary

    intent = question
    try:
        intent = str(get_intent_summary(question)).strip() or question
    except Exception as exc:
        # Keep stream path resilient if intent extraction fails transiently.
        logger.warning("Intent extraction failed for RAG examples: %s", exc)

    return build_rag_examples(question, intent)


def _get_chatbot() -> Any:
    global checkpointer
    global chatbot
    if chatbot is None:
        if checkpointer is None:
            checkpointer = _build_checkpointer()
        from chatbot8 import build_chatbot

        chatbot = build_chatbot(checkpointer=checkpointer)
    return chatbot


def _get_stream_subgraph() -> Any:
    global stream_subgraph
    if stream_subgraph is None:
        from subgraph_7 import build_graph as build_stream_graph

        stream_subgraph = build_stream_graph(checkpointer=None)
    return stream_subgraph


def _parse_agent_output(text: str) -> dict[str, Optional[str]]:
    sections = {
        "sql_query": None,
        "result_summary": None,
        "query_results": None,
        "visualization_code": None,
        "visualization_spec": None,
        "relevant_questions": None,
    }

    header_map = {
        "sql_query": "SQL Query Executed:",
        "result_summary": "Result Summary:",
        "query_results": "Query Results:",
        "visualization_code": "Visualization Code:",
        "visualization_spec": "Visualization Spec:",
        "relevant_questions": "Relevant Questions:",
    }

    all_headers_pattern = "|".join(re.escape(h) for h in header_map.values())
    patterns = {
        key: rf"{re.escape(header)}\s*(.*?)(?=\n(?:{all_headers_pattern})|$)"
        for key, header in header_map.items()
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections


def _strip_code_fences(code: str) -> str:
    stripped = code.strip()
    stripped = re.sub(r"^```(?:python)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _normalize_plotly_json(figure_json: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        return json.loads(json.dumps(figure_json, cls=PlotlyJSONEncoder))
    except Exception:
        return None


_KNOWN_IMPORT_PREFIXES = (
    "import pandas",
    "import plotly",
    "from plotly",
    "import numpy",
    "import math",
)


def _strip_known_imports(code: str) -> str:
    lines = code.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if any(stripped.startswith(prefix) for prefix in _KNOWN_IMPORT_PREFIXES):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _build_plotly_figure_json(
    visualization_code: Optional[str],
    sql_result: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not visualization_code or not sql_result:
        return None

    raw_code = visualization_code.strip()
    if not raw_code or raw_code.upper() == "NO_VISUALIZATION":
        return None

    rows = sql_result.get("data")
    if not isinstance(rows, list):
        return None

    code = _strip_known_imports(_strip_code_fences(raw_code))
    if not code:
        return None

    df = pd.DataFrame(rows)

    allowed_imports = {
        "math": "math",
        "numpy": "numpy",
        "pandas": "pandas",
        "plotly": "plotly",
        "plotly.express": "plotly.express",
        "plotly.graph_objects": "plotly.graph_objects",
        "plotly.subplots": "plotly.subplots",
    }

    def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0):
        module_name = allowed_imports.get(name)
        if not module_name:
            raise ImportError(f"Import not allowed: {name}")
        return __import__(module_name, globals, locals, fromlist, level)

    safe_builtins = {
        "abs": abs,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": safe_import,
    }
    safe_globals: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df,
        "pd": pd,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
    }

    try:
        import math

        safe_globals["math"] = math
    except Exception:
        pass

    try:
        import numpy as np

        safe_globals["np"] = np
    except Exception:
        pass
    safe_locals: dict[str, Any] = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as exc:
        logger.warning("Visualization code execution failed: %s", exc)
        return None

    fig = safe_locals.get("fig") or safe_globals.get("fig")
    if fig is None or not hasattr(fig, "to_plotly_json"):
        return None

    try:
        figure_json = fig.to_plotly_json()
        if not isinstance(figure_json, dict):
            return None
    except Exception:
        return None

    normalized = _normalize_plotly_json(figure_json)
    if normalized is None:
        return None

    config = normalized.get("config")
    if not isinstance(config, dict):
        config = {}

    normalized["config"] = {
        "displaylogo": False,
        "responsive": True,
        **config,
    }
    return normalized


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            float(text)
            return True
        except Exception:
            return False
    return False


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return 0.0
    return 0.0


def _build_heuristic_plotly_figure_json(
    sql_result: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not sql_result:
        return None

    rows = sql_result.get("data")
    if not isinstance(rows, list) or not rows:
        return None

    columns = sql_result.get("columns")
    if not isinstance(columns, list) or not columns:
        sample = rows[0] if isinstance(rows[0], dict) else {}
        columns = list(sample.keys()) if isinstance(sample, dict) else []
    if not columns:
        return None

    sample_rows = [row for row in rows[:30] if isinstance(row, dict)]
    if not sample_rows:
        return None

    numeric_columns = [
        col
        for col in columns
        if any(_is_numeric_like(row.get(col)) for row in sample_rows)
    ]
    if not numeric_columns:
        return None

    category_column = next((col for col in columns if col not in numeric_columns), columns[0])
    metric_columns = [col for col in numeric_columns if col != category_column][:2]
    if not metric_columns:
        return None

    top_rows = [row for row in rows[:20] if isinstance(row, dict)]
    x_values = [str(row.get(category_column, "")) for row in top_rows]

    fig = go.Figure()
    for metric in metric_columns:
        y_values = [_to_float(row.get(metric)) for row in top_rows]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                name=str(metric),
            )
        )

    fig.update_layout(
        barmode="group",
        margin={"l": 40, "r": 20, "t": 24, "b": 80},
        xaxis={"title": str(category_column), "tickangle": -35},
        yaxis={"title": "Value"},
        template="plotly_white",
    )

    try:
        figure_json = fig.to_plotly_json()
        if not isinstance(figure_json, dict):
            return None
    except Exception:
        return None

    normalized = _normalize_plotly_json(figure_json)
    if normalized is None:
        return None

    config = normalized.get("config")
    if not isinstance(config, dict):
        config = {}

    normalized["config"] = {
        "displaylogo": False,
        "responsive": True,
        **config,
    }
    return normalized


def _feedback_exists(user_id: str, thread_id: str, message_id: str) -> bool:
    row = _db_fetchone(
        (
            "SELECT 1 FROM message_feedback "
            "WHERE user_id=%s AND thread_id=%s AND message_id=%s LIMIT 1"
        ),
        (user_id, thread_id, message_id),
    )
    return row is not None


def _save_feedback_if_missing(request: VoteRequest, user_id: str) -> bool:
    if _feedback_exists(user_id, request.thread_id, request.message_id):
        return False

    _db_execute(
        """
        INSERT INTO message_feedback
        (user_id, thread_id, message_id, user_query, assistant_response, rating, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            request.thread_id,
            request.message_id,
            request.user_query,
            request.assistant_response,
            request.rating,
            datetime.now(UTC).isoformat(),
        ),
    )
    return True


def _extract_thread_timestamp(thread_id: str) -> datetime:
    try:
        ts = thread_id[:-1] if thread_id.endswith("T") else thread_id
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=UTC)


def _checkpoint_thread_id(thread_id: str, user_id: str) -> str:
    return f"{user_id}:{thread_id}"


def _register_thread_if_missing(thread_id: str, user_id: str) -> None:
    _db_execute(
        """
        INSERT INTO thread_registry (user_id, thread_id, created_at, title)
        VALUES (%s, %s, %s, NULL)
        ON CONFLICT (user_id, thread_id) DO NOTHING
        """,
        (user_id, thread_id, datetime.now(UTC).isoformat()),
    )


def _set_thread_title_if_missing(
    thread_id: str,
    user_id: str,
    title: Optional[str],
    max_len: int = 80,
) -> None:
    if not title:
        return

    normalized = title.strip().replace("\n", " ")
    if not normalized:
        return

    truncated = normalized[:max_len] + ("..." if len(normalized) > max_len else "")

    _db_execute(
        """
        UPDATE thread_registry
        SET title = %s
        WHERE user_id = %s AND thread_id = %s AND (title IS NULL OR TRIM(title) = '')
        """,
        (truncated, user_id, thread_id),
    )


def _get_thread_created_at(thread_id: str, user_id: str) -> datetime:
    row = _db_fetchone(
        "SELECT created_at FROM thread_registry WHERE user_id=%s AND thread_id=%s",
        (user_id, thread_id),
    )

    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            pass

    return _extract_thread_timestamp(thread_id)


def _coerce_thread_created_at(thread_id: str, raw_created_at: Any) -> datetime:
    if raw_created_at:
        try:
            return datetime.fromisoformat(str(raw_created_at))
        except Exception:
            pass
    return _extract_thread_timestamp(thread_id)


SEARCH_TEXT_MAX_CHARS = 20000
SEARCH_SQL_MAX_ROWS = 25
SEARCH_SQL_MAX_CELL_CHARS = 100
SEARCH_VIS_MAX_TRACES = 25
SEARCH_VIS_CODE_MAX_CHARS = 2000


def _normalize_search_text(text: str) -> str:
    return " ".join(text.split())


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text[:limit]


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _extend_sql_search_chunks(chunks: list[str], sql_result: dict[str, Any]) -> None:
    columns = sql_result.get("columns")
    if isinstance(columns, list) and columns:
        chunks.append(" ".join(str(col) for col in columns if col is not None))

    rows = sql_result.get("data")
    if not isinstance(rows, list):
        return

    for row in rows[:SEARCH_SQL_MAX_ROWS]:
        values: list[Any]
        if isinstance(row, dict):
            if isinstance(columns, list) and columns:
                values = [row.get(col) for col in columns]
            else:
                values = list(row.values())
        elif isinstance(row, (list, tuple)):
            values = list(row)
        else:
            values = [row]

        for value in values:
            text = _coerce_text(value)
            if not text:
                continue
            chunks.append(_truncate_text(text, SEARCH_SQL_MAX_CELL_CHARS))


def _extend_visualization_search_chunks(chunks: list[str], figure: dict[str, Any]) -> None:
    layout = figure.get("layout")
    if isinstance(layout, dict):
        title = layout.get("title")
        if isinstance(title, dict):
            text = _coerce_text(title.get("text"))
            if text:
                chunks.append(text)
        else:
            text = _coerce_text(title)
            if text:
                chunks.append(text)

        for axis_key in ("xaxis", "yaxis"):
            axis = layout.get(axis_key)
            if isinstance(axis, dict):
                axis_title = axis.get("title")
                if isinstance(axis_title, dict):
                    text = _coerce_text(axis_title.get("text"))
                else:
                    text = _coerce_text(axis_title)
                if text:
                    chunks.append(text)

    traces = figure.get("data")
    if isinstance(traces, list):
        for trace in traces[:SEARCH_VIS_MAX_TRACES]:
            if not isinstance(trace, dict):
                continue
            name = _coerce_text(trace.get("name"))
            if name:
                chunks.append(name)


def _build_search_text_from_cached(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        parts = message.get("parts")
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue

            part_type = part.get("type")
            if part_type == "data-relevantQuestions":
                continue

            if part_type == "text":
                text = _coerce_text(part.get("text"))
                if text:
                    chunks.append(text)
                continue

            if part_type == "data-resultSummary":
                text = _coerce_text(part.get("data"))
                if text:
                    chunks.append(text)
                continue

            if part_type == "data-sqlQuery":
                text = _coerce_text(part.get("data"))
                if text:
                    chunks.append(text)
                continue

            if part_type == "data-sqlColumns":
                data = part.get("data")
                if isinstance(data, list) and data:
                    chunks.append(" ".join(str(col) for col in data if col is not None))
                else:
                    text = _coerce_text(data)
                    if text:
                        chunks.append(text)
                continue

            if part_type == "data-sqlResult":
                data = part.get("data")
                if isinstance(data, dict):
                    _extend_sql_search_chunks(chunks, data)
                continue

            if part_type == "data-visualizationFigure":
                data = part.get("data")
                if isinstance(data, dict):
                    _extend_visualization_search_chunks(chunks, data)
                continue

            if part_type == "data-visualizationCode":
                text = _coerce_text(part.get("data"))
                if text:
                    chunks.append(_truncate_text(text, SEARCH_VIS_CODE_MAX_CHARS))
                continue

    if not chunks:
        return ""

    normalized = _normalize_search_text(" ".join(chunks))
    return _truncate_text(normalized, SEARCH_TEXT_MAX_CHARS)


def _build_search_text_from_cached_payload(payload: Any) -> str:
    if isinstance(payload, list):
        return _build_search_text_from_cached(payload)

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return ""

        if isinstance(parsed, list):
            return _build_search_text_from_cached(parsed)

    return ""


def _load_cached_messages_with_presence(
    thread_id: str, user_id: str
) -> tuple[list[dict[str, Any]], bool]:
    row = _db_fetchone(
        "SELECT messages_json FROM thread_message_cache WHERE user_id=%s AND thread_id=%s",
        (user_id, thread_id),
    )

    if not row:
        return [], False

    if not row[0]:
        return [], True

    try:
        parsed = json.loads(row[0])
        if isinstance(parsed, list):
            return parsed, True
    except Exception:
        return [], True

    return [], True


def _thread_matches_search(
    chatbot_instance: Any,
    thread_id: str,
    user_id: str,
    query: str,
    thread_title: Optional[str] = None,
) -> bool:
    if not query:
        return True

    lowered = query.lower()
    if thread_title and lowered in thread_title.lower():
        return True

    cached_messages, has_cache = _load_cached_messages_with_presence(thread_id, user_id)
    if has_cache:
        cached_text = _build_search_text_from_cached(cached_messages)
        if cached_text and lowered in cached_text.lower():
            return True
        # Cache is a fast path. If cache misses (or is empty), fall back to
        # checkpoint state to avoid false negatives from stale/incomplete cache.

    try:
        state = chatbot_instance.get_state(
            config={"configurable": {"thread_id": _checkpoint_thread_id(thread_id, user_id)}}
        )
        messages = state.values.get("messages", [])
    except Exception:
        return False

    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and lowered in content.lower():
            return True

    return False


def _get_thread_title(
    chatbot_instance: Any,
    thread_id: str,
    user_id: str,
    max_len: int = 80,
) -> str:
    cached_row = _db_fetchone(
        "SELECT title FROM thread_registry WHERE user_id=%s AND thread_id=%s",
        (user_id, thread_id),
    )
    if cached_row and isinstance(cached_row[0], str) and cached_row[0].strip():
        return cached_row[0].strip()

    try:
        state = chatbot_instance.get_state(
            config={"configurable": {"thread_id": _checkpoint_thread_id(thread_id, user_id)}}
        )
        messages = state.values.get("messages", [])

        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type in {"human", "HumanMessage"}:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    title = content.strip().replace("\n", " ")
                    resolved = title[:max_len] + ("..." if len(title) > max_len else "")
                    _set_thread_title_if_missing(
                        thread_id,
                        user_id,
                        resolved,
                        max_len=max_len,
                    )
                    return resolved
    except Exception:
        pass

    # Fallback for stream-first threads before checkpoint message state is persisted.
    cached = _load_cached_messages(thread_id, user_id)
    for msg in cached:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        parts = msg.get("parts")
        if not isinstance(parts, list):
            continue

        text_chunks = [
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        text = " ".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip()).strip()
        if text:
            title = text.replace("\n", " ")
            resolved = title[:max_len] + ("..." if len(title) > max_len else "")
            _set_thread_title_if_missing(thread_id, user_id, resolved, max_len=max_len)
            return resolved

    return "Current conversation"


def _list_visible_threads(chatbot_instance: Any, user_id: str) -> list[str]:
    registry_rows = _db_fetchall(
        """
        SELECT thread_id
        FROM thread_registry
        WHERE user_id=%s AND COALESCE(is_hidden, FALSE)=FALSE
        """,
        (user_id,),
    )

    return [str(row[0]) for row in registry_rows if row and row[0]]


def _is_thread_visible(chatbot_instance: Any, thread_id: str, user_id: str) -> bool:
    registry_row = _db_fetchone(
        """
        SELECT 1
        FROM thread_registry
        WHERE user_id=%s
          AND thread_id=%s
          AND COALESCE(is_hidden, FALSE)=FALSE
        """,
        (user_id, thread_id),
    )
    return registry_row is not None


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                    continue

                if isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "\n".join(part for part in text_parts if part.strip())

    return ""


def _load_cached_messages(thread_id: str, user_id: str) -> list[dict[str, Any]]:
    row = _db_fetchone(
        "SELECT messages_json FROM thread_message_cache WHERE user_id=%s AND thread_id=%s",
        (user_id, thread_id),
    )
    if not row or not row[0]:
        return []

    try:
        parsed = json.loads(row[0])
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []


def _save_cached_messages(thread_id: str, user_id: str, messages: list[dict[str, Any]]) -> None:
    payload = json.dumps(messages, ensure_ascii=True, default=str)
    _db_execute(
        """
        INSERT INTO thread_message_cache (user_id, thread_id, messages_json, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(user_id, thread_id)
        DO UPDATE SET messages_json=EXCLUDED.messages_json, updated_at=EXCLUDED.updated_at
        """,
        (user_id, thread_id, payload, datetime.now(UTC).isoformat()),
    )


def _serialize_thread_messages(
    chatbot_instance: Any,
    thread_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    state = chatbot_instance.get_state(
        config={"configurable": {"thread_id": _checkpoint_thread_id(thread_id, user_id)}}
    )
    messages = state.values.get("messages", [])

    serialized: list[dict[str, Any]] = []
    last_assistant_index: Optional[int] = None

    def _append_assistant_data_part(part: dict[str, Any]) -> None:
        if last_assistant_index is None:
            return
        serialized[last_assistant_index]["parts"].append(part)

    def _get_assistant_sql_result() -> Optional[dict[str, Any]]:
        if last_assistant_index is None:
            return None
        for part in serialized[last_assistant_index].get("parts", []):
            if part.get("type") == "data-sqlResult" and isinstance(part.get("data"), dict):
                return part["data"]
        return None

    for message in messages:
        msg_type = getattr(message, "type", None)
        if msg_type in {"human", "HumanMessage"}:
            role = "user"
        elif msg_type in {"ai", "AIMessage"}:
            role = "assistant"
        elif msg_type in {"system", "SystemMessage"}:
            role = "system"
        else:
            continue

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}

        if role == "assistant" and isinstance(additional_kwargs, dict):
            structured_type = additional_kwargs.get("type")

            if structured_type == "sql_result":
                raw_data = additional_kwargs.get("data")
                if isinstance(raw_data, dict):
                    _append_assistant_data_part({"type": "data-sqlResult", "data": raw_data})

                    raw_columns = raw_data.get("columns")
                    if isinstance(raw_columns, list) and raw_columns:
                        _append_assistant_data_part(
                            {"type": "data-sqlColumns", "data": [str(column) for column in raw_columns]}
                        )

                    raw_rows = raw_data.get("data")
                    if isinstance(raw_rows, list):
                        _append_assistant_data_part(
                            {"type": "data-sqlRowCount", "data": len(raw_rows)}
                        )
                continue

            if structured_type == "visualization":
                raw_code = additional_kwargs.get("code")
                if isinstance(raw_code, str):
                    _append_assistant_data_part({"type": "data-visualizationCode", "data": raw_code})

                    figure_json = _build_plotly_figure_json(raw_code, _get_assistant_sql_result())
                    if isinstance(figure_json, dict) and figure_json.get("data"):
                        _append_assistant_data_part(
                            {"type": "data-visualizationFigure", "data": figure_json}
                        )
                continue

        content = _extract_text_from_content(getattr(message, "content", ""))
        if not content.strip():
            continue

        parts: list[dict[str, Any]] = [{"type": "text", "text": content}]

        if role == "assistant":
            parsed_sections = _parse_agent_output(content)

            if parsed_sections.get("sql_query"):
                parts.append({"type": "data-sqlQuery", "data": parsed_sections["sql_query"]})

            if parsed_sections.get("result_summary"):
                parts.append({"type": "data-resultSummary", "data": parsed_sections["result_summary"]})

            if parsed_sections.get("relevant_questions"):
                relevant_questions = [
                    line.strip().lstrip("-").strip()
                    for line in parsed_sections["relevant_questions"].splitlines()
                    if line.strip()
                ]
                if relevant_questions:
                    parts.append({"type": "data-relevantQuestions", "data": relevant_questions})

        serialized.append(
            {
                "id": str(uuid.uuid4()),
                "role": role,
                "parts": parts,
            }
        )

        if role == "assistant":
            last_assistant_index = len(serialized) - 1
        else:
            last_assistant_index = None

    return serialized


def _serialize_thread_messages_with_timeout(
    chatbot_instance: Any,
    thread_id: str,
    user_id: str,
    timeout_seconds: float = HISTORY_REBUILD_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    # Guard slow checkpoint reconstruction so page render doesn't stall for minutes.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            _serialize_thread_messages,
            chatbot_instance,
            thread_id,
            user_id,
        )
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        logger.warning(
            "History message reconstruction timed out for user=%s thread=%s after %.2fs",
            user_id,
            thread_id,
            timeout_seconds,
        )
    except Exception as exc:
        logger.warning(
            "History message reconstruction failed for user=%s thread=%s: %s",
            user_id,
            thread_id,
            exc,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return []


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db": {
            "ready": db_ready,
            "retryAttempts": db_retry_attempts,
            "lastError": db_last_error,
            "lastSuccessAt": db_last_success_at,
        },
    }


@app.get("/api/v1/daily-pulse/questions")
def get_daily_pulse_questions(raw_request: Request) -> dict[str, Any]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]

    for attempt in (1, 2):
        questions: list[str] = []
        try:
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT question
                    FROM daily_pulse_questions
                    WHERE user_id=%s
                    ORDER BY order_index ASC, created_at ASC
                    """,
                    (current_user_id,),
                ).fetchall()
                questions = [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]

                if not questions:
                    now_iso = datetime.now(UTC).isoformat()
                    for index, question in enumerate(DEFAULT_DAILY_PULSE_QUESTIONS):
                        conn.execute(
                            """
                            INSERT INTO daily_pulse_questions (
                                user_id,
                                question,
                                order_index,
                                created_at,
                                updated_at
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (current_user_id, question, index, now_iso, now_iso),
                        )
                    questions = list(DEFAULT_DAILY_PULSE_QUESTIONS)
                return {"questions": questions, "count": len(questions)}
        except psycopg.errors.UndefinedTable as exc:
            if attempt == 1 and _ensure_table_if_needed("daily_pulse_questions"):
                _log_db_event("db.query_retry_after_ensure", table="daily_pulse_questions")
                continue
            raise HTTPException(
                status_code=500,
                detail=(
                    "daily_pulse_questions table is missing and auto-create failed. "
                    f"Error: {exc}"
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read daily pulse questions: {exc}")

    raise HTTPException(status_code=500, detail="Failed to read daily pulse questions")


@app.put("/api/v1/daily-pulse/questions")
def update_daily_pulse_questions(
    request: DailyPulseUpdateRequest,
    raw_request: Request,
) -> dict[str, Any]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]

    normalized = [
        str(question).strip()
        for question in request.questions
        if isinstance(question, str) and str(question).strip()
    ]
    deduped = list(dict.fromkeys(normalized))

    if not deduped:
        raise HTTPException(status_code=400, detail="At least one question is required")

    for attempt in (1, 2):
        try:
            now_iso = datetime.now(UTC).isoformat()
            pool = _ensure_db_pool()
            with pool.connection() as conn:
                conn.execute("DELETE FROM daily_pulse_questions WHERE user_id=%s", (current_user_id,))
                for index, question in enumerate(deduped):
                    conn.execute(
                        """
                        INSERT INTO daily_pulse_questions (
                            user_id,
                            question,
                            order_index,
                            created_at,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (current_user_id, question, index, now_iso, now_iso),
                    )
            return {"questions": deduped, "count": len(deduped)}
        except psycopg.errors.UndefinedTable as exc:
            if attempt == 1 and _ensure_table_if_needed("daily_pulse_questions"):
                _log_db_event("db.query_retry_after_ensure", table="daily_pulse_questions")
                continue
            raise HTTPException(
                status_code=500,
                detail=(
                    "daily_pulse_questions table is missing and auto-create failed. "
                    f"Error: {exc}"
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to update daily pulse questions: {exc}")

    raise HTTPException(status_code=500, detail="Failed to update daily pulse questions")


def _run_chat_request(request: ChatRequest, user_id: str) -> ChatResponse:
    try:
        _register_thread_if_missing(request.thread_id, user_id)
        _set_thread_title_if_missing(request.thread_id, user_id, request.question)
        chatbot_instance = _get_chatbot()
        result = chatbot_instance.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config={"configurable": {"thread_id": _checkpoint_thread_id(request.thread_id, user_id)}},
        )
    except Exception as exc:
        error_text = str(exc)
        if "api_key" in error_text.lower() or "openai_api_key" in error_text.lower():
            raise HTTPException(
                status_code=500,
                detail=(
                    "Missing OPENAI_API_KEY. Set the environment variable and restart the backend."
                ),
            )
        if "insufficient_quota" in error_text.lower() or "error code: 429" in error_text.lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    "OpenAI quota exceeded. Update billing/quota or switch to a key with available credits."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {exc}")

    assistant_text: str = ""
    sql_result: Optional[dict[str, Any]] = None
    visualization_code: Optional[str] = None

    for msg in result.get("messages", []):
        role = getattr(msg, "type", None)
        if role not in {"ai", "AIMessage"}:
            continue

        additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
        msg_type = additional_kwargs.get("type")

        if msg_type == "sql_result":
            raw_data = additional_kwargs.get("data")
            if isinstance(raw_data, dict):
                sql_result = raw_data
            continue

        if msg_type == "visualization":
            raw_code = additional_kwargs.get("code")
            if isinstance(raw_code, str):
                visualization_code = raw_code
            continue

    # Mirror Streamlit behavior: render the latest assistant narrative message,
    # not every intermediate AI text block.
    for msg in reversed(result.get("messages", [])):
        role = getattr(msg, "type", None)
        if role not in {"ai", "AIMessage"}:
            continue

        additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
        if additional_kwargs:
            continue

        content = _extract_text_from_content(getattr(msg, "content", ""))
        if content.strip():
            assistant_text = content.strip()
            break

    assistant_text = assistant_text or "Completed"
    parsed_sections = _parse_agent_output(assistant_text)
    visualization_figure = _build_plotly_figure_json(visualization_code, sql_result)
    if visualization_figure is None:
        visualization_figure = _build_heuristic_plotly_figure_json(sql_result)

    relevant_questions: Optional[list[str]] = None
    if parsed_sections["relevant_questions"]:
        relevant_questions = [
            line.strip().lstrip("-").strip()
            for line in parsed_sections["relevant_questions"].splitlines()
            if line.strip()
        ]

    return ChatResponse(
        thread_id=request.thread_id,
        assistant_text=assistant_text,
        sql_query=parsed_sections["sql_query"],
        result_summary=parsed_sections["result_summary"],
        relevant_questions=relevant_questions,
        sql_result=sql_result,
        visualization_code=visualization_code,
        visualization_spec=parsed_sections["visualization_spec"],
        visualization_meta=None,
        visualization_figure=visualization_figure,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    current_user = _get_request_user(raw_request)
    return _run_chat_request(request, current_user["user_id"])


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=True, default=str)}\n\n"


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request) -> StreamingResponse:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]

    async def event_generator():
        final_summary_text: str = ""
        final_sql_query: Optional[str] = None
        final_sql_result: Optional[dict[str, Any]] = None
        final_visualization_code: Optional[str] = None
        final_visualization_spec: Optional[str] = None
        final_visualization_figure: Optional[dict[str, Any]] = None
        relevant_questions: list[str] = []

        async def _client_disconnected() -> bool:
            try:
                return await raw_request.is_disconnected()
            except Exception:
                return False

        def _persist_stream_messages(always: bool = False) -> None:
            has_materialized_content = any(
                [
                    bool(final_summary_text.strip()),
                    bool(final_sql_query),
                    isinstance(final_sql_result, dict),
                    bool(final_visualization_code),
                    isinstance(final_visualization_figure, dict),
                    bool(relevant_questions),
                ]
            )

            if not always and not has_materialized_content:
                return

            cached_messages = _load_cached_messages(request.thread_id, current_user_id)
            user_message = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"type": "text", "text": request.question}],
            }
            assistant_parts: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": final_summary_text or "Completed",
                }
            ]
            if final_summary_text:
                assistant_parts.append({"type": "data-resultSummary", "data": final_summary_text})
            if final_sql_query:
                assistant_parts.append({"type": "data-sqlQuery", "data": final_sql_query})
            if isinstance(final_sql_result, dict):
                assistant_parts.append({"type": "data-sqlResult", "data": final_sql_result})
                columns = final_sql_result.get("columns")
                if isinstance(columns, list) and columns:
                    assistant_parts.append({"type": "data-sqlColumns", "data": columns})
                rows = final_sql_result.get("data")
                if isinstance(rows, list):
                    assistant_parts.append({"type": "data-sqlRowCount", "data": len(rows)})
            if final_visualization_code:
                assistant_parts.append({"type": "data-visualizationCode", "data": final_visualization_code})
            if isinstance(final_visualization_figure, dict):
                assistant_parts.append({"type": "data-visualizationFigure", "data": final_visualization_figure})
            if relevant_questions:
                assistant_parts.append({"type": "data-relevantQuestions", "data": relevant_questions})

            assistant_message = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "parts": assistant_parts,
            }

            cached_messages.extend([user_message, assistant_message])
            _save_cached_messages(request.thread_id, current_user_id, cached_messages)

        try:
            yield _sse_event(
                "status",
                {"key": "analyzing", "label": "Analyzing", "state": "active"},
            )

            if await _client_disconnected():
                return

            _register_thread_if_missing(request.thread_id, current_user_id)
            _set_thread_title_if_missing(request.thread_id, current_user_id, request.question)

            seed_message = HumanMessage(content=request.question)
            sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions = (
                await asyncio.to_thread(_build_rag_examples_for_question, request.question)
            )

            yield _sse_event(
                "status",
                {"key": "analyzing", "label": "Analyzing", "state": "completed"},
            )
            yield _sse_event(
                "status",
                {"key": "analyzing_data", "label": "Analyzing data", "state": "active"},
            )

            stream_graph = _get_stream_subgraph()
            config = {
                "configurable": {
                    "thread_id": _checkpoint_thread_id(request.thread_id, current_user_id)
                }
            }

            initial_state: dict[str, Any] = {
                "question": request.question,
                "messages": [seed_message],
                "run_id": datetime.now(UTC).isoformat() + "Z",
                "last_output": "",
                "query_decomposer_output": None,
                "sql_generator_output": None,
                "sql_reviewer_output": None,
                "human_reviewer_output": None,
                "active_review": None,
                "query_decomposer_rag_examples_text": query_decomposer_rag_examples_text,
                "sql_generator_rag_examples_text": sql_generator_rag_examples_text,
                "result_summary": None,
                "sql_executor_output": None,
                "visualization_code": None,
                "visualization_spec": None,
                "trace": [],
            }

            state_accumulator: dict[str, Any] = dict(initial_state)
            summary_emitted = False
            sql_emitted = False
            results_emitted = False
            chart_status_active_emitted = False
            chart_status_completed_emitted = False
            last_chart_signature: str | None = None

            async for update in stream_graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                if await _client_disconnected():
                    return

                if not isinstance(update, dict):
                    continue

                for node_name, node_delta in update.items():
                    if not isinstance(node_delta, dict):
                        continue

                    state_accumulator.update(node_delta)

                    if node_name == "query_decomposer":
                        yield _sse_event(
                            "status",
                            {"key": "analyzing_data", "label": "Analyzing data", "state": "completed"},
                        )
                        yield _sse_event(
                            "status",
                            {"key": "generating_sql", "label": "Generating SQL", "state": "active"},
                        )
                        continue

                    if node_name == "sql_generator":
                        yield _sse_event(
                            "status",
                            {"key": "generating_sql", "label": "Generating SQL", "state": "completed"},
                        )
                        yield _sse_event(
                            "status",
                            {"key": "fetching_results", "label": "Fetching Results", "state": "active"},
                        )
                        continue

                    if node_name == "sql_executor":
                        yield _sse_event(
                            "status",
                            {"key": "fetching_results", "label": "Fetching Results", "state": "completed"},
                        )
                        yield _sse_event(
                            "status",
                            {"key": "rendering_summary", "label": "Rendering Summary", "state": "active"},
                        )
                        continue

                    if node_name == "summarizer_node" and not summary_emitted:
                        summary_text = str(state_accumulator.get("result_summary") or "").strip()
                        if summary_text:
                            final_summary_text = summary_text
                            for token in re.findall(r"\S+\s*", summary_text):
                                if await _client_disconnected():
                                    return
                                yield _sse_event("summary_token", {"delta": token})
                                await asyncio.sleep(0.01)

                            yield _sse_event("summary_done", {"summary": summary_text})
                            yield _sse_event(
                                "status",
                                {
                                    "key": "rendering_summary",
                                    "label": "Rendering Summary",
                                    "state": "completed",
                                },
                            )
                            summary_emitted = True

                        if summary_emitted:
                            yield _sse_event(
                                "status",
                                {
                                    "key": "preparing_result_table",
                                    "label": "Preparing result table",
                                    "state": "active",
                                },
                            )

                        sql_query = state_accumulator.get("sql_generator_output")
                        if isinstance(sql_query, str) and sql_query.strip() and not sql_emitted:
                            final_sql_query = sql_query
                            yield _sse_event("sql_ready", {"sql_query": sql_query})
                            sql_emitted = True

                        sql_result = state_accumulator.get("sql_executor_output")
                        if isinstance(sql_result, dict) and not results_emitted:
                            final_sql_result = sql_result
                            yield _sse_event("results_ready", {"sql_result": sql_result})
                            results_emitted = True
                            yield _sse_event(
                                "status",
                                {
                                    "key": "preparing_result_table",
                                    "label": "Preparing result table",
                                    "state": "completed",
                                },
                            )

                        if not chart_status_active_emitted:
                            yield _sse_event(
                                "status",
                                {
                                    "key": "generating_visualization",
                                    "label": "Building visualization",
                                    "state": "active",
                                },
                            )
                            chart_status_active_emitted = True
                        continue

                    if node_name == "visualization_node":
                        visualization_code = state_accumulator.get("visualization_code")
                        sql_result = state_accumulator.get("sql_executor_output")

                        visualization_figure = None
                        if isinstance(sql_result, dict) and isinstance(visualization_code, str):
                            visualization_figure = await asyncio.to_thread(
                                _build_plotly_figure_json,
                                visualization_code,
                                sql_result,
                            )
                        if visualization_figure is None and isinstance(sql_result, dict):
                            visualization_figure = await asyncio.to_thread(
                                _build_heuristic_plotly_figure_json,
                                sql_result,
                            )

                        if any(
                            [
                                isinstance(visualization_code, str) and visualization_code.strip(),
                                isinstance(visualization_figure, dict),
                            ]
                        ):
                            chart_payload = {
                                "visualization_code": visualization_code,
                                "visualization_figure": visualization_figure,
                                "visualization_meta": None,
                            }
                            chart_signature = json.dumps(chart_payload, sort_keys=True, default=str)
                            if chart_signature == last_chart_signature:
                                continue

                            yield _sse_event(
                                "chart_ready",
                                chart_payload,
                            )
                            last_chart_signature = chart_signature
                            final_visualization_code = (
                                visualization_code if isinstance(visualization_code, str) else None
                            )
                            final_visualization_figure = (
                                visualization_figure if isinstance(visualization_figure, dict) else None
                            )

                        if (
                            chart_status_active_emitted
                            and not chart_status_completed_emitted
                            and node_name == "visualization_node"
                        ):
                            yield _sse_event(
                                "status",
                                {
                                    "key": "generating_visualization",
                                    "label": "Building visualization",
                                    "state": "completed",
                                },
                            )
                            chart_status_completed_emitted = True

            if not summary_emitted:
                fallback_summary = str(state_accumulator.get("result_summary") or "Completed").strip()
                final_summary_text = fallback_summary
                yield _sse_event("summary_done", {"summary": fallback_summary})

            if not sql_emitted:
                sql_query = state_accumulator.get("sql_generator_output")
                if isinstance(sql_query, str) and sql_query.strip():
                    final_sql_query = sql_query
                    yield _sse_event("sql_ready", {"sql_query": sql_query})

            if not results_emitted:
                sql_result = state_accumulator.get("sql_executor_output")
                if isinstance(sql_result, dict):
                    final_sql_result = sql_result
                    yield _sse_event("results_ready", {"sql_result": sql_result})

            if last_chart_signature is None:
                visualization_code = state_accumulator.get("visualization_code")
                sql_result = state_accumulator.get("sql_executor_output")
                visualization_figure = None
                if isinstance(sql_result, dict) and isinstance(visualization_code, str):
                    visualization_figure = await asyncio.to_thread(
                        _build_plotly_figure_json,
                        visualization_code,
                        sql_result,
                    )
                if visualization_figure is None and isinstance(sql_result, dict):
                    visualization_figure = await asyncio.to_thread(
                        _build_heuristic_plotly_figure_json,
                        sql_result,
                    )

                if any(
                    [
                        isinstance(visualization_code, str) and visualization_code.strip(),
                        isinstance(visualization_figure, dict),
                    ]
                ):
                    chart_payload = {
                        "visualization_code": visualization_code,
                        "visualization_figure": visualization_figure,
                        "visualization_meta": None,
                    }
                    yield _sse_event(
                        "chart_ready",
                        chart_payload,
                    )
                    last_chart_signature = json.dumps(chart_payload, sort_keys=True, default=str)
                    final_visualization_code = (
                        visualization_code if isinstance(visualization_code, str) else None
                    )
                    final_visualization_figure = (
                        visualization_figure if isinstance(visualization_figure, dict) else None
                    )

            if chart_status_active_emitted and not chart_status_completed_emitted:
                yield _sse_event(
                    "status",
                    {
                        "key": "generating_visualization",
                        "label": "Building visualization",
                        "state": "completed",
                    },
                )

            if relevant_questions:
                yield _sse_event(
                    "related_questions_ready",
                    {"relevant_questions": relevant_questions},
                )

            # Persist stream conversation so refresh/history works even when
            # this path does not write wrapper-checkpoint messages.

            _persist_stream_messages(always=True)

            if await _client_disconnected():
                return

            yield _sse_event("complete", {"thread_id": request.thread_id})
        except asyncio.CancelledError:
            try:
                _persist_stream_messages(always=False)
            except Exception:
                pass
            return
        except HTTPException as exc:
            yield _sse_event(
                "error",
                {
                    "status_code": exc.status_code,
                    "detail": str(exc.detail),
                },
            )
        except Exception as exc:
            # If we already produced visible assistant content before the
            # stream failed, persist a best-effort message for refresh/history.
            if final_summary_text.strip():
                _persist_stream_messages(always=False)

            yield _sse_event(
                "error",
                {
                    "status_code": 500,
                    "detail": f"Chat stream failed: {exc}",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/history")
def get_history(
    raw_request: Request,
    limit: int = 20,
    ending_before: Optional[str] = None,
    q: Optional[str] = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]

    normalized_q = q.strip() if q else None
    normalized_limit = max(1, limit)

    # Hot path optimization: first page without search should stay SQL-limited.
    if not ending_before and not normalized_q:
        rows = _db_fetchall(
            """
                            SELECT r.thread_id, r.created_at, r.title
            FROM thread_registry r
                            WHERE r.user_id=%s AND COALESCE(r.is_hidden, FALSE)=FALSE
            ORDER BY r.created_at DESC
            LIMIT %s
            """,
            (current_user_id, normalized_limit + 1),
        )

        page_rows = rows[:normalized_limit]
        has_more = len(rows) > normalized_limit
        chats = [
            {
                "id": str(row[0]),
                "createdAt": _coerce_thread_created_at(str(row[0]), row[1]).isoformat(),
                "title": (str(row[2] or "").strip() or "Current conversation"),
                "userId": current_user_id,
                "visibility": "private",
            }
            for row in page_rows
            if row and row[0]
        ]

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 1500:
            logger.warning("Slow /api/v1/history user=%s elapsed_ms=%.1f", current_user_id, elapsed_ms)
        return {"chats": chats, "hasMore": has_more}

    # Keep list endpoint DB-first; avoid expensive checkpoint hydration for titles.
    thread_ids = _list_visible_threads(None, current_user_id)
    thread_meta: dict[str, dict[str, Any]] = {}
    registry_rows = _db_fetchall(
        "SELECT thread_id, created_at, title FROM thread_registry WHERE user_id=%s",
        (current_user_id,),
    )
    for row in registry_rows:
        if not row or not row[0]:
            continue
        thread_meta[str(row[0])] = {
            "created_at": row[1],
            "title": row[2],
        }

    title_match_ids: set[str] = set()
    cache_match_ids: set[str] = set()
    if normalized_q and len(normalized_q) >= 2:
        like_pattern = f"%{normalized_q}%"

        title_match_rows = _db_fetchall(
            """
            SELECT thread_id
            FROM thread_registry
            WHERE user_id=%s
              AND COALESCE(is_hidden, FALSE)=FALSE
              AND COALESCE(title, '') ILIKE %s
            """,
            (current_user_id, like_pattern),
        )
        cache_match_rows = _db_fetchall(
            """
            SELECT thread_id
            FROM thread_message_cache
            WHERE user_id=%s
              AND messages_json ILIKE %s
            """,
            (current_user_id, like_pattern),
        )

        title_match_ids = {str(row[0]) for row in title_match_rows if row and row[0]}
        cache_match_ids = {str(row[0]) for row in cache_match_rows if row and row[0]}

        matched_thread_ids = title_match_ids | cache_match_ids
        thread_ids = [thread_id for thread_id in thread_ids if thread_id in matched_thread_ids]

    if normalized_q and len(normalized_q) >= 2:
        lowered_q = normalized_q.lower()

        def _relevance_score(thread_id: str) -> tuple[int, int]:
            title_text = str(thread_meta.get(thread_id, {}).get("title") or "").strip().lower()
            starts_with = 1 if title_text.startswith(lowered_q) else 0
            in_title = 1 if thread_id in title_match_ids else 0
            in_cache = 1 if thread_id in cache_match_ids else 0
            return (starts_with, (in_title * 2) + in_cache)

        sorted_threads = sorted(
            thread_ids,
            key=lambda thread_id: (
                _relevance_score(thread_id),
                _coerce_thread_created_at(thread_id, thread_meta.get(thread_id, {}).get("created_at")),
            ),
            reverse=True,
        )
    else:
        sorted_threads = sorted(
            thread_ids,
            key=lambda thread_id: (
                _coerce_thread_created_at(thread_id, thread_meta.get(thread_id, {}).get("created_at"))
            ),
            reverse=True,
        )

    start_index = 0
    if ending_before:
        try:
            start_index = sorted_threads.index(ending_before) + 1
        except ValueError:
            start_index = 0

    page = sorted_threads[start_index : start_index + normalized_limit]
    has_more = start_index + normalized_limit < len(sorted_threads)

    chats = [
        {
            "id": thread_id,
            "createdAt": (
                _coerce_thread_created_at(
                    thread_id,
                    thread_meta.get(thread_id, {}).get("created_at"),
                ).isoformat()
            ),
            "title": (str(thread_meta.get(thread_id, {}).get("title") or "").strip() or "Current conversation"),
            "userId": current_user_id,
            "visibility": "private",
        }
        for thread_id in page
    ]

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    if elapsed_ms > 1500:
        logger.warning("Slow /api/v1/history user=%s elapsed_ms=%.1f", current_user_id, elapsed_ms)

    return {"chats": chats, "hasMore": has_more}


@app.get("/api/v1/history/{thread_id}")
def get_history_messages(thread_id: str, raw_request: Request) -> dict[str, Any]:
    started_at = time.perf_counter()
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]
    if not _is_thread_visible(None, thread_id, current_user_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    # Fast path for UI refresh: prefer persisted stream cache when available.
    cached_messages = _load_cached_messages(thread_id, current_user_id)
    if cached_messages:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 1500:
            logger.warning(
                "Slow /api/v1/history/{thread_id} cache-hit user=%s thread=%s elapsed_ms=%.1f",
                current_user_id,
                thread_id,
                elapsed_ms,
            )
        return {"messages": cached_messages}

    chatbot_instance = _get_chatbot()
    serialized = _serialize_thread_messages_with_timeout(
        chatbot_instance,
        thread_id,
        current_user_id,
    )
    if serialized:
        _save_cached_messages(thread_id, current_user_id, serialized)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 1500:
            logger.warning(
                "Slow /api/v1/history/{thread_id} rebuilt user=%s thread=%s elapsed_ms=%.1f",
                current_user_id,
                thread_id,
                elapsed_ms,
            )
        return {"messages": serialized}

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    if elapsed_ms > 1500:
        logger.warning(
            "Slow /api/v1/history/{thread_id} empty user=%s thread=%s elapsed_ms=%.1f",
            current_user_id,
            thread_id,
            elapsed_ms,
        )

    return {"messages": []}


@app.delete("/api/v1/history")
def delete_all_history(raw_request: Request) -> dict[str, bool]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]

    _db_execute(
        """
        UPDATE thread_registry
        SET is_hidden=TRUE, hidden_at=%s
        WHERE user_id=%s AND COALESCE(is_hidden, FALSE)=FALSE
        """,
        (datetime.now(UTC).isoformat(), current_user_id),
    )
    return {"success": True}


@app.delete("/api/v1/history/{thread_id}")
def delete_history(thread_id: str, raw_request: Request) -> dict[str, bool]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]
    if not _is_thread_visible(None, thread_id, current_user_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    _db_execute(
        """
        UPDATE thread_registry
        SET is_hidden=TRUE, hidden_at=%s
        WHERE user_id=%s AND thread_id=%s
        """,
        (datetime.now(UTC).isoformat(), current_user_id, thread_id),
    )
    _db_execute(
        "DELETE FROM message_feedback WHERE user_id=%s AND thread_id=%s",
        (current_user_id, thread_id),
    )
    return {"success": True}


@app.get("/api/v1/votes")
def get_votes(thread_id: str, raw_request: Request) -> list[dict[str, Any]]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]
    if not _is_thread_visible(None, thread_id, current_user_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    rows = _db_fetchall(
        """
        SELECT thread_id, message_id, rating
        FROM message_feedback
        WHERE user_id=%s AND thread_id=%s
        ORDER BY id DESC
        """,
        (current_user_id, thread_id),
    )

    latest_by_message: dict[str, dict[str, Any]] = {}
    for current_thread, message_id, rating in rows:
        if message_id in latest_by_message:
            continue
        latest_by_message[message_id] = {
            "chatId": current_thread,
            "messageId": message_id,
            "isUpvoted": bool(rating == 1),
        }

    return list(latest_by_message.values())


@app.patch("/api/v1/votes")
def save_vote(request: VoteRequest, raw_request: Request) -> dict[str, Any]:
    current_user = _get_request_user(raw_request)
    current_user_id = current_user["user_id"]
    if not _is_thread_visible(None, request.thread_id, current_user_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    inserted = _save_feedback_if_missing(request, current_user_id)
    return {"success": True, "inserted": inserted}
