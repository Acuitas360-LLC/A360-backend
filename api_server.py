from __future__ import annotations

import os
import re
import sqlite3
import sys
import uuid
import json
import asyncio
import csv
from importlib import import_module
from datetime import UTC, datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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

from chatbot7 import build_chatbot, build_rag_examples
from subgraph5 import build_graph as build_stream_graph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

SqliteSaver = None
try:
    sqlite_module = import_module("langgraph.checkpoint.sqlite")
    SqliteSaver = getattr(sqlite_module, "SqliteSaver", None)
except Exception:
    SqliteSaver = None


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


app = FastAPI(title="A360 Backend API", version="0.1.0")

SQLITE_DB_PATH = os.path.join(BACKEND_DIR, "chatbot.db")


def _init_feedback_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            thread_id TEXT PRIMARY KEY,
            hidden_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_registry (
            thread_id TEXT PRIMARY KEY,
            created_at TEXT,
            title TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_message_cache (
            thread_id TEXT PRIMARY KEY,
            messages_json TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )
    conn.commit()

    # Migrate old local DBs that were created before title caching was added.
    thread_registry_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(thread_registry)").fetchall()
        if len(row) > 1
    }
    if "title" not in thread_registry_columns:
        conn.execute("ALTER TABLE thread_registry ADD COLUMN title TEXT")
        conn.commit()


sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
_init_feedback_db(sqlite_conn)


def _build_checkpointer() -> Any:
    try:
        if SqliteSaver is None:
            return MemorySaver()
        return SqliteSaver(sqlite_conn)
    except Exception:
        # Fallback keeps development unblocked if sqlite saver is unavailable.
        return MemorySaver()


checkpointer = _build_checkpointer()
chatbot = None
stream_subgraph = None


def _get_chatbot() -> Any:
    global chatbot
    if chatbot is None:
        chatbot = build_chatbot(checkpointer=checkpointer)
    return chatbot


def _get_stream_subgraph() -> Any:
    global stream_subgraph
    if stream_subgraph is None:
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
    except Exception:
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


def _feedback_exists(thread_id: str, message_id: str) -> bool:
    cursor = sqlite_conn.execute(
        "SELECT 1 FROM message_feedback WHERE thread_id=? AND message_id=? LIMIT 1",
        (thread_id, message_id),
    )
    return cursor.fetchone() is not None


def _save_feedback_if_missing(request: VoteRequest) -> bool:
    if _feedback_exists(request.thread_id, request.message_id):
        return False

    sqlite_conn.execute(
        """
        INSERT INTO message_feedback
        (thread_id, message_id, user_query, assistant_response, rating, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            request.thread_id,
            request.message_id,
            request.user_query,
            request.assistant_response,
            request.rating,
            datetime.now(UTC).isoformat(),
        ),
    )
    sqlite_conn.commit()
    return True


def _extract_thread_timestamp(thread_id: str) -> datetime:
    try:
        ts = thread_id[:-1] if thread_id.endswith("T") else thread_id
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=UTC)


def _register_thread_if_missing(thread_id: str) -> None:
    sqlite_conn.execute(
        """
        INSERT OR IGNORE INTO thread_registry (thread_id, created_at, title)
        VALUES (?, ?, NULL)
        """,
        (thread_id, datetime.now(UTC).isoformat()),
    )
    sqlite_conn.commit()


def _set_thread_title_if_missing(thread_id: str, title: Optional[str], max_len: int = 80) -> None:
    if not title:
        return

    normalized = title.strip().replace("\n", " ")
    if not normalized:
        return

    truncated = normalized[:max_len] + ("..." if len(normalized) > max_len else "")

    sqlite_conn.execute(
        """
        UPDATE thread_registry
        SET title = ?
        WHERE thread_id = ? AND (title IS NULL OR TRIM(title) = '')
        """,
        (truncated, thread_id),
    )
    sqlite_conn.commit()


def _get_thread_created_at(thread_id: str) -> datetime:
    row = sqlite_conn.execute(
        "SELECT created_at FROM thread_registry WHERE thread_id=?",
        (thread_id,),
    ).fetchone()

    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            pass

    return _extract_thread_timestamp(thread_id)


def _thread_matches_search(chatbot_instance: Any, thread_id: str, query: str) -> bool:
    if not query:
        return True

    state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    lowered = query.lower()

    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and lowered in content.lower():
            return True

    return False


def _get_thread_title(chatbot_instance: Any, thread_id: str, max_len: int = 80) -> str:
    cached_row = sqlite_conn.execute(
        "SELECT title FROM thread_registry WHERE thread_id=?",
        (thread_id,),
    ).fetchone()
    if cached_row and isinstance(cached_row[0], str) and cached_row[0].strip():
        return cached_row[0].strip()

    try:
        state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])

        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type in {"human", "HumanMessage"}:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    title = content.strip().replace("\n", " ")
                    resolved = title[:max_len] + ("..." if len(title) > max_len else "")
                    _set_thread_title_if_missing(thread_id, resolved, max_len=max_len)
                    return resolved
    except Exception:
        pass

    # Fallback for stream-first threads before checkpoint message state is persisted.
    cached = _load_cached_messages(thread_id)
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
            _set_thread_title_if_missing(thread_id, resolved, max_len=max_len)
            return resolved

    return "Current conversation"


def _list_visible_threads(chatbot_instance: Any) -> list[str]:
    hidden_rows = sqlite_conn.execute("SELECT thread_id FROM hidden_threads").fetchall()
    hidden = {row[0] for row in hidden_rows}

    all_threads: set[str] = set()

    # Include persisted registry threads so first-turn failures are still visible
    # in history even if no checkpoint was written yet.
    registry_rows = sqlite_conn.execute("SELECT thread_id FROM thread_registry").fetchall()
    for row in registry_rows:
        if row and row[0]:
            all_threads.add(str(row[0]))

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return [thread_id for thread_id in all_threads if thread_id not in hidden]


def _is_thread_visible(chatbot_instance: Any, thread_id: str) -> bool:
    hidden_row = sqlite_conn.execute(
        "SELECT 1 FROM hidden_threads WHERE thread_id=?",
        (thread_id,),
    ).fetchone()
    if hidden_row:
        return False

    registry_row = sqlite_conn.execute(
        "SELECT 1 FROM thread_registry WHERE thread_id=?",
        (thread_id,),
    ).fetchone()
    if registry_row:
        return True

    try:
        state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return bool(messages)
    except Exception:
        return False


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


def _load_cached_messages(thread_id: str) -> list[dict[str, Any]]:
    row = sqlite_conn.execute(
        "SELECT messages_json FROM thread_message_cache WHERE thread_id=?",
        (thread_id,),
    ).fetchone()
    if not row or not row[0]:
        return []

    try:
        parsed = json.loads(row[0])
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []


def _save_cached_messages(thread_id: str, messages: list[dict[str, Any]]) -> None:
    payload = json.dumps(messages, ensure_ascii=True, default=str)
    sqlite_conn.execute(
        """
        INSERT INTO thread_message_cache (thread_id, messages_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id)
        DO UPDATE SET messages_json=excluded.messages_json, updated_at=excluded.updated_at
        """,
        (thread_id, payload, datetime.now(UTC).isoformat()),
    )
    sqlite_conn.commit()


def _serialize_thread_messages(chatbot_instance: Any, thread_id: str) -> list[dict[str, Any]]:
    state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/daily-pulse/questions")
def get_daily_pulse_questions() -> dict[str, Any]:
    faq_path = os.path.join(BACKEND_DIR, "FAQ.csv")
    if not os.path.exists(faq_path):
        raise HTTPException(status_code=404, detail="FAQ.csv not found")

    questions: list[str] = []
    try:
        with open(faq_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                return {"questions": [], "count": 0}

            field_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames if name}
            question_field = field_map.get("questions")
            if not question_field:
                raise HTTPException(
                    status_code=400,
                    detail="FAQ.csv is missing a 'Questions' column",
                )

            for row in reader:
                if not isinstance(row, dict):
                    continue
                value = str(row.get(question_field, "")).strip()
                if value:
                    questions.append(value)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read FAQ.csv: {exc}")

    return {"questions": questions, "count": len(questions)}


@app.put("/api/v1/daily-pulse/questions")
def update_daily_pulse_questions(request: DailyPulseUpdateRequest) -> dict[str, Any]:
    faq_path = os.path.join(BACKEND_DIR, "FAQ.csv")

    normalized = [
        str(question).strip()
        for question in request.questions
        if isinstance(question, str) and str(question).strip()
    ]
    deduped = list(dict.fromkeys(normalized))

    if not deduped:
        raise HTTPException(status_code=400, detail="At least one question is required")

    try:
        with open(faq_path, "w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["Questions"])
            writer.writeheader()
            for question in deduped:
                writer.writerow({"Questions": question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update FAQ.csv: {exc}")

    return {"questions": deduped, "count": len(deduped)}


def _run_chat_request(request: ChatRequest) -> ChatResponse:
    try:
        _register_thread_if_missing(request.thread_id)
        _set_thread_title_if_missing(request.thread_id, request.question)
        chatbot_instance = _get_chatbot()
        result = chatbot_instance.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config={"configurable": {"thread_id": request.thread_id}},
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
def chat(request: ChatRequest) -> ChatResponse:
    return _run_chat_request(request)


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request) -> StreamingResponse:
    async def event_generator():
        async def _client_disconnected() -> bool:
            try:
                return await raw_request.is_disconnected()
            except Exception:
                return False

        try:
            yield _sse_event(
                "status",
                {"key": "analyzing", "label": "Analyzing", "state": "active"},
            )
            yield _sse_event(
                "status",
                {"key": "generating_sql", "label": "Generating SQL", "state": "active"},
            )

            if await _client_disconnected():
                return

            _register_thread_if_missing(request.thread_id)
            _set_thread_title_if_missing(request.thread_id, request.question)

            seed_message = HumanMessage(content=request.question)
            sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions = (
                await asyncio.to_thread(build_rag_examples, request.question)
            )

            stream_graph = _get_stream_subgraph()
            config = {"configurable": {"thread_id": request.thread_id}}

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
            final_summary_text: str = ""
            final_sql_query: Optional[str] = None
            final_sql_result: Optional[dict[str, Any]] = None
            final_visualization_code: Optional[str] = None
            final_visualization_spec: Optional[str] = None
            final_visualization_figure: Optional[dict[str, Any]] = None

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
                            {"key": "analyzing", "label": "Analyzing", "state": "completed"},
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

                        if not chart_status_active_emitted:
                            yield _sse_event(
                                "status",
                                {
                                    "key": "generating_visualization",
                                    "label": "Generating Visualization",
                                    "state": "active",
                                },
                            )
                            chart_status_active_emitted = True
                        continue

                    if node_name in {"visualization_node", "visualization_spec_node"}:
                        visualization_code = state_accumulator.get("visualization_code")
                        visualization_spec = state_accumulator.get("visualization_spec")
                        sql_result = state_accumulator.get("sql_executor_output")

                        visualization_figure = None
                        if isinstance(sql_result, dict) and isinstance(visualization_code, str):
                            visualization_figure = _build_plotly_figure_json(
                                visualization_code,
                                sql_result,
                            )
                        if visualization_figure is None and isinstance(sql_result, dict):
                            visualization_figure = _build_heuristic_plotly_figure_json(sql_result)

                        if any(
                            [
                                isinstance(visualization_code, str) and visualization_code.strip(),
                                isinstance(visualization_spec, str) and visualization_spec.strip(),
                                isinstance(visualization_figure, dict),
                            ]
                        ):
                            chart_payload = {
                                "visualization_code": visualization_code,
                                "visualization_spec": visualization_spec,
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
                            final_visualization_spec = (
                                visualization_spec if isinstance(visualization_spec, str) else None
                            )
                            final_visualization_figure = (
                                visualization_figure if isinstance(visualization_figure, dict) else None
                            )

                        if (
                            chart_status_active_emitted
                            and not chart_status_completed_emitted
                            and node_name == "visualization_spec_node"
                        ):
                            yield _sse_event(
                                "status",
                                {
                                    "key": "generating_visualization",
                                    "label": "Generating Visualization",
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
                visualization_spec = state_accumulator.get("visualization_spec")
                sql_result = state_accumulator.get("sql_executor_output")
                visualization_figure = None
                if isinstance(sql_result, dict) and isinstance(visualization_code, str):
                    visualization_figure = _build_plotly_figure_json(visualization_code, sql_result)
                if visualization_figure is None and isinstance(sql_result, dict):
                    visualization_figure = _build_heuristic_plotly_figure_json(sql_result)

                if any(
                    [
                        isinstance(visualization_code, str) and visualization_code.strip(),
                        isinstance(visualization_spec, str) and visualization_spec.strip(),
                        isinstance(visualization_figure, dict),
                    ]
                ):
                    chart_payload = {
                        "visualization_code": visualization_code,
                        "visualization_spec": visualization_spec,
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
                    final_visualization_spec = (
                        visualization_spec if isinstance(visualization_spec, str) else None
                    )
                    final_visualization_figure = (
                        visualization_figure if isinstance(visualization_figure, dict) else None
                    )

            if chart_status_active_emitted and not chart_status_completed_emitted:
                yield _sse_event(
                    "status",
                    {
                        "key": "generating_visualization",
                        "label": "Generating Visualization",
                        "state": "completed",
                    },
                )

            if relevant_questions:
                yield _sse_event(
                    "related_questions_ready",
                    {"relevant_questions": relevant_questions},
                )

            if await _client_disconnected():
                return

            # Persist stream conversation so refresh/history works even when
            # this path does not write wrapper-checkpoint messages.
            cached_messages = _load_cached_messages(request.thread_id)
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
            if final_visualization_spec:
                assistant_parts.append({"type": "data-visualizationSpec", "data": final_visualization_spec})
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
            _save_cached_messages(request.thread_id, cached_messages)

            yield _sse_event("complete", {"thread_id": request.thread_id})
        except asyncio.CancelledError:
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
                cached_messages = _load_cached_messages(request.thread_id)
                user_message = {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"type": "text", "text": request.question}],
                }
                assistant_parts: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": final_summary_text,
                    },
                    {"type": "data-resultSummary", "data": final_summary_text},
                ]
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
                if final_visualization_spec:
                    assistant_parts.append({"type": "data-visualizationSpec", "data": final_visualization_spec})
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
                _save_cached_messages(request.thread_id, cached_messages)

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
def get_history(limit: int = 20, ending_before: Optional[str] = None, q: Optional[str] = None) -> dict[str, Any]:
    chatbot_instance = _get_chatbot()

    thread_ids = _list_visible_threads(chatbot_instance)

    normalized_q = q.strip() if q else None

    if normalized_q and len(normalized_q) >= 2:
        thread_ids = [
            thread_id
            for thread_id in thread_ids
            if _thread_matches_search(chatbot_instance, thread_id, normalized_q)
        ]

    sorted_threads = sorted(
        thread_ids,
        key=_get_thread_created_at,
        reverse=True,
    )

    start_index = 0
    if ending_before:
        try:
            start_index = sorted_threads.index(ending_before) + 1
        except ValueError:
            start_index = 0

    page = sorted_threads[start_index : start_index + max(1, limit)]
    has_more = start_index + max(1, limit) < len(sorted_threads)

    chats = [
        {
            "id": thread_id,
            "createdAt": _get_thread_created_at(thread_id).isoformat(),
            "title": _get_thread_title(chatbot_instance, thread_id),
            "userId": "local-user",
            "visibility": "private",
        }
        for thread_id in page
    ]

    return {"chats": chats, "hasMore": has_more}


@app.get("/api/v1/history/{thread_id}")
def get_history_messages(thread_id: str) -> dict[str, Any]:
    chatbot_instance = _get_chatbot()
    if not _is_thread_visible(chatbot_instance, thread_id):
        return {"messages": []}

    # Fast path for UI refresh: prefer persisted stream cache when available.
    cached_messages = _load_cached_messages(thread_id)
    if cached_messages:
        return {"messages": cached_messages}

    serialized = _serialize_thread_messages(chatbot_instance, thread_id)
    if serialized:
        return {"messages": serialized}

    return {"messages": []}


@app.delete("/api/v1/history")
def delete_all_history() -> dict[str, bool]:
    chatbot_instance = _get_chatbot()
    thread_ids = _list_visible_threads(chatbot_instance)

    for thread_id in thread_ids:
        sqlite_conn.execute(
            "INSERT OR IGNORE INTO hidden_threads (thread_id, hidden_at) VALUES (?, ?)",
            (thread_id, datetime.now(UTC).isoformat()),
        )

    sqlite_conn.commit()
    return {"success": True}


@app.delete("/api/v1/history/{thread_id}")
def delete_history(thread_id: str) -> dict[str, bool]:
    sqlite_conn.execute(
        "INSERT OR IGNORE INTO hidden_threads (thread_id, hidden_at) VALUES (?, ?)",
        (thread_id, datetime.now(UTC).isoformat()),
    )
    sqlite_conn.execute("DELETE FROM message_feedback WHERE thread_id=?", (thread_id,))
    sqlite_conn.commit()
    return {"success": True}


@app.get("/api/v1/votes")
def get_votes(thread_id: str) -> list[dict[str, Any]]:
    rows = sqlite_conn.execute(
        """
        SELECT thread_id, message_id, rating
        FROM message_feedback
        WHERE thread_id=?
        ORDER BY id DESC
        """,
        (thread_id,),
    ).fetchall()

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
def save_vote(request: VoteRequest) -> dict[str, Any]:
    inserted = _save_feedback_if_missing(request)
    return {"success": True, "inserted": inserted}
