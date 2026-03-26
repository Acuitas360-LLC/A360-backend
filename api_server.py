from __future__ import annotations

import os
import re
import sqlite3
import sys
import uuid
from importlib import import_module
from datetime import UTC, datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Support legacy absolute imports used across backend modules.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
# Legacy modules use relative file paths (e.g., payload_store3.json),
# so pin process cwd to backend directory for consistent resolution.
os.chdir(BACKEND_DIR)
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from chatbot7 import build_chatbot
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


class VoteRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    rating: int
    user_query: Optional[str] = None
    assistant_response: Optional[str] = None


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
            created_at TEXT
        )
        """
    )
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


def _get_chatbot() -> Any:
    global chatbot
    if chatbot is None:
        chatbot = build_chatbot(checkpointer=checkpointer)
    return chatbot


def _parse_agent_output(text: str) -> dict[str, Optional[str]]:
    sections = {
        "sql_query": None,
        "result_summary": None,
        "query_results": None,
        "visualization_code": None,
        "relevant_questions": None,
    }

    header_map = {
        "sql_query": "SQL Query Executed:",
        "result_summary": "Result Summary:",
        "query_results": "Query Results:",
        "visualization_code": "Visualization Code:",
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
        INSERT OR IGNORE INTO thread_registry (thread_id, created_at)
        VALUES (?, ?)
        """,
        (thread_id, datetime.now(UTC).isoformat()),
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
    state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type in {"human", "HumanMessage"}:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                title = content.strip().replace("\n", " ")
                return title[:max_len] + ("..." if len(title) > max_len else "")

    return "Current conversation"


def _list_visible_threads(chatbot_instance: Any) -> list[str]:
    hidden_rows = sqlite_conn.execute("SELECT thread_id FROM hidden_threads").fetchall()
    hidden = {row[0] for row in hidden_rows}

    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return [thread_id for thread_id in all_threads if thread_id not in hidden]


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


def _serialize_thread_messages(chatbot_instance: Any, thread_id: str) -> list[dict[str, Any]]:
    state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    serialized: list[dict[str, Any]] = []
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

        content = _extract_text_from_content(getattr(message, "content", ""))
        if not content.strip():
            continue

        serialized.append(
            {
                "id": str(uuid.uuid4()),
                "role": role,
                "parts": [{"type": "text", "text": content}],
            }
        )

    return serialized


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        _register_thread_if_missing(request.thread_id)
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
    visible_threads = set(_list_visible_threads(chatbot_instance))

    if thread_id not in visible_threads:
        return {"messages": []}

    return {"messages": _serialize_thread_messages(chatbot_instance, thread_id)}


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
