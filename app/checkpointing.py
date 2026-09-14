"""Checkpointer factory (Step 13).

Lets a research run persist its `ResearchState` after every graph node and
resume from where it left off — a partially completed run doesn't need to
restart from zero after an interruption.

Kept as a single factory function so the backend is swappable: `"memory"`
for local development (state lost on process exit) and `"sqlite"` for a
run that survives process restarts. A production deployment could later
add a `"postgres"` branch here without touching any calling code — every
caller only ever sees the `checkpointer` object returned by this function,
never the backend-specific class directly.
"""

from __future__ import annotations

import sqlite3

from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


def build_checkpointer(settings: Settings | None = None):
    """Returns a compiled-graph-ready checkpointer per `settings.checkpoint_backend`.

    "memory": `MemorySaver` — fast, zero setup, lost when the process exits.
    "sqlite": `SqliteSaver` backed by `settings.checkpoint_db_path` — survives
        process restarts, suitable for local/single-node persistence.

    Note: `SqliteSaver.from_conn_string(...)` returns a context manager in
    this version of langgraph-checkpoint-sqlite, not a ready-to-use saver —
    callers that need the connection to stay open for the process lifetime
    should use `open_sqlite_checkpointer` instead, which handles that.
    """
    settings = settings or get_settings()

    if settings.checkpoint_backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("Using in-memory checkpointer (state lost on process exit).")
        return MemorySaver()

    if settings.checkpoint_backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info("Using SQLite checkpointer at %s", settings.checkpoint_db_path)
        conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        return SqliteSaver(conn)

    raise ValueError(f"Unknown checkpoint backend: {settings.checkpoint_backend!r}")
