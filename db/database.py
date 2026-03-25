"""Database connection and query helpers using SQLAlchemy."""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from config import settings

logger = logging.getLogger(__name__)


def _get_engine() -> Engine:
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Apply SQLite pragmas for safety
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


_engine = _get_engine()


def init_db() -> None:
    """Initialize database schema from schema.sql."""
    schema_file = Path(__file__).parent / "schema.sql"
    ddl = schema_file.read_text()
    with _engine.begin() as conn:
        for statement in ddl.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("Database initialized at %s", settings.db_path)


def execute_query(sql: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Execute a SQL query and return results as a list of dicts."""
    with _engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        if result.returns_rows:
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]
        return []


def execute_write(sql: str, params: dict | None = None) -> int:
    """Execute a write query (INSERT/UPDATE/DELETE). Returns rowcount."""
    with _engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        return result.rowcount


def get_engine() -> Engine:
    return _engine
