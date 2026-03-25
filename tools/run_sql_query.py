"""
Tool: run_sql_query

Input:  {"query": "SQL SELECT string"}
Output: dict with query results
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import database as _db

# Allowed statement prefixes (read-only guard)
ALLOWED_PREFIXES = re.compile(
    r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE
)

# Block dangerous keywords even in subqueries
BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def run_sql_query(query: str) -> dict:
    """
    Execute a read-only SQL query against the products database.

    Args:
        query: SQL SELECT statement

    Returns:
        {
          "success": bool,
          "query": str,
          "row_count": int,
          "results": list[dict],
          "error": str (only on failure)
        }
    """
    q = query.strip()

    # Safety: only allow SELECT / WITH / EXPLAIN
    if not ALLOWED_PREFIXES.match(q):
        return {
            "success": False,
            "query": q,
            "error": "Only SELECT queries are permitted. Mutations are not allowed.",
        }

    if BLOCKED_KEYWORDS.search(q):
        return {
            "success": False,
            "query": q,
            "error": "Query contains a disallowed keyword (DML/DDL).",
        }

    try:
        results = _db.execute_query(q)
        return {
            "success": True,
            "query": q,
            "row_count": len(results),
            "results": results,
        }
    except Exception as exc:
        return {
            "success": False,
            "query": q,
            "error": str(exc),
        }
