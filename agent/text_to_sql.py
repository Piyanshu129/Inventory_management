"""
Text-to-SQL converter — translates natural language to SQL SELECT statements.

Strategy:
  1. Build schema-aware prompt
  2. Call LLM to generate SQL
  3. Validate output is a SELECT statement
  4. Return SQL (or raise on failure)
"""

from __future__ import annotations
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

SCHEMA = """
Table: products
Columns:
  - product_id TEXT   — Unique product identifier (e.g. P0001)
  - name       TEXT   — Product name
  - category   TEXT   — Category: Electronics, Machinery, Tools, Safety, Electrical
  - stock      INT    — Current stock quantity
  - reorder_level INT — Minimum stock before reorder
  - price      REAL   — Unit price in USD
  - description TEXT  — Full product description

Key query patterns:
  "low stock"    → WHERE stock <= reorder_level
  "out of stock" → WHERE stock = 0
  "cheap / affordable" → WHERE price < <threshold>
  "high value"   → ORDER BY price DESC
  "needs reorder" → WHERE stock <= reorder_level
"""

SYSTEM_PROMPT = f"""\
You are a SQL expert for a warehouse inventory database.
Convert the user's natural language query into a valid SQL SELECT statement.

{SCHEMA}

Rules:
- Return ONLY the SQL statement, no explanation, no markdown code fences.
- Always use lowercase SQL keywords.
- ONLY produce SELECT statements.
- Use exact column names from the schema above.
- For category filters, use exact values: Electronics, Machinery, Tools, Safety, Electrical.
"""

# Validate generated SQL
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_DANGER_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE)\b", re.IGNORECASE
)


def text_to_sql(query: str, context: str = "") -> str:
    """
    Convert a natural language query to SQL.

    Priority:
      1. Fine-tuned local model (finetune/checkpoints/text_to_sql/checkpoint-100)
      2. OpenRouter API (Qwen 72B) — fallback if local model unavailable

    Args:
        query:   User's natural language query
        context: Optional conversation context string

    Returns:
        Valid SQL SELECT string

    Raises:
        ValueError: If generated SQL is invalid or dangerous
    """
    # Build a single combined prompt for both local and API paths
    system_content = SYSTEM_PROMPT
    if context:
        system_content = SYSTEM_PROMPT.rstrip() + f"\n\nConversation context:\n{context}"

    # ── 1. Try local fine-tuned model ────────────────────────────────────────
    try:
        from agent.local_text_to_sql import is_available, generate_sql as _local_gen

        if is_available():
            # Build Qwen chat-template style prompt manually
            few_shot = (
                "<|im_start|>system\n" + system_content + "<|im_end|>\n"
                "<|im_start|>user\nShow all low stock items<|im_end|>\n"
                "<|im_start|>assistant\n"
                "SELECT * FROM products WHERE stock <= reorder_level ORDER BY stock ASC"
                "<|im_end|>\n"
                "<|im_start|>user\nCount products per category<|im_end|>\n"
                "<|im_start|>assistant\n"
                "SELECT category, COUNT(*) AS count FROM products GROUP BY category ORDER BY count DESC"
                "<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            raw = _local_gen(few_shot)
            sql = _clean_sql(raw)
            if _SELECT_RE.match(sql) and not _DANGER_RE.search(sql):
                logger.info("[local-model] Generated SQL: %s", sql)
                return sql
            logger.warning("[local-model] Output invalid, falling back to API: %r", sql)
    except Exception as exc:
        logger.warning("[local-model] Error: %s — falling back to API", exc)

    # ── 2. Fall back to OpenRouter API ───────────────────────────────────────
    from agent.llm_client import chat_completion

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.append({"role": "user", "content": "Show all low stock items"})
    messages.append({
        "role": "assistant",
        "content": "SELECT * FROM products WHERE stock <= reorder_level ORDER BY stock ASC",
    })
    messages.append({"role": "user", "content": "Count products per category"})
    messages.append({
        "role": "assistant",
        "content": "SELECT category, COUNT(*) AS count FROM products GROUP BY category ORDER BY count DESC",
    })
    messages.append({"role": "user", "content": query})

    raw = chat_completion(messages, temperature=0.0)
    sql = _clean_sql(raw)

    if not _SELECT_RE.match(sql):
        raise ValueError(f"Generated SQL is not a SELECT statement: {sql!r}")
    if _DANGER_RE.search(sql):
        raise ValueError(f"Generated SQL contains a dangerous keyword: {sql!r}")

    logger.info("[api] Generated SQL: %s", sql)
    return sql


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM output."""
    sql = raw.strip()
    # Remove ```sql ... ``` fences
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip("`").strip()
    # Take first statement only (stop at semicolon)
    sql = sql.split(";")[0].strip()
    return sql


def build_fallback_sql(intent_info: dict) -> str | None:
    """
    Build a simple SQL query from intent info without needing an LLM.
    Used as a last resort when LLM is unavailable.
    """
    intent = intent_info.get("intent", "")
    cat = intent_info.get("category")
    pid = intent_info.get("product_id")

    if intent == "check_stock" and pid:
        return f"SELECT * FROM products WHERE product_id = '{pid}'"
    if intent == "generate_report":
        rtype = intent_info.get("report_type", "low_stock")
        if rtype == "low_stock":
            return "SELECT * FROM products WHERE stock <= reorder_level ORDER BY stock ASC"
        return "SELECT * FROM products ORDER BY category, name"

    # Generic fallback
    parts = ["SELECT * FROM products WHERE 1=1"]
    if cat:
        parts.append(f"AND category = '{cat}'")
    parts.append("ORDER BY category, name LIMIT 20")
    return " ".join(parts)
