"""
Intent Classifier — determines what action the agent should take.

Intents:
  - check_stock      → tool: check_stock
  - update_stock     → tool: update_inventory
  - generate_report  → tool: generate_report
  - semantic_search  → vector DB retrieval
  - sql_query        → text-to-SQL + run_sql_query
  - general          → plain LLM response
"""

from __future__ import annotations
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

INTENT_LABELS = [
    "check_stock",
    "update_stock",
    "generate_report",
    "semantic_search",
    "sql_query",
    "general",
]

# ── Rule-based fast-path (no LLM needed) ─────────────────────────────────────

_RULES: list[tuple[re.Pattern, str]] = [
    # update_stock — MUST come before check_stock (stock-for pattern overlaps)
    (re.compile(r"\b(update|set|change|adjust)\s+(stock|inventory|quantity)\b", re.I), "update_stock"),
    (re.compile(r"\b(set|record)\s+\d+\s+(units?|qty)\b", re.I), "update_stock"),
    # check_stock — product ID present without update verb
    (re.compile(r"\bcheck\s+stock\b.*\bP\d{4}\b|\bstock\s+(for|of)\b.*\bP\d{4}\b", re.I), "check_stock"),
    # generate_report
    (re.compile(r"\b(generate|create|produce|run|give\s+me)\s+(a\s+)?(low\s*stock|full\s+inventory|restocking|warehouse)\s+report\b", re.I), "generate_report"),
    (re.compile(r"\b(low\s*stock|restock)\s+report\b", re.I), "generate_report"),
    (re.compile(r"\bfull\s+inventory\s+report\b", re.I), "generate_report"),
    # semantic_search — descriptive / attribute-based queries
    (re.compile(r"\b(220V|110V|24VDC|hydraulic|pneumatic|ATEX|IP65|IP67|NPN|PLC|VFD|brushless|neoprene|PPE)\b", re.I), "semantic_search"),
    (re.compile(r"\b(heavy.?duty|high.?voltage|industrial.?grade|chemical.?resistant)\b", re.I), "semantic_search"),
    # sql_query
    (re.compile(r"\b(SELECT|WHERE|GROUP BY|ORDER BY|HAVING|JOIN)\b", re.I), "sql_query"),
    (re.compile(r"\b(run|execute|query|show)\s+(sql|query|database)\b", re.I), "sql_query"),
]

# Structured query keywords → sql_query
_SQL_KEYWORDS = re.compile(
    r"\b(all|list|show|how many|count|average|total|cheapest|most expensive|"
    r"below|above|under|over|low stock|reorder)\b",
    re.I,
)


def classify_intent(query: str, context: str = "") -> dict:
    """
    Classify the user query intent using rule-based matching first,
    then fall back to LLM if ambiguous.

    Returns:
        {
          "intent": str,
          "confidence": "high" | "low",
          "product_id": str | None,
          "report_type": str | None,
        }
    """
    # Try rule-based fast-path
    for pattern, intent in _RULES:
        if pattern.search(query):
            result = {"intent": intent, "confidence": "high", "product_id": None, "report_type": None}
            _enrich(result, query)
            logger.debug("Rule-matched intent: %s", intent)
            return result

    # Heuristic: structured list/count queries → sql_query
    if _SQL_KEYWORDS.search(query):
        result = {"intent": "sql_query", "confidence": "high", "product_id": None, "report_type": None}
        logger.debug("SQL keyword heuristic matched.")
        return result

    # Fall back to LLM classification
    result = _llm_classify(query, context)
    logger.debug("LLM-classified intent: %s", result)
    return result


def _enrich(result: dict, query: str) -> None:
    """Extract product_id and report_type from query string."""
    # Product ID
    pid_match = re.search(r"\bP(\d{4})\b", query, re.I)
    if pid_match:
        result["product_id"] = f"P{pid_match.group(1)}"

    # Report type
    if re.search(r"\blow.?stock\b", query, re.I):
        result["report_type"] = "low_stock"
    elif re.search(r"\bfull.?inventory\b", query, re.I):
        result["report_type"] = "full_inventory"

    # Quantity for update_stock
    qty_match = re.search(r"\b(\d+)\s*(units?|qty)?\b", query)
    if qty_match and result.get("intent") == "update_stock":
        result["quantity"] = int(qty_match.group(1))


def _llm_classify(query: str, context: str) -> dict:
    """Use LLM to classify intent when rules don't match."""
    from agent.llm_client import chat_completion

    # Build ONE system message (Qwen/llama.cpp Jinja requires system at position 0 only)
    system = (
        "You are an intent classifier for a warehouse inventory system.\n"
        "Classify the user query into exactly one of these intents:\n"
        "  check_stock, update_stock, generate_report, semantic_search, sql_query, general\n\n"
        "Database schema: products(product_id, name, category, stock, reorder_level, price, description)\n\n"
        "Respond ONLY with a JSON object (no markdown, no explanation):\n"
        '{"intent": "<label>", "product_id": "<P####> or null", '
        '"report_type": "low_stock|full_inventory|null", "quantity": <int or null>}'
    )
    if context:
        system += f"\n\nRecent conversation:\n{context}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    try:
        # Use json_mode=False — llama.cpp may not support response_format
        raw = chat_completion(messages, temperature=0.0, json_mode=False)
        # Strip any accidental markdown fences
        raw = raw.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "general")
        if intent not in INTENT_LABELS:
            intent = "general"
        return {
            "intent": intent,
            "confidence": "low",
            "product_id": parsed.get("product_id"),
            "report_type": parsed.get("report_type"),
            "quantity": parsed.get("quantity"),
        }
    except Exception as exc:
        logger.warning("LLM intent classification failed: %s", exc)
        return {"intent": "sql_query", "confidence": "low", "product_id": None, "report_type": None}
