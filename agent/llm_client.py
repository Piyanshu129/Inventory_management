"""
LLM Client — OpenAI-compatible wrapper with offline stub fallback.

Reads LLM_BASE_URL, LLM_MODEL, LLM_API_KEY from config/settings.
If the server is unavailable (connection error), falls back to a simple
regex-based stub so the agent can still run without an LLM.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/inventory-agent",
                "X-Title": "Inventory Management Agent",
            },
        )
    return _client


def reset_client() -> None:
    """Force a fresh client on next call (e.g. after config changes)."""
    global _client
    _client = None


def chat_completion(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Call the LLM and return the response text.

    Falls back to offline stub on connection errors.
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens or settings.llm_max_tokens

    kwargs: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        client = _get_client()
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM call failed (%s). Using offline stub.", exc)
        return _offline_stub(messages)


def _offline_stub(messages: list[dict]) -> str:
    """
    Very basic offline fallback. Handles the most common agent prompts
    so the system can function without a live LLM for testing.
    """
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "").lower()
            break

    # Intent detection stub
    if "check stock" in last_user or "stock for" in last_user:
        return json.dumps({"intent": "check_stock", "tool": "check_stock"})
    if "update" in last_user or "set stock" in last_user:
        return json.dumps({"intent": "update_stock", "tool": "update_inventory"})
    if "low stock report" in last_user or "restock report" in last_user:
        return json.dumps({"intent": "generate_report", "report_type": "low_stock"})
    if "full inventory" in last_user or "all products" in last_user:
        return json.dumps({"intent": "generate_report", "report_type": "full_inventory"})
    if "220v" in last_user or "power supply" in last_user or "hydraulic" in last_user:
        return json.dumps({"intent": "semantic_search"})

    return json.dumps({"intent": "sql_query"})
