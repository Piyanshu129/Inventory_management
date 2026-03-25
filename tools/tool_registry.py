"""
Tool Registry — unified dispatcher for all agent tools.

Usage:
    from tools.tool_registry import execute_tool, TOOL_SCHEMAS
    result = execute_tool("check_stock", {"product_id": "P0001"})
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.check_stock import check_stock
from tools.update_inventory import update_inventory
from tools.generate_report import generate_report
from tools.run_sql_query import run_sql_query

# ── Tool schemas (for LLM prompting) ─────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "check_stock",
        "description": "Check current stock level and reorder status for a specific product by ID.",
        "parameters": {
            "product_id": {"type": "string", "description": "Unique product identifier (e.g. P0001)"}
        },
        "required": ["product_id"],
    },
    {
        "name": "update_inventory",
        "description": "Update (set) the stock quantity for a specific product.",
        "parameters": {
            "product_id": {"type": "string", "description": "Unique product identifier"},
            "quantity": {"type": "integer", "description": "New absolute stock quantity (>= 0)"},
        },
        "required": ["product_id", "quantity"],
    },
    {
        "name": "generate_report",
        "description": "Generate an inventory report. Use 'low_stock' for items needing reorder, 'full_inventory' for complete listing.",
        "parameters": {
            "type": {"type": "string", "enum": ["low_stock", "full_inventory"]},
        },
        "required": ["type"],
    },
    {
        "name": "run_sql_query",
        "description": "Execute a read-only SQL SELECT query against the products database.",
        "parameters": {
            "query": {"type": "string", "description": "Valid SQL SELECT statement"},
        },
        "required": ["query"],
    },
]

# ── Dispatcher ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {
    "check_stock": lambda inp: check_stock(inp["product_id"]),
    "update_inventory": lambda inp: update_inventory(inp["product_id"], int(inp["quantity"])),
    "generate_report": lambda inp: generate_report(inp["type"]),
    "run_sql_query": lambda inp: run_sql_query(inp["query"]),
}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Route a tool call to the appropriate implementation.

    Args:
        tool_name:  Name of the tool
        tool_input: Dict of validated parameters

    Returns:
        Tool result as a dict. Always includes 'tool_name' key.
    """
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return {
            "tool_name": tool_name,
            "error": f"Unknown tool '{tool_name}'. Available: {list(_REGISTRY.keys())}",
        }
    try:
        result = fn(tool_input)
        result["tool_name"] = tool_name
        return result
    except KeyError as e:
        return {
            "tool_name": tool_name,
            "error": f"Missing required parameter: {e}",
        }
    except Exception as e:
        return {
            "tool_name": tool_name,
            "error": str(e),
        }


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_REGISTRY.keys())
