"""
ReAct Agent — the core reasoning and action loop.

Implements Think → Act → Observe → Respond cycle.

Step-by-step:
  1. Classify intent
  2. Decide path: semantic search / SQL / tool call / general
  3. Execute action
  4. (Optionally) execute a follow-up tool call
  5. Synthesize and format the final answer using the LLM
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from agent.memory import ConversationMemory
from agent.intent_classifier import classify_intent
from agent.text_to_sql import text_to_sql, build_fallback_sql
from tools.tool_registry import execute_tool, TOOL_SCHEMAS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an intelligent warehouse inventory assistant.
You have access to a real-time product database. When a user asks about inventory,
stock levels, or products, you reason step-by-step and provide concise, accurate answers.

Rules:
- Always cite actual product names, IDs, and stock numbers from retrieved data.
- NEVER invent product data — only use information retrieved from tools or the database.
- If stock is below reorder_level, proactively mention it.
- Format numeric values clearly (e.g., "42 units in stock, reorder level: 30").
- Keep answers concise but complete.
"""


class InventoryAgent:
    """
    Production-grade inventory management agent.

    Usage:
        agent = InventoryAgent()
        response = agent.chat("Check stock for P0001")
    """

    def __init__(self):
        self.memory = ConversationMemory(system_prompt=SYSTEM_PROMPT)
        self._step_log: list[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_query: str) -> str:
        """
        Process a user query and return the agent response.

        Args:
            user_query: Natural language query from the user

        Returns:
            Final answer string
        """
        self._step_log = []
        self.memory.add_user(user_query)
        context = self.memory.get_context_string(last_n=6)

        # ── THINK ──────────────────────────────────────────────────────────────
        intent_info = classify_intent(user_query, context)
        intent = intent_info["intent"]
        self._log("think", f"Intent classified: {intent}", intent_info)

        # Try coreference resolution
        resolved_pid = self._resolve_product_id(user_query, intent_info)
        if resolved_pid:
            intent_info["product_id"] = resolved_pid
            self._log("think", f"Resolved product_id via coreference: {resolved_pid}")

        # ── ACT ────────────────────────────────────────────────────────────────
        observation = self._act(intent, intent_info, user_query, context)
        self._log("observe", "Action complete", observation)

        # ── RESPOND ────────────────────────────────────────────────────────────
        answer = self._synthesize(user_query, intent, observation)
        self.memory.add_assistant(answer)
        self._log("respond", answer)

        return answer

    def get_step_log(self) -> list[dict]:
        """Return the reasoning trace for the last query."""
        return self._step_log.copy()

    def reset(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        self._step_log = []

    # ── Internal: ACT ──────────────────────────────────────────────────────────

    def _act(self, intent: str, intent_info: dict, query: str, context: str) -> dict:
        if intent == "check_stock":
            return self._act_check_stock(intent_info, query, context)
        if intent == "update_stock":
            return self._act_update_stock(intent_info, query)
        if intent == "generate_report":
            return self._act_generate_report(intent_info)
        if intent == "semantic_search":
            return self._act_semantic_search(query)
        if intent == "sql_query":
            return self._act_sql_query(query, context, intent_info)
        # general / fallback
        return {"type": "general", "data": None}

    def _act_check_stock(self, intent_info: dict, query: str, context: str) -> dict:
        pid = intent_info.get("product_id")

        if not pid:
            # Try semantic search first to identify the product
            sem_results = self._semantic_search(query)
            if sem_results:
                pid = sem_results[0]["product_id"]
                self._log("act", f"Identified product via semantic search: {pid}")
            else:
                return {"type": "error", "data": "Could not identify product from query."}

        tool_result = execute_tool("check_stock", {"product_id": pid})
        self.memory.add_tool_result("check_stock", json.dumps(tool_result))
        self._log("act", f"check_stock({pid})", tool_result)
        return {"type": "check_stock", "data": tool_result}

    def _act_update_stock(self, intent_info: dict, query: str) -> dict:
        pid = intent_info.get("product_id")
        qty = intent_info.get("quantity")

        # Parse quantity from query if not extracted
        if qty is None:
            import re
            m = re.search(r"\b(\d+)\s*(?:units?|qty|pieces?|pcs?)?\b", query)
            if m:
                qty = int(m.group(1))

        if not pid or qty is None:
            return {
                "type": "error",
                "data": "Could not extract product_id or quantity. Please specify both (e.g., 'Update P0001 to 150 units').",
            }

        tool_result = execute_tool("update_inventory", {"product_id": pid, "quantity": qty})
        self.memory.add_tool_result("update_inventory", json.dumps(tool_result))
        self._log("act", f"update_inventory({pid}, {qty})", tool_result)
        return {"type": "update_stock", "data": tool_result}

    def _act_generate_report(self, intent_info: dict) -> dict:
        rtype = intent_info.get("report_type") or "low_stock"
        tool_result = execute_tool("generate_report", {"type": rtype})
        self.memory.add_tool_result("generate_report", json.dumps(tool_result, default=str))
        self._log("act", f"generate_report({rtype})", {"summary": tool_result.get("summary")})
        return {"type": "generate_report", "data": tool_result}

    def _act_semantic_search(self, query: str) -> dict:
        results = self._semantic_search(query)
        if not results:
            return {"type": "semantic_search", "data": [], "message": "No matching products found."}

        # Follow up with stock check for the top result
        top = results[0]
        stock_result = execute_tool("check_stock", {"product_id": top["product_id"]})
        self._log("act", f"Semantic top match: {top['name']} ({top['product_id']})")
        return {
            "type": "semantic_search",
            "data": results,
            "top_result_stock": stock_result,
        }

    def _act_sql_query(self, query: str, context: str, intent_info: dict) -> dict:
        try:
            sql = text_to_sql(query, context)
        except Exception as exc:
            logger.warning("text_to_sql failed: %s", exc)
            sql = build_fallback_sql(intent_info)
            if not sql:
                return {"type": "error", "data": str(exc)}

        tool_result = execute_tool("run_sql_query", {"query": sql})
        self.memory.add_tool_result("run_sql_query", json.dumps(tool_result, default=str))
        self._log("act", f"run_sql_query: {sql}", {"row_count": tool_result.get("row_count")})
        return {"type": "sql_query", "data": tool_result, "sql": sql}

    # ── Internal: RESPOND ──────────────────────────────────────────────────────

    def _synthesize(self, query: str, intent: str, observation: dict) -> str:
        """Generate the final natural language answer from the observation."""
        from agent.llm_client import chat_completion

        obs_type = observation.get("type", "general")

        # For general queries (no tool data), use memory history directly
        if obs_type == "general":
            msgs = self.memory.get_messages()  # already has single system at [0]
            answer = chat_completion(msgs)
            if answer.strip().startswith("{"):
                return "I'm a warehouse inventory assistant. I can help you check stock levels, update inventory, generate reports, or search for products. What would you like to know?"
            return answer

        # Merge SYSTEM_PROMPT + retrieved data into ONE system message
        # (Qwen/llama.cpp Jinja templates require a single system message at position 0)
        obs_summary = _format_observation(observation)
        combined_system = (
            SYSTEM_PROMPT.rstrip()
            + "\n\n---\nRetrieved data (use ONLY this, do not invent values):\n"
            + obs_summary
        )
        messages = [
            {"role": "system", "content": combined_system},
            {"role": "user",   "content": query},
        ]

        answer = chat_completion(messages)

        # If LLM is offline, build answer from observation directly
        if answer.strip().startswith("{"):
            return _offline_answer(obs_type, observation)

        return answer

    # ── Internal: helpers ──────────────────────────────────────────────────────

    def _semantic_search(self, query: str) -> list[dict]:
        try:
            from vector_db.retriever import semantic_search
            return semantic_search(query, top_k=settings.rag_top_k)
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    def _resolve_product_id(self, query: str, intent_info: dict) -> str | None:
        """Attempt coreference resolution if 'it/that/this' refers to a previous product."""
        pid = intent_info.get("product_id")
        if pid:
            return None  # Already have one

        # Get known product IDs from recent memory
        context = self.memory.get_context_string(last_n=6)
        import re
        pids_in_context = re.findall(r"P\d{4}", context)
        if not pids_in_context:
            return None

        resolved = self.memory.resolve_coreference(query, pids_in_context)
        return resolved

    def _log(self, step: str, message: str, data: dict | None = None) -> None:
        self._step_log.append({"step": step, "message": message, "data": data})
        logger.debug("[%s] %s — %s", step.upper(), message, data or "")


# ── Observation formatter ──────────────────────────────────────────────────────

def _format_observation(obs: dict) -> str:
    """Convert observation dict to a human-readable summary for LLM synthesis."""
    otype = obs.get("type")

    if otype == "check_stock":
        d = obs.get("data", {})
        if d.get("status") == "not_found":
            return d.get("message", "Product not found.")
        return (
            f"Product: {d.get('name')} (ID: {d.get('product_id')})\n"
            f"Category: {d.get('category')}\n"
            f"Stock: {d.get('stock')} units\n"
            f"Reorder Level: {d.get('reorder_level')} units\n"
            f"Price: ${d.get('price', 0):.2f}\n"
            f"Status: {d.get('status', 'unknown').upper()}"
        )

    if otype == "update_stock":
        d = obs.get("data", {})
        if not d.get("success"):
            return f"Update failed: {d.get('error')}"
        return (
            f"Updated: {d.get('name')} (ID: {d.get('product_id')})\n"
            f"Previous stock: {d.get('previous_stock')} → New stock: {d.get('new_stock')}\n"
            f"Reorder level: {d.get('reorder_level')}\n"
            f"{'⚠️ Still below reorder level!' if d.get('is_low_stock') else '✅ Stock is adequate.'}"
        )

    if otype == "generate_report":
        d = obs.get("data", {})
        summary = d.get("summary", {})
        products = d.get("products", [])
        lines = [f"Report Type: {d.get('report_type', '').upper()}",
                 f"Generated At: {d.get('generated_at', '')}",
                 f"Total Items: {d.get('total_items', 0)}",
                 f"Summary: {json.dumps(summary)}"]
        # Include first 10 products
        for p in products[:10]:
            lines.append(
                f"  - {p['name']} (ID: {p['product_id']}, Cat: {p['category']}, "
                f"Stock: {p['stock']}, Reorder: {p['reorder_level']}, ${p['price']:.2f})"
            )
        if len(products) > 10:
            lines.append(f"  ... and {len(products) - 10} more items.")
        return "\n".join(lines)

    if otype == "semantic_search":
        results = obs.get("data", [])
        top_stock = obs.get("top_result_stock", {})
        if not results:
            return obs.get("message", "No results found.")
        lines = ["Top matching products:"]
        for r in results[:5]:
            lines.append(
                f"  - {r['name']} (ID: {r['product_id']}, Cat: {r['category']}, "
                f"Score: {r['similarity_score']:.2f}, Stock: {r['stock']}, ${r['price']:.2f})"
            )
        if top_stock and top_stock.get("status") != "not_found":
            lines.append(
                f"\nTop result stock detail: {top_stock.get('name')} — "
                f"{top_stock.get('stock')} units, status: {top_stock.get('status', '').upper()}"
            )
        return "\n".join(lines)

    if otype == "sql_query":
        d = obs.get("data", {})
        sql = obs.get("sql", "")
        results = d.get("results", [])
        lines = [f"SQL: {sql}", f"Rows returned: {d.get('row_count', 0)}"]
        for row in results[:10]:
            lines.append(f"  {row}")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more rows.")
        return "\n".join(lines)

    if otype == "error":
        return f"Error: {obs.get('data', 'Unknown error')}"

    return str(obs)


def _offline_answer(obs_type: str, observation: dict) -> str:
    """
    Build a plain-text answer directly from the observation when the LLM is offline.
    Called when chat_completion() returns raw JSON (the offline stub response).
    """
    if obs_type == "check_stock":
        d = observation.get("data", {})
        if d.get("status") == "not_found":
            return f"Product {d.get('product_id', '')} was not found in the database."
        status_msg = {
            "ok": "✅ Stock is adequate.",
            "low": "⚠️ Stock is below the reorder level — restocking recommended.",
            "out_of_stock": "🚨 Out of stock!",
        }.get(d.get("status", ""), "")
        return (
            f"**{d.get('name')}** (ID: {d.get('product_id')})\n"
            f"- Category: {d.get('category')}\n"
            f"- Stock: **{d.get('stock')} units** (reorder at {d.get('reorder_level')})\n"
            f"- Price: ${d.get('price', 0):.2f}\n"
            f"- {status_msg}"
        )

    if obs_type == "update_stock":
        d = observation.get("data", {})
        if not d.get("success"):
            return f"❌ Update failed: {d.get('error')}"
        arrow = "⚠️ Still below reorder level." if d.get("is_low_stock") else "✅ Stock is now adequate."
        return (
            f"✅ Updated **{d.get('name')}** (ID: {d.get('product_id')})\n"
            f"- Previous stock: {d.get('previous_stock')} → New stock: **{d.get('new_stock')}**\n"
            f"- Reorder level: {d.get('reorder_level')}\n"
            f"- {arrow}"
        )

    if obs_type == "generate_report":
        d = observation.get("data", {})
        summary = d.get("summary", {})
        products = d.get("products", [])
        rtype = d.get("report_type", "").replace("_", " ").title()
        lines = [f"**{rtype} Report** — {d.get('total_items', 0)} items\n"]
        for k, v in summary.items():
            lines.append(f"- {k.replace('_', ' ').title()}: {v}")
        lines.append("")
        for p in products[:15]:
            flag = "🔴" if p["stock"] <= p["reorder_level"] else "🟢"
            lines.append(
                f"{flag} **{p['name']}** (ID: {p['product_id']}) — "
                f"Stock: {p['stock']}, Reorder: {p['reorder_level']}, ${p['price']:.2f}"
            )
        if len(products) > 15:
            lines.append(f"\n... and {len(products) - 15} more items.")
        return "\n".join(lines)

    if obs_type == "semantic_search":
        results = observation.get("data", [])
        top_stock = observation.get("top_result_stock", {})
        if not results:
            return "No matching products found for your query."
        lines = ["**Top matching products:**\n"]
        for r in results:
            flag = "⚠️" if r.get("stock", 0) <= r.get("reorder_level", 0) else "✅"
            lines.append(
                f"{flag} **{r['name']}** (ID: {r['product_id']}, {r['category']}) — "
                f"Stock: {r['stock']}, ${r['price']:.2f} (score: {r['similarity_score']:.0%})"
            )
        if top_stock and top_stock.get("status") != "not_found":
            s = top_stock
            status_icon = {"ok": "✅", "low": "⚠️", "out_of_stock": "🚨"}.get(s.get("status", ""), "")
            lines.append(
                f"\n{status_icon} **Best match detail:** {s.get('name')} — "
                f"{s.get('stock')} units in stock (reorder at {s.get('reorder_level')})"
            )
        return "\n".join(lines)

    if obs_type == "sql_query":
        d = observation.get("data", {})
        sql = observation.get("sql", "")
        results = d.get("results", [])
        row_count = d.get("row_count", 0)
        if not d.get("success"):
            return f"Query failed: {d.get('error')}"
        lines = [f"**Query results** — {row_count} row(s) found\n```sql\n{sql}\n```\n"]
        for row in results[:15]:
            name = row.get("name", "")
            pid = row.get("product_id", "")
            stock = row.get("stock", "")
            price = row.get("price", "")
            cat = row.get("category", "")
            lines.append(f"- **{name}** (ID: {pid}, {cat}) — Stock: {stock}, ${price}")
        if row_count > 15:
            lines.append(f"\n... and {row_count - 15} more rows.")
        return "\n".join(lines)

    if obs_type == "error":
        return f"❌ {observation.get('data', 'An error occurred.')}"

    return _format_observation(observation)
