"""
Tool: generate_report

Input:  {"type": "low_stock" | "full_inventory"}
Output: dict with report data
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import database as _db


def generate_report(report_type: str) -> dict:
    """
    Generate inventory reports.

    Args:
        report_type: "low_stock" or "full_inventory"

    Returns:
        {
          "report_type": ...,
          "generated_at": ISO timestamp,
          "total_items": int,
          "products": [...],
          "summary": {...}
        }
    """
    rtype = report_type.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    if rtype == "low_stock":
        products = _db.execute_query(
            """
            SELECT product_id, name, category, stock, reorder_level, price
            FROM products
            WHERE stock <= reorder_level
            ORDER BY (reorder_level - stock) DESC, stock ASC
            """
        )
        deficit = sum(p["reorder_level"] - p["stock"] for p in products)
        restock_cost = sum(
            max(0, p["reorder_level"] - p["stock"]) * p["price"]
            for p in products
        )
        summary = {
            "total_low_stock_items": len(products),
            "total_units_deficit": deficit,
            "estimated_restock_cost": round(restock_cost, 2),
            "categories_affected": list({p["category"] for p in products}),
        }

    elif rtype == "full_inventory":
        products = _db.execute_query(
            """
            SELECT product_id, name, category, stock, reorder_level, price
            FROM products
            ORDER BY category, name
            """
        )
        total_value = sum(p["stock"] * p["price"] for p in products)
        low_stock_count = sum(1 for p in products if p["stock"] <= p["reorder_level"])
        by_category: dict[str, int] = {}
        for p in products:
            by_category[p["category"]] = by_category.get(p["category"], 0) + 1

        summary = {
            "total_products": len(products),
            "total_inventory_value": round(total_value, 2),
            "low_stock_count": low_stock_count,
            "products_by_category": by_category,
        }
    else:
        return {
            "error": f"Unknown report type '{report_type}'. Use 'low_stock' or 'full_inventory'."
        }

    return {
        "report_type": rtype,
        "generated_at": now,
        "total_items": len(products),
        "products": products,
        "summary": summary,
    }
