"""
Tool: check_stock

Input:  {"product_id": "string"}
Output: dict with product info + reorder status
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import database as _db


def check_stock(product_id: str) -> dict:
    """
    Check stock level for a given product_id.

    Returns:
        {
          "product_id": ...,
          "name": ...,
          "category": ...,
          "stock": ...,
          "reorder_level": ...,
          "price": ...,
          "is_low_stock": bool,
          "status": "ok" | "low" | "out_of_stock" | "not_found"
        }
    """
    rows = _db.execute_query(
        "SELECT product_id, name, category, stock, reorder_level, price "
        "FROM products WHERE product_id = :pid",
        {"pid": product_id.strip().upper()},
    )

    if not rows:
        return {
            "product_id": product_id,
            "status": "not_found",
            "message": f"Product '{product_id}' not found in the database.",
        }

    row = rows[0]
    stock = row["stock"]
    reorder = row["reorder_level"]

    if stock == 0:
        status = "out_of_stock"
    elif stock <= reorder:
        status = "low"
    else:
        status = "ok"

    return {
        **row,
        "is_low_stock": stock <= reorder,
        "status": status,
    }
