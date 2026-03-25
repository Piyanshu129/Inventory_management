"""
Tool: update_inventory

Input:  {"product_id": "string", "quantity": integer}
Output: dict with updated product info
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import database as _db


def update_inventory(product_id: str, quantity: int) -> dict:
    """
    Update stock level for a given product.

    Args:
        product_id: product identifier
        quantity:   new absolute stock quantity (must be >= 0)

    Returns result dict with previous and new stock values.
    """
    pid = product_id.strip().upper()

    if not isinstance(quantity, int) or quantity < 0:
        return {
            "success": False,
            "error": f"Invalid quantity '{quantity}'. Must be a non-negative integer.",
        }

    # Check product exists
    rows = _db.execute_query(
        "SELECT product_id, name, stock, reorder_level FROM products WHERE product_id = :pid",
        {"pid": pid},
    )
    if not rows:
        return {
            "success": False,
            "error": f"Product '{pid}' not found.",
        }

    old_stock = rows[0]["stock"]
    reorder = rows[0]["reorder_level"]

    rowcount = _db.execute_write(
        "UPDATE products SET stock = :qty WHERE product_id = :pid",
        {"qty": quantity, "pid": pid},
    )

    if rowcount == 0:
        return {"success": False, "error": "Update failed (no rows affected)."}

    # Also update ChromaDB metadata if collection exists
    try:
        from vector_db.embedder import get_collection
        col = get_collection()
        if col.count() > 0:
            col.update(
                ids=[pid],
                metadatas=[{
                    **{k: v for k, v in rows[0].items()},
                    "stock": quantity,
                }],
            )
    except Exception:
        pass  # ChromaDB sync is best-effort

    return {
        "success": True,
        "product_id": pid,
        "name": rows[0]["name"],
        "previous_stock": old_stock,
        "new_stock": quantity,
        "reorder_level": reorder,
        "is_low_stock": quantity <= reorder,
        "message": f"Updated stock for '{pid}' from {old_stock} to {quantity} units.",
    }
