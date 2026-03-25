"""
Generate tool-calling training dataset.

Produces: data/tool_calling_dataset.jsonl
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

random.seed(42)

PRODUCT_IDS = [f"P{i:04d}" for i in range(1, 201)]


# ── Templates per tool ────────────────────────────────────────────────────────

def check_stock_examples() -> list[dict]:
    templates = [
        "Check stock for product {pid}",
        "What is the stock level for {pid}?",
        "How many units of {pid} do we have?",
        "Is product {pid} in stock?",
        "Get stock info for {pid}",
        "Show inventory for item {pid}",
        "What's the current quantity for {pid}?",
        "Retrieve stock details for product {pid}",
        "Look up inventory for {pid}",
        "Check if we have {pid} available",
    ]
    examples = []
    for i, pid in enumerate(random.sample(PRODUCT_IDS, min(60, len(PRODUCT_IDS)))):
        nl = random.choice(templates).format(pid=pid)
        examples.append({
            "id": f"tool_{i:04d}",
            "instruction": "Output the correct tool call in strict JSON format.",
            "input": nl,
            "output": json.dumps({
                "tool": "check_stock",
                "input": {"product_id": pid},
            }),
        })
    return examples


def update_inventory_examples() -> list[dict]:
    templates = [
        "Update stock for {pid} to {qty} units",
        "Set quantity of {pid} to {qty}",
        "Change inventory level for {pid} to {qty}",
        "Adjust stock of product {pid} to {qty}",
        "Record {qty} units for {pid}",
    ]
    examples = []
    for i, pid in enumerate(random.sample(PRODUCT_IDS, min(60, len(PRODUCT_IDS)))):
        qty = random.randint(0, 500)
        nl = random.choice(templates).format(pid=pid, qty=qty)
        examples.append({
            "id": f"tool_{60 + i:04d}",
            "instruction": "Output the correct tool call in strict JSON format.",
            "input": nl,
            "output": json.dumps({
                "tool": "update_inventory",
                "input": {"product_id": pid, "quantity": qty},
            }),
        })
    return examples


def generate_report_examples() -> list[dict]:
    low_stock_nlqs = [
        "Generate a low stock report",
        "Show me which items need reordering",
        "Create a report for items below reorder level",
        "Give me the low inventory report",
        "Which products are running low? Give me a report",
        "Run a low stock analysis",
        "Produce a restocking report",
        "Generate report: items below reorder threshold",
        "What needs to be ordered? Generate report",
        "Show low stock summary",
    ]
    full_inventory_nlqs = [
        "Generate full inventory report",
        "Show me the complete inventory",
        "Create a full stock report",
        "Get a complete product listing",
        "Generate report for all products",
        "Show entire warehouse inventory",
        "Full inventory dump",
        "Give me the comprehensive inventory report",
        "List all items in a report",
        "Generate complete warehouse report",
    ]
    examples = []
    for i, nl in enumerate(low_stock_nlqs):
        examples.append({
            "id": f"tool_{120 + i:04d}",
            "instruction": "Output the correct tool call in strict JSON format.",
            "input": nl,
            "output": json.dumps({
                "tool": "generate_report",
                "input": {"type": "low_stock"},
            }),
        })
    for i, nl in enumerate(full_inventory_nlqs):
        examples.append({
            "id": f"tool_{130 + i:04d}",
            "instruction": "Output the correct tool call in strict JSON format.",
            "input": nl,
            "output": json.dumps({
                "tool": "generate_report",
                "input": {"type": "full_inventory"},
            }),
        })
    return examples


def run_sql_examples() -> list[dict]:
    sql_pairs = [
        ("Run this SQL: SELECT * FROM products WHERE stock < 10",
         "SELECT * FROM products WHERE stock < 10"),
        ("Execute query: SELECT name, price FROM products ORDER BY price DESC LIMIT 5",
         "SELECT name, price FROM products ORDER BY price DESC LIMIT 5"),
        ("Query the database: SELECT category, COUNT(*) FROM products GROUP BY category",
         "SELECT category, COUNT(*) FROM products GROUP BY category"),
        ("Run SQL query to get Electronics products",
         "SELECT * FROM products WHERE category = 'Electronics'"),
        ("Execute: SELECT AVG(price) FROM products",
         "SELECT AVG(price) FROM products"),
    ]
    examples = []
    for i, (nl, sql) in enumerate(sql_pairs):
        examples.append({
            "id": f"tool_{140 + i:04d}",
            "instruction": "Output the correct tool call in strict JSON format.",
            "input": nl,
            "output": json.dumps({
                "tool": "run_sql_query",
                "input": {"query": sql},
            }),
        })
    return examples


def main():
    out_path = Path(__file__).parent / "tool_calling_dataset.jsonl"
    examples = (
        check_stock_examples()
        + update_inventory_examples()
        + generate_report_examples()
        + run_sql_examples()
    )

    # Shuffle and trim
    random.shuffle(examples)
    examples = examples[: settings.num_tool_pairs]

    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Saved {len(examples)} tool-calling examples → {out_path}")


if __name__ == "__main__":
    main()
