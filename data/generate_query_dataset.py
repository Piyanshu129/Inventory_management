"""
Generate NL → SQL training pairs dataset.

Produces: data/nl_to_sql_dataset.jsonl
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

random.seed(42)

CATEGORIES = ["Electronics", "Machinery", "Tools", "Safety", "Electrical"]
PRICE_THRESHOLDS = [50, 100, 200, 500, 1000]
STOCK_VALUES = [5, 10, 20, 50, 100]


# ── Template pairs ────────────────────────────────────────────────────────────

TEMPLATES: list[tuple[str, str]] = [
    # Low stock
    ("Show all low stock items",
     "SELECT * FROM products WHERE stock < reorder_level"),
    ("Which products need restocking?",
     "SELECT * FROM products WHERE stock < reorder_level"),
    ("List items running low",
     "SELECT product_id, name, stock, reorder_level FROM products WHERE stock < reorder_level"),
    ("Get critical stock items",
     "SELECT * FROM products WHERE stock <= reorder_level ORDER BY stock ASC"),
    ("Show products where stock is below reorder level",
     "SELECT * FROM products WHERE stock < reorder_level"),

    # Category-specific
    *[
        (f"Show all {cat} items",
         f"SELECT * FROM products WHERE category = '{cat}'")
        for cat in CATEGORIES
    ],
    *[
        (f"List all products in the {cat} category",
         f"SELECT * FROM products WHERE category = '{cat}'")
        for cat in CATEGORIES
    ],
    *[
        (f"Which {cat} products are low in stock?",
         f"SELECT * FROM products WHERE category = '{cat}' AND stock < reorder_level")
        for cat in CATEGORIES
    ],

    # Price-based
    *[
        (f"Show cheap items under ${t}",
         f"SELECT * FROM products WHERE price < {t} ORDER BY price ASC")
        for t in PRICE_THRESHOLDS
    ],
    *[
        (f"List expensive products over ${t}",
         f"SELECT * FROM products WHERE price > {t} ORDER BY price DESC")
        for t in PRICE_THRESHOLDS
    ],
    ("Find the 10 most expensive products",
     "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    ("Find the 10 cheapest products",
     "SELECT * FROM products ORDER BY price ASC LIMIT 10"),
    ("What is the average price of all products?",
     "SELECT AVG(price) AS avg_price FROM products"),

    # Stock level queries
    *[
        (f"Show items with stock below {s}",
         f"SELECT * FROM products WHERE stock < {s}")
        for s in STOCK_VALUES
    ],
    *[
        (f"List products with more than {s} units in stock",
         f"SELECT * FROM products WHERE stock > {s}")
        for s in STOCK_VALUES
    ],
    ("Show all out-of-stock items",
     "SELECT * FROM products WHERE stock = 0"),
    ("Which products have zero inventory?",
     "SELECT product_id, name, category FROM products WHERE stock = 0"),

    # Aggregation
    ("How many products are in each category?",
     "SELECT category, COUNT(*) AS count FROM products GROUP BY category"),
    ("What is the total inventory value?",
     "SELECT SUM(stock * price) AS total_value FROM products"),
    ("Show average stock per category",
     "SELECT category, AVG(stock) AS avg_stock FROM products GROUP BY category"),
    ("Count low stock items per category",
     "SELECT category, COUNT(*) AS low_stock_count FROM products WHERE stock < reorder_level GROUP BY category"),
    ("What is the total number of products?",
     "SELECT COUNT(*) AS total_products FROM products"),

    # Ordering
    ("Show all products sorted by stock ascending",
     "SELECT * FROM products ORDER BY stock ASC"),
    ("List products sorted by price descending",
     "SELECT * FROM products ORDER BY price DESC"),
    ("Show top 5 items by stock",
     "SELECT * FROM products ORDER BY stock DESC LIMIT 5"),
    ("List bottom 5 items by price",
     "SELECT * FROM products ORDER BY price ASC LIMIT 5"),

    # Combined filters
    *[
        (f"Show low stock {cat} items under ${t}",
         f"SELECT * FROM products WHERE category = '{cat}' AND stock < reorder_level AND price < {t}")
        for cat in CATEGORIES
        for t in [100, 500, 1000]
    ],

    # Full inventory
    ("Show full inventory",
     "SELECT * FROM products"),
    ("List all products",
     "SELECT product_id, name, category, stock, reorder_level, price FROM products"),
    ("Get all product details",
     "SELECT * FROM products ORDER BY category, name"),

    # Named product lookup
    ("Find product with ID P0001",
     "SELECT * FROM products WHERE product_id = 'P0001'"),
    ("Look up details for product P0050",
     "SELECT * FROM products WHERE product_id = 'P0050'"),

    # Reorder level
    ("Show products where reorder level is above 30",
     "SELECT * FROM products WHERE reorder_level > 30"),
    ("Which products have a high reorder threshold?",
     "SELECT * FROM products WHERE reorder_level > 25 ORDER BY reorder_level DESC"),
]


def augment(nl: str, sql: str, index: int) -> dict:
    """Wrap a pair in training format."""
    return {
        "id": f"nl_sql_{index:04d}",
        "instruction": (
            "You are a SQL expert for a warehouse inventory database. "
            "Convert the following natural language query into a valid SQL SELECT statement. "
            "Table: products(product_id TEXT, name TEXT, category TEXT, stock INT, reorder_level INT, price FLOAT, description TEXT)"
        ),
        "input": nl,
        "output": sql,
    }


def generate_nl_sql_pairs(n: int = 500) -> list[dict]:
    pairs = []
    base = [augment(nl, sql, i) for i, (nl, sql) in enumerate(TEMPLATES)]
    pairs.extend(base)

    # Fill remaining by sampling with slight NL variations
    variations = [
        ("Show me ", ""), ("Can you list ", ""), ("What are the ", ""),
        ("Get all ", ""), ("Find all ", ""), ("Display ", ""),
    ]
    idx = len(base)
    while len(pairs) < n:
        nl, sql = random.choice(TEMPLATES)
        prefix, _ = random.choice(variations)
        new_nl = prefix + nl[0].lower() + nl[1:]
        pairs.append(augment(new_nl, sql, idx))
        idx += 1

    return pairs[:n]


def main():
    out_path = Path(__file__).parent / "nl_to_sql_dataset.jsonl"
    pairs = generate_nl_sql_pairs(settings.num_nl_sql_pairs)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"✅ Saved {len(pairs)} NL→SQL pairs → {out_path}")


if __name__ == "__main__":
    main()
