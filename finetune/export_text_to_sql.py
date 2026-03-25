"""
Export NL→SQL fine-tuning dataset in Alpaca and ChatML formats.

Reads from: data/nl_to_sql_dataset.jsonl
Writes to:  finetune/text_to_sql_alpaca.jsonl
            finetune/text_to_sql_chatml.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_FILE = Path(__file__).parent.parent / "data" / "nl_to_sql_dataset.jsonl"
OUT_DIR = Path(__file__).parent
OUT_ALPACA = OUT_DIR / "text_to_sql_alpaca.jsonl"
OUT_CHATML = OUT_DIR / "text_to_sql_chatml.jsonl"

SCHEMA_CONTEXT = (
    "Table: products(product_id TEXT, name TEXT, category TEXT, "
    "stock INT, reorder_level INT, price FLOAT, description TEXT)"
)


def to_alpaca(item: dict) -> dict:
    return {
        "instruction": f"{item['instruction']}\n\nSchema: {SCHEMA_CONTEXT}",
        "input": item["input"],
        "output": item["output"],
    }


def to_chatml(item: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": f"{item['instruction']}\n\nSchema: {SCHEMA_CONTEXT}"},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
    }


def main():
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("   Run: python data/generate_query_dataset.py first.")
        return

    items = [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]

    with OUT_ALPACA.open("w") as fa, OUT_CHATML.open("w") as fc:
        for item in items:
            fa.write(json.dumps(to_alpaca(item)) + "\n")
            fc.write(json.dumps(to_chatml(item)) + "\n")

    print(f"✅ Exported {len(items)} NL→SQL pairs:")
    print(f"   Alpaca  → {OUT_ALPACA}")
    print(f"   ChatML  → {OUT_CHATML}")


if __name__ == "__main__":
    main()
