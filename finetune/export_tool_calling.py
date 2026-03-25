"""
Export tool-calling fine-tuning dataset in ChatML format.

Reads from: data/tool_calling_dataset.jsonl
Writes to:  finetune/tool_calling_chatml.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_FILE = Path(__file__).parent.parent / "data" / "tool_calling_dataset.jsonl"
OUT_FILE = Path(__file__).parent / "tool_calling_chatml.jsonl"

TOOL_SYSTEM_PROMPT = """\
You are a warehouse inventory assistant. When the user requests an action,
output a JSON tool call in this exact format:
{"tool": "<tool_name>", "input": {<parameters>}}

Available tools:
- check_stock: {"product_id": string}
- update_inventory: {"product_id": string, "quantity": integer}
- generate_report: {"type": "low_stock" | "full_inventory"}
- run_sql_query: {"query": "SQL SELECT string"}

Output ONLY valid JSON. No explanation."""


def to_chatml(item: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
    }


def main():
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("   Run: python data/generate_tool_calling_dataset.py first.")
        return

    items = [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]

    with OUT_FILE.open("w") as f:
        for item in items:
            f.write(json.dumps(to_chatml(item)) + "\n")

    print(f"✅ Exported {len(items)} tool-calling examples → {OUT_FILE}")


if __name__ == "__main__":
    main()
