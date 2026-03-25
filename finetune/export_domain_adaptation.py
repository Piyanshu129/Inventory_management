"""
Export domain adaptation dataset for semantic embedding fine-tuning.

Reads from: data/semantic_dataset.jsonl + data/inventory.json (product descriptions)
Writes to:  finetune/domain_adaptation.jsonl
            finetune/contrastive_pairs.jsonl  (query + positive description)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEM_FILE = Path(__file__).parent.parent / "data" / "semantic_dataset.jsonl"
INV_FILE = Path(__file__).parent.parent / "data" / "inventory.json"
OUT_DIR = Path(__file__).parent
OUT_DOMAIN = OUT_DIR / "domain_adaptation.jsonl"
OUT_CONTRASTIVE = OUT_DIR / "contrastive_pairs.jsonl"


def main():
    missing = [f for f in [SEM_FILE, INV_FILE] if not f.exists()]
    if missing:
        for f in missing:
            print(f"❌ Missing: {f}")
        print("   Run the data generators first.")
        return

    sem_pairs = [json.loads(l) for l in SEM_FILE.read_text().splitlines() if l.strip()]
    products: list[dict] = json.loads(INV_FILE.read_text())

    # Build category → products map for contrastive matching
    cat_map: dict[str, list[dict]] = {}
    for p in products:
        cat_map.setdefault(p["category"], []).append(p)

    domain_records = []
    contrastive_records = []

    for pair in sem_pairs:
        query = pair["query"]
        rel_cat = pair["relevant_category"]
        keywords = pair["relevant_keywords"]
        snippet = pair["expected_description_snippet"]

        # Domain adaptation record: query + enriched description
        domain_records.append({
            "text": f"Query: {query}\nDescription: {snippet}",
            "category": rel_cat,
            "keywords": keywords,
        })

        # Contrastive record: query + positive (same category) description
        cat_products = cat_map.get(rel_cat, [])
        if cat_products:
            positive = cat_products[len(contrastive_records) % len(cat_products)]
            contrastive_records.append({
                "query": query,
                "positive": positive["description"],
                "positive_id": positive["product_id"],
                "category": rel_cat,
            })

    with OUT_DOMAIN.open("w") as f:
        for r in domain_records:
            f.write(json.dumps(r) + "\n")

    with OUT_CONTRASTIVE.open("w") as f:
        for r in contrastive_records:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Exported {len(domain_records)} domain adaptation records → {OUT_DOMAIN}")
    print(f"✅ Exported {len(contrastive_records)} contrastive pairs → {OUT_CONTRASTIVE}")


if __name__ == "__main__":
    main()
