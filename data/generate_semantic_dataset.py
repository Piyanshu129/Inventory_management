"""
Generate semantic search training dataset (query → product description pairs).

Produces: data/semantic_dataset.jsonl
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

random.seed(42)

# ── Semantic Query → Relevant Product Description mappings ───────────────────

SEMANTIC_PAIRS: list[dict] = [
    # Electronics
    {"query": "high voltage equipment",
     "relevant_category": "Electronics",
     "relevant_keywords": ["220V", "voltage", "power", "electrical"],
     "description_snippet": "220V/110V dual input industrial power supply with overload protection"},

    {"query": "industrial power supplies",
     "relevant_category": "Electronics",
     "relevant_keywords": ["24VDC", "power", "industrial", "UL listed"],
     "description_snippet": "Industrial 24VDC power supply with built-in short-circuit protection"},

    {"query": "factory automation sensors",
     "relevant_category": "Electronics",
     "relevant_keywords": ["sensor", "proximity", "inductive", "NPN"],
     "description_snippet": "NPN inductive proximity sensor with IP67 rating and M12 connector"},

    {"query": "motor control drives",
     "relevant_category": "Electronics",
     "relevant_keywords": ["VFD", "variable frequency", "3-phase", "motor"],
     "description_snippet": "Variable frequency drive for 3-phase motors with RS-485 Modbus RTU interface"},

    {"query": "PLC control modules",
     "relevant_category": "Electronics",
     "relevant_keywords": ["PLC", "programmable", "input module", "IEC 61131"],
     "description_snippet": "16-channel digital input module for PLCs, IEC 61131-2 compliant"},

    # Machinery
    {"query": "hydraulic systems heavy duty",
     "relevant_category": "Machinery",
     "relevant_keywords": ["hydraulic", "pump", "pressure", "bar"],
     "description_snippet": "Hydraulic gear pump rated 250 bar max pressure for heavy industrial use"},

    {"query": "industrial pumping equipment",
     "relevant_category": "Machinery",
     "relevant_keywords": ["pump", "centrifugal", "flow", "mechanical seal"],
     "description_snippet": "Centrifugal pump with cast iron casing and mechanical seal for industrial fluids"},

    {"query": "pneumatic actuators and cylinders",
     "relevant_category": "Machinery",
     "relevant_keywords": ["pneumatic", "cylinder", "double-acting", "ISO"],
     "description_snippet": "Double-acting pneumatic cylinder ISO 15552 with magnetic piston"},

    {"query": "precision ball bearings",
     "relevant_category": "Machinery",
     "relevant_keywords": ["bearing", "ball bearing", "sealed", "grease"],
     "description_snippet": "Deep-groove ball bearing with rubber seal and 2-year grease fill"},

    {"query": "conveyor and material handling parts",
     "relevant_category": "Machinery",
     "relevant_keywords": ["conveyor", "belt", "PVC", "heat resistant"],
     "description_snippet": "PVC conveyor belt heat resistant to 80°C for industrial material handling"},

    # Tools
    {"query": "precision measurement tools",
     "relevant_category": "Tools",
     "relevant_keywords": ["caliper", "digital", "measurement", "resolution"],
     "description_snippet": "Digital caliper 0-150mm range with 0.01mm resolution and RS-232 output"},

    {"query": "heavy duty power tools",
     "relevant_category": "Tools",
     "relevant_keywords": ["angle grinder", "impact", "power", "brushless"],
     "description_snippet": "Brushless impact driver 180 Nm with 18V Li-Ion battery"},

    {"query": "electrical installation tools",
     "relevant_category": "Tools",
     "relevant_keywords": ["wire stripper", "crimp", "cable", "AWG"],
     "description_snippet": "Self-adjusting wire stripper for AWG 10-22 with crimping die"},

    {"query": "torque and fastening equipment",
     "relevant_category": "Tools",
     "relevant_keywords": ["torque wrench", "Nm", "ratchet", "drive"],
     "description_snippet": "3/8-inch drive torque wrench 10-60 Nm with ratchet mechanism"},

    {"query": "lifting and jacking equipment",
     "relevant_category": "Tools",
     "relevant_keywords": ["jack", "hydraulic", "floor jack", "lift"],
     "description_snippet": "3-ton hydraulic floor jack with 200-480mm lift range and safety valve"},

    # Safety
    {"query": "personal protective equipment PPE",
     "relevant_category": "Safety",
     "relevant_keywords": ["helmet", "harness", "gloves", "goggles", "protection"],
     "description_snippet": "Full-body safety harness 140kg capacity ANSI Z359.11 with dorsal D-ring"},

    {"query": "fire suppression and safety equipment",
     "relevant_category": "Safety",
     "relevant_keywords": ["fire", "extinguisher", "ABC", "dry powder"],
     "description_snippet": "ABC dry powder fire extinguisher 6kg with pressure gauge"},

    {"query": "gas detection hazardous environments",
     "relevant_category": "Safety",
     "relevant_keywords": ["gas detector", "CO", "H2S", "ATEX", "LEL"],
     "description_snippet": "4-in-1 gas detector for CO, H2S, O2, LEL in ATEX certified environments"},

    {"query": "chemical handling protective gear",
     "relevant_category": "Safety",
     "relevant_keywords": ["chemical", "gloves", "neoprene", "acid", "alkali"],
     "description_snippet": "Neoprene chemical gloves EN374 Level 6, acid and alkali resistant"},

    {"query": "industrial spill containment",
     "relevant_category": "Safety",
     "relevant_keywords": ["spill kit", "absorbent", "oil", "disposal"],
     "description_snippet": "30L oil-only spill kit with absorbent pads, boom and disposal bag"},

    # Electrical
    {"query": "circuit protection devices",
     "relevant_category": "Electrical",
     "relevant_keywords": ["circuit breaker", "MCB", "DIN", "breaking capacity"],
     "description_snippet": "3-pole 32A circuit breaker 10kA breaking capacity DIN rail mount"},

    {"query": "industrial lighting LED",
     "relevant_category": "Electrical",
     "relevant_keywords": ["LED", "light", "lumens", "IP65", "work light"],
     "description_snippet": "50W LED work light 5000 lm IP65 die-cast aluminium with tempered glass"},

    {"query": "cable management systems",
     "relevant_category": "Electrical",
     "relevant_keywords": ["cable tray", "conduit", "perforated", "galvanized"],
     "description_snippet": "Perforated hot-dip galvanized cable tray 150×60mm 3m length"},

    {"query": "power distribution equipment",
     "relevant_category": "Electrical",
     "relevant_keywords": ["busbar", "isolator", "distribution", "DIN"],
     "description_snippet": "63A busbar system 10-way DIN rail with pre-insulated PVC cover"},

    {"query": "earth leakage and RCD protection",
     "relevant_category": "Electrical",
     "relevant_keywords": ["earth leakage", "RCD", "30mA", "fault protection"],
     "description_snippet": "Earth leakage relay 30mA sensitivity 300ms delay with test/reset function"},
]

QUERY_VARIATIONS = [
    "{query}",
    "industrial {query}",
    "warehouse {query}",
    "heavy-duty {query}",
    "factory grade {query}",
    "{query} for manufacturing",
    "{query} equipment",
    "buy {query}",
    "need {query}",
    "looking for {query}",
    "search {query}",
    "{query} products",
    "best {query} for industrial use",
    "commercial {query}",
]


def generate_semantic_pairs(n: int = 150) -> list[dict]:
    examples = []
    idx = 0
    for pair in SEMANTIC_PAIRS:
        query_base = pair["query"].lower()
        for var_tmpl in QUERY_VARIATIONS:
            augmented_query = var_tmpl.format(query=query_base)
            examples.append({
                "id": f"sem_{idx:04d}",
                "query": augmented_query,
                "relevant_category": pair["relevant_category"],
                "relevant_keywords": pair["relevant_keywords"],
                "expected_description_snippet": pair["description_snippet"],
            })
            idx += 1
            if len(examples) >= n:
                break
        if len(examples) >= n:
            break

    random.shuffle(examples)
    return examples[:n]


def main():
    out_path = Path(__file__).parent / "semantic_dataset.jsonl"
    pairs = generate_semantic_pairs(settings.num_semantic_pairs)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"✅ Saved {len(pairs)} semantic search pairs → {out_path}")


if __name__ == "__main__":
    main()
