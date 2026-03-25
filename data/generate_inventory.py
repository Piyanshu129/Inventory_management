"""
Generate synthetic inventory dataset and seed the SQLite database.

Produces:
  - data/inventory.json  (raw JSON list)
  - data/inventory.db    (seeded SQLite via db.database)
"""

import json
import random
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker
from config import settings
from db.database import init_db, execute_write

fake = Faker()
random.seed(42)

# ── Domain templates ─────────────────────────────────────────────────────────

CATEGORIES: list[dict] = [
    {
        "name": "Electronics",
        "products": [
            ("Voltage Regulator", "220V/110V dual input, ±2% regulation, 5A continuous output, DIN-rail mount"),
            ("Power Supply Unit", "Industrial 24VDC, 10A, built-in overload and short-circuit protection, UL listed"),
            ("Signal Converter", "4-20mA to 0-10V isolated converter, 0.1% accuracy, DIN-rail"),
            ("PLC Input Module", "16-channel digital input, 24VDC, IEC 61131-2 compliant, LED indicators"),
            ("Relay Module", "8-channel electromechanical relay, 10A/250VAC contact rating, optoisolated"),
            ("Proximity Sensor", "NPN inductive, 12-24VDC, IP67, M12 connector, 8mm sensing range"),
            ("Photoelectric Sensor", "Diffuse-reflective, 200mm range, NPN output, IP65, 10-30VDC"),
            ("Temperature Controller", "PID controller, SSR output, PT100/J/K thermocouple input, 1/32 DIN panel"),
            ("Variable Frequency Drive", "3-phase, 0.75kW, 415VAC input, RS-485 Modbus RTU, IP20"),
            ("HMI Touch Panel", "7-inch TFT, 800×480, 256MB RAM, RS-232/485/Ethernet, IP65 front panel"),
        ],
        "price_range": (50, 2000),
        "stock_range": (5, 300),
        "reorder_level": (15, 50),
    },
    {
        "name": "Machinery",
        "products": [
            ("Hydraulic Pump", "Gear pump, 25 L/min flow, 250 bar max pressure, SAE B flange, shaft seal"),
            ("Pneumatic Cylinder", "Double-acting, 50mm bore, 200mm stroke, ISO 15552, magnetic piston"),
            ("Ball Bearing", "Deep-groove, 6205-2RS, 25mm bore, 52mm OD, rubber sealed, 2-year grease fill"),
            ("Conveyor Belt", "PVC top, 800mm wide, 5m length, heat resistant to 80°C, 3-ply"),
            ("Chain Drive Sprocket", "40B-1 pitch chain, 20 teeth, 1-inch bore, hardened steel, keyway"),
            ("Centrifugal Pump", "Cast iron casing, 2.2kW, 50Hz, 30 m head, ANSI 150 flanges, mechanical seal"),
            ("Gear Reducer", "Helical, ratio 1:20, 5kW input, IEC B3 mount, oil bath lubrication, IP55"),
            ("Servo Motor", "400W, 220VAC, 3000 RPM, absolute encoder 17-bit, IP65, keyway + brake option"),
            ("Air Compressor", "Reciprocating, 2-stage, 11 bar, 500L/min FAD, 3-phase 415VAC, ASME tank"),
            ("Linear Actuator", "ACME screw, 1000N force, 300mm stroke, 24VDC, IP54, integrated Hall encoder"),
        ],
        "price_range": (100, 5000),
        "stock_range": (2, 100),
        "reorder_level": (5, 20),
    },
    {
        "name": "Tools",
        "products": [
            ("Torque Wrench", "3/8-inch drive, 10-60 Nm, ±3% accuracy, ratchet mechanism, ergonomic grip"),
            ("Digital Caliper", "0-150mm range, 0.01mm resolution, IP54, stainless steel jaws, RS-232 output"),
            ("Angle Grinder", "115mm disc, 720W, 11,000 RPM, tool-less guard adjustment, anti-vibration handle"),
            ("Impact Driver", "18V Li-Ion, 180 Nm max torque, brushless motor, quick-release chuck, LED"),
            ("Wire Stripper", "AWG 10-22, self-adjusting cam mechanism, looping hole, crimping die"),
            ("Pipe Wrench", "14-inch chrome-vanadium steel, drop-forged jaw, heel-jaw design, replaceable parts"),
            ("Level Gauge", "48-inch, aluminium frame, 3 bubble vials, magnetic base, accuracy ±0.5mm/m"),
            ("Multimeter", "True-RMS, CAT III 600V, 10A AC/DC, capacitance, frequency, NCV detector"),
            ("Hydraulic Jack", "3-ton floor jack, 200mm–480mm lift range, safety valve, chrome-plated cylinder"),
            ("Crimping Tool", "Hex crimping, 6-50mm² WAGO/Bootlace ferrule, hardened jaw, 16 positions"),
        ],
        "price_range": (20, 800),
        "stock_range": (10, 500),
        "reorder_level": (20, 80),
    },
    {
        "name": "Safety",
        "products": [
            ("Safety Helmet", "ANSI Z89.1 Class E, ABS shell, 6-point suspension, UV stabilized, ventilated"),
            ("Chemical Gloves", "Neoprene, 14-inch length, EN374 Level 6, acid and alkali resistant, flock lined"),
            ("Safety Harness", "Full-body, 140kg capacity, ANSI Z359.11, dorsal D-ring, padded shoulder straps"),
            ("Fire Extinguisher", "ABC dry powder, 6kg, 21A 113B C rating, pressure gauge, wall bracket"),
            ("Gas Detector", "4-in-1: CO, H2S, O2, LEL, ATEX certified, vibration + audible + visual alert"),
            ("Safety Goggles", "EN166 indirect vent, anti-fog, anti-scratch, polycarbonate lens, UV400 protection"),
            ("Ear Muffs", "NRR 30dB, padded cushion, adjustable headband, foldable, SNR 35dB"),
            ("Spill Kit", "30L absorbent capacity, oil-only pads, boom, disposal bag, hi-vis carry case"),
            ("Safety Sign Kit", "ISO 7010 compliant, rigid PVC, 10 signs, emergency exit / fire / hazard"),
            ("First Aid Cabinet", "ANSI 2021, 250-person, wall-mount, 170+ items, bilingual labels, AED compatible"),
        ],
        "price_range": (15, 600),
        "stock_range": (5, 200),
        "reorder_level": (10, 40),
    },
    {
        "name": "Electrical",
        "products": [
            ("Circuit Breaker", "3-pole, 32A, 10kA breaking capacity, DIN rail mount, IEC60947-2, trip indicator"),
            ("Cable Tray", "Perforated steel, hot-dip galvanized, 150×60mm, 3m length, load 75 kg/m"),
            ("Junction Box", "IP66, polycarbonate, 200×150×80mm, cable glands included, UV stabilized"),
            ("LED Work Light", "50W, 5000 lm, 6000K, IP65, die-cast aluminium, tempered glass, 50°C rated"),
            ("Conduit Pipe", "20mm EMT steel, hot-dip galvanized, 3m length, UL listed, threadable"),
            ("Cable Lug", "Copper tinned, 16mm², M8 bolt hole, DIN 46235, 100-piece pack, insulated collar"),
            ("Busbar System", "63A, 3P+N, 10-way, 35mm DIN, 10kA, pre-insulated PVC cover, pluggable"),
            ("Isolator Switch", "4-pole, 63A, 415VAC, IP65 enclosure, pad-lockable, 690V rated contacts"),
            ("Earth Leakage Relay", "30mA, 300ms delay, DIN-rail, test/reset, LED status, 110-415VAC supply"),
            ("Timer Relay", "Multifunction, 0.05s–100h range, SPDT, DIN-rail, 12-240VAC/DC supply"),
        ],
        "price_range": (10, 1200),
        "stock_range": (10, 400),
        "reorder_level": (20, 60),
    },
]


def make_product_id(index: int) -> str:
    return f"P{index:04d}"


def generate_products(n: int = 200) -> list[dict]:
    products = []
    idx = 1
    products_per_cat = n // len(CATEGORIES)
    extra = n % len(CATEGORIES)

    for cat_i, cat in enumerate(CATEGORIES):
        count = products_per_cat + (1 if cat_i < extra else 0)
        templates = cat["products"]

        for i in range(count):
            tmpl = templates[i % len(templates)]
            name, base_desc = tmpl
            # Slightly vary name to avoid exact duplicates
            suffix = fake.random_element(["", " Pro", " Plus", " Lite", " Industrial", " Heavy Duty"])
            full_name = f"{name}{suffix}".strip()

            price_lo, price_hi = cat["price_range"]
            stock_lo, stock_hi = cat["stock_range"]
            reorder_lo, reorder_hi = cat["reorder_level"]

            stock = random.randint(stock_lo, stock_hi)
            reorder_level = random.randint(reorder_lo, reorder_hi)
            price = round(random.uniform(price_lo, price_hi), 2)

            # Rich description with semantic vocabulary
            description = (
                f"{full_name} — {base_desc}. "
                f"Suitable for industrial warehouse and manufacturing environments. "
                f"Category: {cat['name']}. Unit price: ${price:.2f}. "
                f"Current stock: {stock} units. Reorder level: {reorder_level} units."
            )

            products.append(
                {
                    "product_id": make_product_id(idx),
                    "name": full_name,
                    "category": cat["name"],
                    "stock": stock,
                    "reorder_level": reorder_level,
                    "price": price,
                    "description": description,
                }
            )
            idx += 1

    random.shuffle(products)
    return products


def seed_database(products: list[dict]) -> None:
    init_db()
    for p in products:
        execute_write(
            """
            INSERT OR REPLACE INTO products
              (product_id, name, category, stock, reorder_level, price, description)
            VALUES
              (:product_id, :name, :category, :stock, :reorder_level, :price, :description)
            """,
            p,
        )
    print(f"✅ Seeded {len(products)} products into database.")


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🏭 Generating {settings.num_products} synthetic products...")
    products = generate_products(settings.num_products)

    out_json = out_dir / "inventory.json"
    out_json.write_text(json.dumps(products, indent=2))
    print(f"📄 Saved {len(products)} products → {out_json}")

    seed_database(products)
    print("🗄️  Database seeded successfully.")
    return products


if __name__ == "__main__":
    main()
