"""
Tests validating synthetic data generation schema correctness.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInventoryDataGeneration:
    def test_generate_products_count(self):
        from data.generate_inventory import generate_products
        products = generate_products(50)
        assert len(products) == 50

    def test_product_schema(self):
        from data.generate_inventory import generate_products
        products = generate_products(10)
        required = {"product_id", "name", "category", "stock", "reorder_level", "price", "description"}
        for p in products:
            assert required.issubset(p.keys()), f"Missing keys in: {p}"

    def test_product_id_format(self):
        from data.generate_inventory import generate_products
        products = generate_products(20)
        for p in products:
            assert p["product_id"].startswith("P"), f"Bad product_id: {p['product_id']}"
            assert len(p["product_id"]) == 5

    def test_stock_non_negative(self):
        from data.generate_inventory import generate_products
        products = generate_products(50)
        for p in products:
            assert p["stock"] >= 0
            assert p["reorder_level"] > 0
            assert p["price"] > 0

    def test_categories_valid(self):
        from data.generate_inventory import generate_products
        valid_cats = {"Electronics", "Machinery", "Tools", "Safety", "Electrical"}
        products = generate_products(50)
        for p in products:
            assert p["category"] in valid_cats

    def test_description_non_empty(self):
        from data.generate_inventory import generate_products
        products = generate_products(10)
        for p in products:
            assert len(p["description"]) > 20


class TestQueryDatasetGeneration:
    def test_nl_sql_pair_schema(self):
        from data.generate_query_dataset import generate_nl_sql_pairs
        pairs = generate_nl_sql_pairs(50)
        for p in pairs:
            assert "input" in p
            assert "output" in p
            assert p["output"].strip().upper().startswith("SELECT")

    def test_nl_sql_count(self):
        from data.generate_query_dataset import generate_nl_sql_pairs
        pairs = generate_nl_sql_pairs(100)
        assert len(pairs) == 100


class TestToolCallingDatasetGeneration:
    def test_tool_schema(self):
        from data.generate_tool_calling_dataset import (
            check_stock_examples,
            update_inventory_examples,
            generate_report_examples,
        )
        examples = check_stock_examples()
        for ex in examples:
            output = json.loads(ex["output"])
            assert "tool" in output
            assert "input" in output
            assert output["tool"] in {"check_stock", "update_inventory", "generate_report", "run_sql_query"}

    def test_update_quantity_is_int(self):
        from data.generate_tool_calling_dataset import update_inventory_examples
        for ex in update_inventory_examples():
            output = json.loads(ex["output"])
            assert isinstance(output["input"]["quantity"], int)


class TestSemanticDatasetGeneration:
    def test_semantic_schema(self):
        from data.generate_semantic_dataset import generate_semantic_pairs
        pairs = generate_semantic_pairs(30)
        for p in pairs:
            assert "query" in p
            assert "relevant_category" in p
            assert "relevant_keywords" in p
            assert len(p["relevant_keywords"]) > 0
