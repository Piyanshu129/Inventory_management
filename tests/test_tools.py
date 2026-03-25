"""
Unit tests for all 4 inventory tools.

Uses a temporary SQLite database created fresh per test via a shared conftest-style fixture.
We patch `db.database.execute_query` and `db.database.execute_write` directly to point
at a temp engine so we never touch the module-level singleton.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_engine(tmp_path):
    """Create a fresh in-memory SQLAlchemy engine, run schema, and seed test data."""
    from sqlalchemy import create_engine, text

    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )

    schema_file = Path(__file__).parent.parent / "db" / "schema.sql"
    ddl = schema_file.read_text()
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        conn.execute(text("""
            INSERT INTO products VALUES
              ('P0001', 'Hydraulic Pump Pro', 'Machinery', 5, 10, 499.99, 'Gear pump 250 bar SAE B flange'),
              ('P0002', 'Circuit Breaker', 'Electrical', 50, 20, 45.00, '3-pole 32A DIN rail mount')
        """))
    return engine


@pytest.fixture(autouse=True)
def patch_db(db_engine, monkeypatch):
    """Patch db.database helpers AND all tool modules that import them."""
    from sqlalchemy import text as sa_text

    def _query(sql, params=None):
        with db_engine.connect() as conn:
            result = conn.execute(sa_text(sql), params or {})
            if result.returns_rows:
                cols = list(result.keys())
                return [dict(zip(cols, row)) for row in result.fetchall()]
            return []

    def _write(sql, params=None):
        with db_engine.begin() as conn:
            result = conn.execute(sa_text(sql), params or {})
            return result.rowcount

    import db.database as db_mod
    import tools.check_stock as cs
    import tools.update_inventory as ui
    import tools.generate_report as gr
    import tools.run_sql_query as rq
    monkeypatch.setattr(db_mod, "execute_query", _query)
    monkeypatch.setattr(db_mod, "execute_write", _write)
    # Also patch via the _db alias each tool holds
    monkeypatch.setattr(cs._db, "execute_query", _query)
    monkeypatch.setattr(ui._db, "execute_query", _query)
    monkeypatch.setattr(ui._db, "execute_write", _write)
    monkeypatch.setattr(gr._db, "execute_query", _query)
    monkeypatch.setattr(rq._db, "execute_query", _query)


# ── check_stock ───────────────────────────────────────────────────────────────

class TestCheckStock:
    def test_found_low_stock(self):
        from tools.check_stock import check_stock
        result = check_stock("P0001")
        assert result["product_id"] == "P0001"
        assert result["stock"] == 5
        assert result["status"] == "low"
        assert result["is_low_stock"] is True

    def test_found_ok_stock(self):
        from tools.check_stock import check_stock
        result = check_stock("P0002")
        assert result["status"] == "ok"
        assert result["is_low_stock"] is False

    def test_not_found(self):
        from tools.check_stock import check_stock
        result = check_stock("P9999")
        assert result["status"] == "not_found"

    def test_case_insensitive(self):
        from tools.check_stock import check_stock
        result = check_stock("p0001")
        assert result["product_id"] == "P0001"


# ── update_inventory ──────────────────────────────────────────────────────────

class TestUpdateInventory:
    def test_update_success(self):
        from tools.update_inventory import update_inventory
        result = update_inventory("P0001", 100)
        assert result["success"] is True
        assert result["previous_stock"] == 5
        assert result["new_stock"] == 100
        assert result["is_low_stock"] is False

    def test_invalid_quantity_negative(self):
        from tools.update_inventory import update_inventory
        result = update_inventory("P0001", -5)
        assert result["success"] is False
        assert "Invalid" in result["error"]

    def test_product_not_found(self):
        from tools.update_inventory import update_inventory
        result = update_inventory("P9999", 50)
        assert result["success"] is False

    def test_zero_quantity(self):
        from tools.update_inventory import update_inventory
        result = update_inventory("P0001", 0)
        assert result["success"] is True
        assert result["new_stock"] == 0


# ── generate_report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_low_stock_report(self):
        from tools.generate_report import generate_report
        result = generate_report("low_stock")
        assert result["report_type"] == "low_stock"
        assert result["total_items"] == 1  # only P0001 (5 ≤ 10)
        assert result["products"][0]["product_id"] == "P0001"
        assert "total_low_stock_items" in result["summary"]

    def test_full_inventory_report(self):
        from tools.generate_report import generate_report
        result = generate_report("full_inventory")
        assert result["report_type"] == "full_inventory"
        assert result["total_items"] == 2
        assert "total_inventory_value" in result["summary"]

    def test_invalid_type(self):
        from tools.generate_report import generate_report
        result = generate_report("unknown_type")
        assert "error" in result


# ── run_sql_query ─────────────────────────────────────────────────────────────

class TestRunSqlQuery:
    def test_valid_select(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("SELECT * FROM products")
        assert result["success"] is True
        assert result["row_count"] == 2

    def test_select_with_filter(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("SELECT * FROM products WHERE category = 'Machinery'")
        assert result["success"] is True
        assert result["row_count"] == 1

    def test_blocks_delete(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("DELETE FROM products")
        assert result["success"] is False

    def test_blocks_update(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("UPDATE products SET stock=0")
        assert result["success"] is False

    def test_blocks_drop(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("DROP TABLE products")
        assert result["success"] is False

    def test_invalid_sql(self):
        from tools.run_sql_query import run_sql_query
        result = run_sql_query("SELECT * FROM nonexistent_table")
        assert result["success"] is False


# ── tool_registry ─────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_execute_check_stock(self):
        from tools.tool_registry import execute_tool
        result = execute_tool("check_stock", {"product_id": "P0001"})
        assert result["tool_name"] == "check_stock"
        assert result["product_id"] == "P0001"

    def test_execute_unknown_tool(self):
        from tools.tool_registry import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert "error" in result

    def test_execute_missing_param(self):
        from tools.tool_registry import execute_tool
        result = execute_tool("check_stock", {})
        assert "error" in result
