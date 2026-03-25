"""
Integration tests for the ReAct agent (with mocked LLM).
Uses same per-test DB patching as test_tools.py.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture()
def db_engine(tmp_path):
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
              ('P0002', 'Power Supply 24VDC', 'Electronics', 50, 15, 129.99, 'Industrial 24VDC 10A UL listed'),
              ('P0003', 'Safety Helmet ANSI', 'Safety', 0, 5, 24.99, 'ANSI Z89.1 Class E ABS shell')
        """))
    return engine


@pytest.fixture(autouse=True)
def patch_db(db_engine, monkeypatch):
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
    monkeypatch.setattr(cs._db, "execute_query", _query)
    monkeypatch.setattr(ui._db, "execute_query", _query)
    monkeypatch.setattr(ui._db, "execute_write", _write)
    monkeypatch.setattr(gr._db, "execute_query", _query)
    monkeypatch.setattr(rq._db, "execute_query", _query)


@pytest.fixture
def mock_llm():
    with patch("agent.llm_client.chat_completion") as mock:
        yield mock


class TestAgentIntentClassification:
    def test_check_stock_by_id(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        mock_llm.return_value = "Product P0001 — Hydraulic Pump Pro has 5 units in stock. Status: LOW."

        response = agent.chat("Check stock for P0001")
        log = agent.get_step_log()
        think_steps = [s for s in log if s["step"] == "think"]
        assert any("check_stock" in str(s["data"]) for s in think_steps)

    def test_update_stock(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        mock_llm.return_value = "Updated P0001 stock from 5 to 100 units."

        response = agent.chat("Update stock for P0001 to 100 units")
        assert response  # Got a response

    def test_low_stock_report(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        mock_llm.return_value = "Found 2 items below reorder level."

        agent.chat("Generate a low stock report")
        log = agent.get_step_log()
        act_steps = [s for s in log if s["step"] == "act"]
        assert any("generate_report" in str(s["message"]) for s in act_steps)

    def test_sql_query_low_stock(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        # Mock: text_to_sql call, then synthesize response
        mock_llm.return_value = "SELECT * FROM products WHERE stock <= reorder_level"
        response = agent.chat("Which items need restocking?")
        assert response  # Got some response

    def test_memory_across_turns(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        mock_llm.return_value = "P0001 has 5 units."
        agent.chat("Check stock for P0001")
        assert len(agent.memory) >= 2

        mock_llm.return_value = "Yes, P0001 is below reorder level of 10."
        agent.chat("Is it below reorder level?")
        assert len(agent.memory) >= 4

    def test_reset_clears_memory(self, mock_llm):
        from agent.react_agent import InventoryAgent
        agent = InventoryAgent()
        mock_llm.return_value = "P0001 checked."
        agent.chat("Check stock for P0001")
        agent.reset()
        assert len(agent.memory) == 0


class TestIntentClassifier:
    def test_check_stock_rule(self):
        from agent.intent_classifier import classify_intent
        result = classify_intent("Check stock for P0001")
        assert result["intent"] == "check_stock"
        assert result["product_id"] == "P0001"

    def test_low_stock_report_rule(self):
        from agent.intent_classifier import classify_intent
        result = classify_intent("Generate a low stock report")
        assert result["intent"] == "generate_report"

    def test_update_stock_rule(self):
        from agent.intent_classifier import classify_intent
        result = classify_intent("Update stock for P0002 to 50 units")
        assert result["intent"] == "update_stock"

    def test_semantic_search_rule(self):
        from agent.intent_classifier import classify_intent
        result = classify_intent("Do we have any 220V power supplies?")
        assert result["intent"] == "semantic_search"

    def test_sql_keyword_heuristic(self):
        from agent.intent_classifier import classify_intent
        result = classify_intent("Show all Electronics items under $200")
        assert result["intent"] == "sql_query"
