"""
Main entrypoint — FastAPI server + CLI REPL mode.

Usage:
  # Start API server:
  uvicorn main:app --reload --port 8000

  # Start CLI REPL:
  python main.py --cli

  # Setup (generate data + seed DB + build vector index):
  python main.py --setup
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)

# ── FastAPI App ────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Inventory Management Agent API",
    description="Production-grade agentic warehouse intelligence system.",
    version="0.1.0",
)

# Lazy-init agent (created once on first request)
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from agent.react_agent import InventoryAgent
        _agent = InventoryAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query")
    session_reset: bool = Field(False, description="Reset conversation memory before this turn")


class ChatResponse(BaseModel):
    response: str
    reasoning_trace: list[dict]


class ToolCallRequest(BaseModel):
    tool: str
    input: dict


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    agent = _get_agent()
    if request.session_reset:
        agent.reset()
    try:
        response = agent.chat(request.message)
        trace = agent.get_step_log()
        return ChatResponse(response=response, reasoning_trace=trace)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/tool")
def call_tool(request: ToolCallRequest):
    """Directly call a tool by name (bypasses agent reasoning)."""
    from tools.tool_registry import execute_tool
    result = execute_tool(request.tool, request.input)
    return result


@app.get("/tools")
def list_tools():
    from tools.tool_registry import TOOL_SCHEMAS
    return {"tools": TOOL_SCHEMAS}


@app.get("/report/{report_type}")
def quick_report(report_type: str):
    """Quick endpoint for low_stock or full_inventory reports."""
    from tools.generate_report import generate_report
    result = generate_report(report_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

cli = typer.Typer(help="Inventory Management Agent CLI")


@cli.command()
def chat_cli(
    cli: bool = typer.Option(False, "--cli", help="Start interactive REPL"),
    setup: bool = typer.Option(False, "--setup", help="Generate data, seed DB, and build vector index"),
    query: str = typer.Option(None, "--query", "-q", help="Single query mode"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging (shows local-model vs API routing)"),
):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Focus on our agent modules only
        for name in ("agent.text_to_sql", "agent.local_text_to_sql", "agent.llm_client",
                     "agent.intent_classifier", "agent.react_agent"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        console.print("[dim]Debug logging enabled — you'll see [local-model] vs [api] routing.[/dim]\n")
    if setup:
        _run_setup()
        return

    from agent.react_agent import InventoryAgent
    agent = InventoryAgent()

    if query:
        _single_query(agent, query)
        return

    if cli:
        _interactive_repl(agent)
        return

    console.print("[bold yellow]No mode selected. Use --cli, --setup, or --query 'your query'[/]")
    console.print("Example: python main.py --cli")


def _run_setup():
    console.rule("[bold blue]🏭 Inventory Agent Setup[/]")
    console.print("[cyan]Step 1/3: Generating synthetic data...[/]")
    from data.generate_inventory import main as gen_inventory
    products = gen_inventory()

    console.print("[cyan]Step 2/3: Generating training datasets...[/]")
    from data.generate_query_dataset import main as gen_nl_sql
    from data.generate_tool_calling_dataset import main as gen_tools
    from data.generate_semantic_dataset import main as gen_semantic
    gen_nl_sql()
    gen_tools()
    gen_semantic()

    console.print("[cyan]Step 3/3: Building ChromaDB vector index...[/]")
    from vector_db.embedder import build_index
    build_index(products)

    console.print("\n[bold green]✅ Setup complete! You can now run:[/]")
    console.print("  python main.py --cli              # interactive REPL")
    console.print("  uvicorn main:app --reload         # API server")


def _single_query(agent, query: str):
    console.rule("[bold blue]🤖 Inventory Agent[/]")
    console.print(f"[dim]Query:[/dim] {query}\n")
    with console.status("[bold cyan]Thinking..."):
        response = agent.chat(query)
    console.print(Markdown(response))

    trace = agent.get_step_log()
    if trace:
        console.print("\n[dim]Reasoning trace:[/dim]")
        for step in trace:
            console.print(f"  [cyan]{step['step'].upper()}[/cyan]: {step['message']}")


def _interactive_repl(agent):
    console.rule("[bold blue]🤖 Inventory Management Agent[/]")
    console.print("[dim]Type your query and press Enter. Type 'exit' to quit, 'reset' to clear memory.[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if query.lower() == "reset":
            agent.reset()
            console.print("[dim]Memory cleared.[/dim]")
            continue

        with console.status("[bold cyan]Thinking..."):
            response = agent.chat(query)

        console.print(Panel(Markdown(response), title="[bold blue]Agent", border_style="blue"))

        trace = agent.get_step_log()
        steps = [(s["step"], s["message"]) for s in trace]
        if steps:
            console.print(f"[dim]  ↳ " + " → ".join(f"[{s}] {m}" for s, m in steps) + "[/dim]\n")


if __name__ == "__main__":
    cli()


#for debug first run
# uv run python main.py --cli --debug 2>agent_debug.log

# then run this on another terminal
# tail -f agent_debug.log | grep -E "\[local-model\]|\[api\]|local_text_to_sql"

#uv run python -c "from config import settings; print('Memory window:', settings.memory_window)"
