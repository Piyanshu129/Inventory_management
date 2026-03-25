# 🤖 Inventory Management Agent

A **production-grade agentic warehouse intelligence system** built with Python. It combines a ReAct reasoning loop, hybrid intent classification, local fine-tuned SQL generation, semantic search via ChromaDB, and a sliding-window conversation memory — all accessible via a CLI REPL or REST API.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **ReAct Agent** | Think → Act → Observe → Respond loop for multi-step reasoning |
| **Hybrid Intent Classifier** | Rule heuristics first, LLM fallback — minimizes API calls |
| **Hybrid SQL Generation** | Local fine-tuned 7B Qwen model first, OpenRouter 72B fallback |
| **Semantic Search** | ChromaDB + SentenceTransformer (`all-MiniLM-L6-v2`) |
| **Sliding-Window Memory** | Last 10 conversation turns kept in context |
| **REST API** | FastAPI with OpenAI-compatible tool call format |
| **CLI REPL** | Interactive terminal with Rich formatting |
| **Fine-Tuning Pipeline** | QLoRA training scripts + dataset exporters included |

---

## 📐 Architecture

```
inventory/
├── agent/
│   ├── react_agent.py         # ReAct loop: Think → Act → Observe → Respond
│   ├── intent_classifier.py   # Rule heuristics + LLM hybrid intent detection
│   ├── text_to_sql.py         # NL→SQL: local model first, OpenRouter fallback
│   ├── local_text_to_sql.py   # Singleton LoRA adapter loader (CPU/GPU auto)
│   ├── memory.py              # Sliding-window conversation memory (RAM)
│   └── llm_client.py          # OpenAI-compatible client wrapper
│
├── tools/
│   ├── check_stock.py         # Query a product's stock level by ID
│   ├── update_inventory.py    # Update stock quantity for a product
│   ├── generate_report.py     # Low-stock / full-inventory reports
│   ├── run_sql_query.py       # Read-only SQL execution (SQL injection guarded)
│   └── tool_registry.py       # Central tool dispatcher
│
├── vector_db/
│   ├── embedder.py            # ChromaDB indexer (product descriptions)
│   └── retriever.py           # Semantic search interface
│
├── db/
│   ├── schema.sql             # SQLite schema definition
│   └── database.py            # Query helpers
│
├── data/
│   ├── generate_inventory.py        # 200 synthetic industrial products
│   ├── generate_query_dataset.py    # 500 NL→SQL training pairs
│   ├── generate_tool_calling_dataset.py  # 200 tool-calling examples
│   └── generate_semantic_dataset.py     # 150 semantic search pairs
│
├── finetune/
│   ├── collab_finetune.ipynb              # QLoRA fine-tuning script (TRL + PEFT)
│   ├── export_text_to_sql.py       # Export NL→SQL in Alpaca / ChatML format
│   ├── export_tool_calling.py      # Export tool-calling dataset (ChatML)
│   └── export_domain_adaptation.py # Contrastive pairs for domain adaptation
│
├── tests/
│   ├── test_tools.py          # Unit tests for all tools
│   ├── test_agent.py          # Integration tests for the agent
│   └── test_data_gen.py       # Schema validation for generated data
│
├── main.py                    # FastAPI app + CLI entrypoint
└── config.py                  # Pydantic-settings config (reads .env)
```

---

## ⚙️ How the Agent Works

```
User Query
    │
    ▼
Intent Classifier ──────────────────────────────────────────┐
  • Rule heuristics (SQL keywords, product IDs, "report")   │
  • LLM fallback (OpenRouter 72B) for ambiguous queries      │
    │                                                        │
    ▼                                                        │
Intent: one of ─────────────────────────────────────────────┘
  check_stock      → Tool: check_stock(product_id)
  update_stock     → Tool: update_inventory(product_id, qty)
  generate_report  → Tool: generate_report(report_type)
  sql_query        → NL→SQL (local 7B model → OpenRouter fallback)
  semantic_search  → ChromaDB vector similarity search
  general          → LLM direct answer (no tool call)
    │
    ▼
Conversation Memory (last 10 turns)
    │
    ▼
LLM Answer Synthesis (OpenRouter 72B)
    │
    ▼
Formatted Response to User
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai) API key (or any OpenAI-compatible LLM)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd inventory
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen-2.5-72b-instruct
LLM_API_KEY=sk-or-your-openrouter-key

# Optional: local fine-tuned model path (for SQL generation)
# LOCAL_SQL_MODEL_PATH=./finetune/checkpoints/text_to_sql/checkpoint-100
```

> **Offline mode**: If no LLM is configured, the agent uses a built-in rule-based stub. All tools and SQL queries still work, but answer synthesis will be basic.

### 3. Setup (generate data + seed DB + build vector index)

```bash
uv run python main.py --setup
```

This will:
- Generate 200 synthetic industrial products in SQLite
- Generate 500 NL→SQL training pairs, 200 tool-calling examples, 150 semantic pairs
- Embed all products into ChromaDB using `all-MiniLM-L6-v2`

### 4. Run CLI REPL

```bash
uv run python main.py --cli
```

Use `--debug` to see which model (local vs API) handled each SQL query:

```bash
uv run python main.py --cli --debug 2>agent_debug.log
# In a second terminal:
tail -f agent_debug.log | grep -E "\[local-model\]|\[api\]|local_text_to_sql"
```

### 5. Run API Server

```bash
uv run uvicorn main:app --reload --port 8000
```

---

## 💬 Sample Queries

| Query | Intent | Tool / Path |
|---|---|---|
| `Check stock for P0001` | `check_stock` | `check_stock` tool |
| `Update P0001 to 150 units` | `update_stock` | `update_inventory` tool |
| `Generate a low stock report` | `generate_report` | `generate_report` tool |
| `Show all Electronics items under $500` | `sql_query` | Local model → SQL |
| `Find items with cable in description` | `semantic_search` | ChromaDB similarity |
| `Which items need restocking?` | `sql_query` | Local model → SQL |
| `How many Machinery products do we have?` | `sql_query` | Local model → SQL |

---

## 🌐 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Chat with the agent |
| `POST` | `/tool` | Direct tool call (bypass agent) |
| `GET` | `/tools` | List all available tools |
| `GET` | `/report/{type}` | Quick report (`low_stock` / `full_inventory`) |

### Chat Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Which items are low in stock?"}'
```

### Direct Tool Call

```bash
curl -X POST http://localhost:8000/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "check_stock", "input": {"product_id": "P0001"}}'
```

### Single-Query Mode (no REPL)

```bash
uv run python main.py --query "Show me all machinery items below reorder level"
```

---

## 🧠 Hybrid Inference (Local Model + OpenRouter)

The agent uses a **two-tier strategy** for SQL generation:

```
SQL Query Intent Detected
        │
        ▼
Try local fine-tuned Qwen2.5-7B (LoRA checkpoint)
        │
   Success? ──YES──► Execute SQL
        │
       NO (model unavailable or invalid SQL)
        │
        ▼
Fall back to OpenRouter 72B API
        │
        ▼
    Execute SQL
```

**Benefits:**
- **Lower latency** for SQL generation (local model is instant on GPU)
- **Cost savings** — the 72B API is only called for reasoning/synthesis
- **Offline capability** — works without internet for SQL queries

### Setting Up the Local Model

If you have trained the fine-tuned adapter (see [Fine-Tuning](#-fine-tuning)):

```bash
# The model loads automatically if the checkpoint folder exists at:
finetune/checkpoints/text_to_sql/checkpoint-100

# Verify which backend is active with --debug:
uv run python main.py --cli --debug 2>agent_debug.log
grep "local-model\|api\]" agent_debug.log
```

You'll see one of:
```
INFO [agent.text_to_sql] [local-model] Generated SQL: SELECT ...
INFO [agent.text_to_sql] [api] Generated SQL: SELECT ...
```

> ⚠️ **GPU Note:** If your CUDA driver is older than `535`, the model will load on **CPU** (slower but functional). Upgrade with `sudo apt install nvidia-driver-535`.

---

## 🔧 Configuration

All settings are in `.env` (loaded by `config.py` via Pydantic Settings):

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Model name to use |
| `LLM_API_KEY` | `none` | API key |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for ChromaDB |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `products` | ChromaDB collection name |
| `DB_PATH` | `data/inventory.db` | SQLite database path |
| `MEMORY_WINDOW` | `10` | Conversation turns to keep in context |
| `RAG_TOP_K` | `5` | Top-k results for semantic search |
| `NUM_PRODUCTS` | `200` | Synthetic products to generate |

---

## 🧪 Running Tests

```bash
uv run pytest tests/ -v
```

Individual test files:
```bash
uv run pytest tests/test_tools.py -v      # Tool unit tests
uv run pytest tests/test_agent.py -v      # Agent integration tests
uv run pytest tests/test_data_gen.py -v   # Data generation schema tests
```

---

## 🗂️ Memory & Vector DB Explained

### Conversation Memory (`agent/memory.py`)
- **Stored in RAM** — not persisted to disk
- Keeps the last **10 turns** (configurable via `MEMORY_WINDOW`)
- Resets when you restart the agent (type `reset` inside CLI to clear mid-session)
- Includes simple **coreference resolution** (`"it"`, `"that"`, `"they"` → resolved to the last mentioned product)

```bash
# Inspect memory window size:
uv run python -c "from config import settings; print('Memory window:', settings.memory_window)"
```

### ChromaDB Vector DB (`vector_db/`)
- **Persisted to disk** at `data/chroma_db/`
- Stores **product descriptions** (not chat history)
- Populated once during `--setup`, reused across restarts
- Used only for `semantic_search` intent

```bash
# Inspect ChromaDB contents:
uv run python -c "
from vector_db.embedder import get_collection
col = get_collection()
print('Products in ChromaDB:', col.count())
for i, (doc, meta) in enumerate(zip(*[col.peek(5)['documents'], col.peek(5)['metadatas']])):
    print(f'  [{i+1}] {meta[\"name\"]} ({meta[\"product_id\"]})')
"
```

---

## 🎯 Fine-Tuning

The `finetune/` directory contains everything needed to train your own SQL generation model.

### Dataset Export

After running `--setup`, export training data:

```bash
# NL→SQL pairs (Alpaca + ChatML formats)
uv run python finetune/export_text_to_sql.py

# Tool-calling dataset (ChatML)
uv run python finetune/export_tool_calling.py

# Domain adaptation contrastive pairs
uv run python finetune/export_domain_adaptation.py
```

### Training with QLoRA

```bash
# Activate your conda/venv with PyTorch+CUDA
conda activate inventory-finetune

# Run QLoRA fine-tuning on Qwen2.5-7B-Instruct
python finetune/hf_finetune.py
```

Training configuration (inside `hf_finetune.py`):
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: QLoRA (4-bit quantized LoRA via PEFT + TRL)
- Dataset: `finetune/text_to_sql_alpaca.jsonl`
- Output: `finetune/checkpoints/text_to_sql/`

### Using Your Checkpoint

The agent automatically picks up your checkpoint at:
```
finetune/checkpoints/text_to_sql/checkpoint-100
```

No code changes needed — just ensure the folder exists.

---

## 🗄️ Database Schema

```sql
CREATE TABLE products (
    product_id    TEXT PRIMARY KEY,   -- e.g. P0001
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,      -- Electronics, Machinery, Electrical, Safety, Tools
    stock         INTEGER NOT NULL,
    reorder_level INTEGER NOT NULL,
    price         REAL NOT NULL,
    description   TEXT
);
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent framework | Custom ReAct loop (Python) |
| LLM API | OpenRouter (Qwen2.5-72B-Instruct) |
| Local SQL model | Qwen2.5-7B-Instruct + QLoRA (PEFT) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Database | SQLite via Python `sqlite3` |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Config | Pydantic Settings |
| Fine-tuning | HuggingFace TRL (SFTTrainer) + PEFT |
| Package manager | `uv` |

---

## 📁 Data Files (Git-ignored)

The following are excluded from version control (see `.gitignore`):

```
data/inventory.db          # SQLite database (auto-generated by --setup)
data/chroma_db/            # ChromaDB vector store (auto-generated by --setup)
finetune/checkpoints/      # Model checkpoints (large binary files)
finetune/text_to_sql_alpaca.jsonl  # Generated training data
.env                       # API keys and secrets
agent_debug.log            # Debug log output
```

Regenerate all data with:
```bash
uv run python main.py --setup
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
