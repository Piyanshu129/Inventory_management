"""Centralized configuration for the Inventory Agent."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "none"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Database
    db_path: str = str(BASE_DIR / "data" / "inventory.db")

    # ChromaDB
    chroma_path: str = str(BASE_DIR / "data" / "chroma_db")
    chroma_collection: str = "products"

    # Agent
    memory_window: int = 10       # number of turns to keep in context
    rag_top_k: int = 5            # top-k semantic results
    low_stock_threshold_pct: float = 1.0  # stock <= reorder_level * threshold

    # Synthetic data sizes
    num_products: int = 200
    num_nl_sql_pairs: int = 500
    num_tool_pairs: int = 200
    num_semantic_pairs: int = 150


settings = Settings()
