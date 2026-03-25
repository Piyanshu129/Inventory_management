"""
ChromaDB vector store for product descriptions.

Provides:
  - build_index()  — embeds all products and upserts into ChromaDB
  - get_collection() — returns the ChromaDB collection handle
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_model: SentenceTransformer | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
      if settings.chroma_host:
            # Running inside Docker — connect to the standalone ChromaDB service
            logger.info(
                "Connecting to ChromaDB HTTP server at %s:%s",
                settings.chroma_host,
                settings.chroma_port,
            )
            _client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            # Local development — persist to disk
            chroma_path = Path(settings.chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(products: list[dict]) -> None:
    """
    Embed product descriptions and upsert into ChromaDB.

    Args:
        products: list of product dicts with at minimum:
                  product_id, name, category, stock, reorder_level, price, description
    """
    if not products:
        logger.warning("No products provided to build_index — skipping.")
        return

    model = _get_model()
    collection = get_collection()

    ids = [p["product_id"] for p in products]
    texts = [p["description"] for p in products]
    metadatas = [
        {
            "product_id": p["product_id"],
            "name": p["name"],
            "category": p["category"],
            "stock": p["stock"],
            "reorder_level": p["reorder_level"],
            "price": p["price"],
        }
        for p in products
    ]

    logger.info("Embedding %d products...", len(products))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        logger.info("Upserted batch %d/%d", i // batch_size + 1, -(-len(ids) // batch_size))

    logger.info("✅ ChromaDB index built with %d products.", len(products))


def index_from_db() -> None:
    """Load products from SQLite and index them into ChromaDB."""
    from db.database import execute_query, init_db
    init_db()
    products = execute_query("SELECT * FROM products")
    if not products:
        raise RuntimeError("No products found in database. Run generate_inventory.py first.")
    build_index(products)
    print(f"✅ Indexed {len(products)} products into ChromaDB.")


if __name__ == "__main__":
    index_from_db()
