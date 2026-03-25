"""
Semantic search retriever backed by ChromaDB.

Usage:
    from vector_db.retriever import semantic_search
    results = semantic_search("high voltage 220V equipment", top_k=5)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)


def semantic_search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Perform semantic search over product descriptions.

    Args:
        query:  Natural language search query
        top_k:  Number of top results to return (default from config)

    Returns:
        List of dicts with keys: product_id, name, category, stock,
        reorder_level, price, description, similarity_score
    """
    from sentence_transformers import SentenceTransformer
    from vector_db.embedder import get_collection

    k = top_k or settings.rag_top_k

    # Lazy-load model (shared with embedder singleton)
    model = SentenceTransformer(settings.embedding_model)
    query_embedding = model.encode([query])[0].tolist()

    collection = get_collection()
    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty. Run embedder.index_from_db() first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        include=["metadatas", "documents", "distances"],
    )

    output = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, pid in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        # ChromaDB cosine distance → similarity score (0–1)
        similarity = round(1 - dist, 4)

        output.append(
            {
                "product_id": pid,
                "name": meta.get("name", ""),
                "category": meta.get("category", ""),
                "stock": meta.get("stock", 0),
                "reorder_level": meta.get("reorder_level", 0),
                "price": meta.get("price", 0.0),
                "description": doc,
                "similarity_score": similarity,
            }
        )

    output.sort(key=lambda x: x["similarity_score"], reverse=True)
    return output
