"""
feedback/feedback_reranker.py
==============================
Section 4 — Re-rank RAG chunks based on past feedback memory.

How it works:
  1. Load chunk reputation from FeedbackMemory
  2. For each retrieved chunk, look up its past good/bad score
  3. Adjust chunk ordering — boost "good" chunks, demote "bad" ones
  4. Return reranked list

Used inside retrieve_docs() in multi_agent_rag.py (already wired in).
"""

from typing import List, Dict
from feedback.feedback_memory import FeedbackMemory


def rerank_with_feedback(
    query:  str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    Re-rank retrieved RAG chunks using past feedback.

    Each chunk gets a combined score:
      combined = retrieval_score_inverted + reputation_boost

    Where:
      retrieval_score_inverted = 1 / (1 + L2_distance)
        → higher is better (inverts FAISS L2 distance)
      reputation_boost = chunk reputation score from feedback
        → 0.0 (always bad) to 1.0 (always good), default 0.5

    Args:
        query  : current user query (used to find similar past feedback)
        chunks : list of dicts with keys: text, source, chunk_id, score

    Returns:
        Reranked list of chunks (best first)
    """

    if not chunks:
        return chunks

    # ── Load chunk reputation from feedback store ──────────────────────────
    fm         = FeedbackMemory()
    reputation = fm.get_chunk_reputation()

    # ── Score each chunk ───────────────────────────────────────────────────
    scored_chunks = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        l2_score = chunk.get("score", 1.0)   # lower L2 = more similar

        # Convert L2 distance to similarity (0 to 1 range, higher = better)
        retrieval_sim = 1.0 / (1.0 + l2_score)

        # Get reputation score from past feedback (default 0.5 = neutral)
        rep = reputation.get(chunk_id, {})
        rep_score = rep.get("score", 0.5)

        # Combined score: 70% retrieval + 30% reputation
        combined = (0.7 * retrieval_sim) + (0.3 * rep_score)

        scored_chunks.append({
            **chunk,
            "retrieval_sim":  round(retrieval_sim, 4),
            "rep_score":      round(rep_score, 4),
            "combined_score": round(combined, 4),
        })

    # ── Sort by combined score (highest first) ─────────────────────────────
    scored_chunks.sort(key=lambda x: x["combined_score"], reverse=True)

    return scored_chunks