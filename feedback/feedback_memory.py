"""
feedback/feedback_memory.py
============================
Section 4 — Feedback Memory System

Features:
  1. Store user feedback (good/bad) against queries + answers
  2. Tag outputs as GOOD / BAD
  3. Find similar past feedback using embedding similarity
  4. Re-rank future retrieval based on past feedback

Folder structure expected:
  your_project/
  ├── multi_agent_rag.py
  ├── feedback/
  │   ├── __init__.py          ← empty file
  │   ├── feedback_memory.py   ← this file
  │   └── feedback_reranker.py ← reranker (separate file)

Usage:
  from feedback.feedback_memory import FeedbackMemory
  fm = FeedbackMemory()
  fm.store("What is LTA?", "9 employees eligible", "good", ["chunk_abc"])
  fm.show_all()
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# ── Embedding model (same as main pipeline) ───────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
FEEDBACK_FILE = Path(__file__).resolve().parent / "feedback_store.json"


# --------------------------------───────────────
# FEEDBACK MEMORY CLASS
# --------------------------------───────────────

class FeedbackMemory:
    """
    Stores and retrieves user feedback for past queries.

    Each feedback entry contains:
      - query         : the user's question
      - answer        : the system's response
      - tag           : "good" or "bad"
      - chunk_ids     : which RAG chunks were used
      - timestamp     : when feedback was given
      - query_embedding : vector for similarity search
    """

    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.store: List[Dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        """Load existing feedback from disk."""
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r") as f:
                self.store = json.load(f)
            print(f"[FeedbackMemory] Loaded {len(self.store)} past feedback entries.")
        else:
            self.store = []

    def _save(self):
        """Save feedback to disk."""
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.store, f, indent=2)

    # ── Store Feedback ────────────────────────────────────────────────────────

    def store_feedback(
        self,
        query:     str,
        answer:    str,
        tag:       str,              # "good" or "bad"
        chunk_ids: List[str] = None, # RAG chunk IDs used
    ):
        """
        Store a feedback entry.

        Args:
            query     : the user's question
            answer    : the answer that was given
            tag       : "good" (helpful) or "bad" (wrong/hallucinated)
            chunk_ids : list of RAG chunk IDs that were retrieved
        """
        if tag not in ("good", "bad"):
            raise ValueError("Tag must be 'good' or 'bad'")

        # Generate embedding for the query
        embedding = self.embedder.embed_query(query)

        entry = {
            "id":              hashlib.md5(
                                   (query + answer + datetime.now().isoformat())
                                   .encode()
                               ).hexdigest()[:10],
            "query":           query,
            "answer":          answer[:300],   # truncate for storage
            "tag":             tag,
            "chunk_ids":       chunk_ids or [],
            "timestamp":       datetime.now().isoformat(),
            "query_embedding": embedding,      # stored as list of floats
        }

        self.store.append(entry)
        self._save()

        
        print(f"[FeedbackMemory] Stored feedback  ID={entry['id']}  tag={tag}")

    # ── Find Similar Past Feedback ────────────────────────────────────────────

    def find_similar(
        self,
        query:   str,
        top_k:   int  = 3,
        tag_filter: Optional[str] = None,   # "good", "bad", or None for all
    ) -> List[Dict]:
        """
        Find past feedback entries most similar to the current query.
        Uses cosine similarity on stored embeddings.

        Args:
            query      : current user query
            top_k      : number of similar entries to return
            tag_filter : filter by "good" or "bad" (None = return all)

        Returns:
            List of similar feedback entries, sorted by similarity (highest first)
        """
        if not self.store:
            return []

        # Filter by tag if requested
        candidates = [
            e for e in self.store
            if tag_filter is None or e["tag"] == tag_filter
        ]

        if not candidates:
            return []

        # Embed current query
        query_vec = np.array(self.embedder.embed_query(query))

        # Compute cosine similarity with each stored entry
        scored = []
        for entry in candidates:
            stored_vec = np.array(entry["query_embedding"])
            similarity = self._cosine_similarity(query_vec, stored_vec)
            scored.append({**entry, "similarity": round(float(similarity), 4)})

        # Sort by similarity descending
        scored.sort(key=lambda x: x["similarity"], reverse=True)

        return scored[:top_k]

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot   = np.dot(vec_a, vec_b)
        norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return dot / norm if norm > 0 else 0.0

    # ── Chunk Reputation ──────────────────────────────────────────────────────

    def get_chunk_reputation(self) -> Dict[str, Dict]:
        """
        Compute good/bad counts for each chunk ID across all feedback.

        Returns a dict like:
          {
            "chunk_abc123": {"good": 3, "bad": 1, "score": 0.75},
            ...
          }
        Score = good / (good + bad)  → higher = more trusted
        """
        reputation = {}

        for entry in self.store:
            for chunk_id in entry.get("chunk_ids", []):
                if chunk_id not in reputation:
                    reputation[chunk_id] = {"good": 0, "bad": 0}
                reputation[chunk_id][entry["tag"]] += 1

        # Add score
        for chunk_id, counts in reputation.items():
            total = counts["good"] + counts["bad"]
            counts["score"] = counts["good"] / total if total > 0 else 0.5

        return reputation

    # Display --------------------------------

    def show_all(self):
        """Print all stored feedback entries (without embeddings)."""
        if not self.store:
            print("[FeedbackMemory] No feedback stored yet.")
            return

        print(f"\n{'='*60}")
        print(f"  STORED FEEDBACK  ({len(self.store)} entries)")
        print(f"{'='*60}")

        for entry in self.store:
            print(f"\n   ID: {entry['id']}  |  {entry['tag'].upper()}")
            print(f"     Query   : {entry['query']}")
            print(f"     Answer  : {entry['answer'][:100]}...")
            print(f"     Chunks  : {entry['chunk_ids']}")
            print(f"     Time    : {entry['timestamp'][:19]}")

        print(f"\n{'='*60}\n")

    def summary(self):
        """Print a short summary of feedback statistics."""
        good = sum(1 for e in self.store if e["tag"] == "good")
        bad  = sum(1 for e in self.store if e["tag"] == "bad")
        print(f"\n[FeedbackMemory] Summary: {good} good  |  {bad} bad  |  {len(self.store)} total")
