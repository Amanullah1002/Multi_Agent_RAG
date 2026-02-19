"""
feedback_demo.py
=================
Section 4 — Feedback Memory  DEMO

Run this file to see the full feedback loop:
  1. Simulate pipeline answers
  2. Store good/bad feedback
  3. Show similar past feedback retrieval
  4. Show chunk reputation table
  5. Show reranking in action

Run:
  python feedback_demo.py
"""

import sys
import os
# Silence HF + Transformers noise
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Optional: suppress hub auth warning
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback.feedback_memory   import FeedbackMemory
from feedback.feedback_reranker import rerank_with_feedback

SEP  = "=" * 60
SEP2 = "-" * 60

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ------------------------------------------------------------------------------
# STEP 1 — Store some feedback (simulating past user interactions)
# ------------------------------------------------------------------------------

header("STEP 1 — Storing User Feedback")

fm = FeedbackMemory()

# Simulate: user liked this answer
fm.store_feedback(
    query     = "How many employees are eligible for LTA?",
    answer    = "9 employees are eligible for LTA based on is_permanent=1 and years_of_service>=1.",
    tag       = "good",
    chunk_ids = ["86803cedeced", "6691d9b3cb89"],
)

# Simulate: user liked this answer too
fm.store_feedback(
    query     = "What is the LTA leave travel policy?",
    answer    = "LTA is available for permanent employees with 1+ year of service.",
    tag       = "good",
    chunk_ids = ["86803cedeced"],   # same chunk used again → builds reputation
)

# Simulate: user disliked this answer (hallucination happened)
fm.store_feedback(
    query     = "What is the maximum LTA amount?",
    answer    = "The maximum LTA is not available in sources.",
    tag       = "bad",
    chunk_ids = ["a33aaa15b07d"],   # this chunk gets a bad reputation
)

# Simulate: user liked RAG answer
fm.store_feedback(
    query     = "What is the remote work policy?",
    answer    = "Employees with 6 months probation can WFH up to 2 days/week.",
    tag       = "good",
    chunk_ids = ["86803cedeced", "6691d9b3cb89"],
)

# Simulate: another bad answer
fm.store_feedback(
    query     = "Explain department structure",
    answer    = "Cannot determine from available sources.",
    tag       = "bad",
    chunk_ids = ["71d049b87021"],
)

print("\n 5 feedback entries stored.")


# ------------------------------------------------------------------------------
# STEP 2 — Show all stored feedback
# ------------------------------------------------------------------------------

header("STEP 2 — All Stored Feedback")
fm.show_all()
fm.summary()


# ------------------------------------------------------------------------------
# STEP 3 — Find similar past feedback using embedding similarity
# ------------------------------------------------------------------------------

header("STEP 3 — Find Similar Past Feedback (Embedding Similarity)")

test_query = "How many staff qualify for LTA benefit?"

print(f"\n  Current query : \"{test_query}\"")
print(f"\n  Finding similar past feedback...\n")

similar = fm.find_similar(test_query, top_k=3)

if similar:
    for i, entry in enumerate(similar, 1):
        
        print(f"  [{i}] Similarity={entry['similarity']:.4f} {entry['tag'].upper()}")
        print(f"       Past query : \"{entry['query']}\"")
        print(f"       Past answer: \"{entry['answer'][:80]}...\"")
        print()
else:
    print("  No similar past feedback found.")

# Also show only BAD feedback similar to query
print(f"  Checking for similar BAD feedback (hallucination warnings)...")
bad_similar = fm.find_similar(test_query, top_k=2, tag_filter="bad")

if bad_similar:
    for entry in bad_similar:
        print(f" Similar BAD answer found (sim={entry['similarity']:.4f}): "
              f"\"{entry['answer'][:80]}\"")
else:
    print("  No similar bad feedback found — safe to proceed.")


# ------------------------------------------------------------------------------
# STEP 4 — Chunk Reputation Table
# ------------------------------------------------------------------------------

header("STEP 4 — Chunk Reputation Table")

reputation = fm.get_chunk_reputation()

print(f"\n  {'Chunk ID':<20} {'Good':>6} {'Bad':>6} {'Score':>8}  {'Status'}")
print(f"  {SEP2}")

for chunk_id, counts in sorted(reputation.items(), key=lambda x: -x[1]["score"]):
    score  = counts["score"]
    status = "TRUSTED" if score >= 0.7 else (" DEMOTE" if score < 0.4 else "NEUTRAL")
    print(f"  {chunk_id:<20} {counts['good']:>6} {counts['bad']:>6} {score:>8.2f}  {status}")

print()
print("  Score = good_count / (good_count + bad_count)")
print("  ≥ 0.7 → chunk is trusted  |  < 0.4 → chunk is demoted in future retrieval")


# ------------------------------------------------------------------------------
# STEP 5 — Re-ranking in Action
# ------------------------------------------------------------------------------

header("STEP 5 — Re-ranking RAG Chunks Using Feedback")

# Simulated retrieved chunks (as if returned by FAISS)
mock_chunks = [
    {
        "chunk_id": "a33aaa15b07d",   # has BAD reputation
        "text":     "Onboarding process week 1 orientation...",
        "source":   "Onboarding Doc",
        "score":    1.1,              # good FAISS score (low L2)
    },
    {
        "chunk_id": "86803cedeced",   # has GOOD reputation (used in 3 good answers)
        "text":     "Remote Work Policy: employees with 6 months probation...",
        "source":   "Remote Work Policy",
        "score":    1.4,              # slightly worse FAISS score
    },
    {
        "chunk_id": "71d049b87021",   # has BAD reputation
        "text":     "Data Privacy GDPR compliance guidelines...",
        "source":   "Data Privacy Policy",
        "score":    1.5,
    },
    {
        "chunk_id": "6691d9b3cb89",   # has GOOD reputation
        "text":     "Equipment and connectivity for remote workers...",
        "source":   "Remote Work Policy",
        "score":    1.6,
    },
]

print("\n  BEFORE reranking (original FAISS order by L2 distance):")
print(f"  {'#':<4} {'Chunk ID':<20} {'L2 Score':>10}  Source")
print(f"  {SEP2}")
for i, c in enumerate(mock_chunks, 1):
    print(f"  {i:<4} {c['chunk_id']:<20} {c['score']:>10.4f}  {c['source']}")

# Apply feedback reranking
reranked = rerank_with_feedback("LTA policy eligibility", mock_chunks)

print("\n  AFTER reranking (feedback-adjusted workflow order):")
print(f"  {'#':<4} {'Chunk ID':<20} {'Retrieval':>10} {'Reputation':>11} {'Combined':>9}  Source")
print(f"  {SEP2}")
for i, c in enumerate(reranked, 1):
    print(f"  {i:<4} {c['chunk_id']:<20} "
          f"{c['retrieval_sim']:>10.4f} "
          f"{c['rep_score']:>11.4f} "
          f"{c['combined_score']:>9.4f}  "
          f"{c['source']}")

print()
print("   Chunks with good feedback history are ranked higher.")
print("   Chunks with bad feedback history are demoted.")
print()
print("  Combined score = (0.7 × retrieval_similarity) + (0.3 × reputation_score)")


# ------------------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------------------

header("SECTION 4 — SUMMARY")

print("""
  What was built:

  feedback_memory.py
  ├── store_feedback()      Store good/bad tags per query+answer
  ├── find_similar()        Embedding cosine similarity search
  ├── get_chunk_reputation() Count good/bad per chunk ID
  └── show_all() / summary() Display stored feedback

  feedback_reranker.py
  └── rerank_with_feedback() Adjust chunk order using reputation
                             Combined = 70% retrieval + 30% reputation

  Integration point:
  └── retrieve_docs() in multi_agent_rag.py already calls
      rerank_with_feedback() after FAISS retrieval (see Section 4 block)

  Feedback loop:
    User query
      → Pipeline runs
        → Answer shown to user
          → User rates: good/ bad
            → store_feedback() saves it
              → Next similar query
                → find_similar() warns if similar bad answer seen before
                → rerank_with_feedback() boosts trusted chunks
""")

print(SEP)
print("  DEMO COMPLETE")
print(SEP + "\n")