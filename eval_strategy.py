"""
Section 3 — AI Evaluation Strategy
Evaluates the Multi-Agent RAG System across 6 dimensions:

  1. Retrieval Quality        (Agent A)
  2. LLM Response Quality     (Agent C)
  3. SQL Agent Quality        (Agent B)
  4. Orchestration Logic      (Orchestrator)
  5. Hallucination & Grounding
  6. Cost & Latency
"""

import re
import os 
import json
import time
import sqlite3
from datetime import datetime

try:
    from multi_agent_rag import build_pipeline, run_query, VECTOR_STORE
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Pipeline not loaded: {e}")
    print("[WARN] Running in standalone mock mode.\n")
    PIPELINE_AVAILABLE = False

# Silence HF + Transformers noise
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Optional: suppress hub auth warning
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")


# 
# PRINT HELPERS
# 

SEP  = "=" * 65
SEP2 = "-" * 65

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subsection(title):
    print(f"\n  ── {title}")
    print(f"  {SEP2}")

def result(label, value, passed=None):
    print(f"{label}: {value}")


# 
# 1. RETRIEVAL QUALITY  (Agent A)
# 

# Golden test set: query → list of keywords that MUST appear in retrieved chunks
RETRIEVAL_GOLDEN = [
    {
        "query": "remote work policy eligibility",
        "must_contain": ["probation", "remote", "WFH", "eligible"],
        "max_score": 2.0,         # L2 distance threshold for "good" chunk
    },
    {
        "query": "LTA leave travel allowance policy",
        "must_contain": ["LTA", "leave", "travel", "permanent"],
        "max_score": 2.0,
    },
    {
        "query": "data privacy GDPR compliance",
        "must_contain": ["GDPR", "data", "privacy", "compliance"],
        "max_score": 2.0,
    },
]

def evaluate_retrieval():
    section("1. RETRIEVAL QUALITY  (Agent A — RAG Retriever)")

    if not PIPELINE_AVAILABLE:
        print("  [SKIP] Pipeline not available — showing metric definitions only.\n")
        print("  Metrics used:")
        print("    • Precision@K  — % of top-K chunks that are relevant")
        print("    • Score Filter — chunks below L2 threshold are accepted")
        print("    • Keyword Hit  — retrieved text contains expected keywords")
        return

    total, passed_precision, passed_score = 0, 0, 0

    for test in RETRIEVAL_GOLDEN:
        query       = test["query"]
        must_contain = test["must_contain"]
        max_score    = test["max_score"]

        subsection(f"Query: '{query}'")

        # Retrieve top-3 chunks with scores
        results = VECTOR_STORE.similarity_search_with_score(query, k=3)

        relevant_count = 0
        for i, (doc, score) in enumerate(results, 1):
            text = doc.page_content
            keyword_hit = any(kw.lower() in text.lower() for kw in must_contain)
            score_ok    = score <= max_score

            print(f"    Chunk [{i}]  Score={score:.4f}  "
                  f"Keyword={'HIT' if keyword_hit else 'MISS'}  "
                  f"Score={'OK' if score_ok else 'HIGH'}")

            if keyword_hit:
                relevant_count += 1

        precision_at_3 = relevant_count / 3
        avg_score      = sum(s for _, s in results) / len(results)

        result("Precision@3",  f"{precision_at_3:.2f}  ({relevant_count}/3 chunks relevant)",
               precision_at_3 >= 0.5)
        result("Avg L2 Score", f"{avg_score:.4f}  (lower = more similar)",
               avg_score <= max_score)

        total += 1
        if precision_at_3 >= 0.5: passed_precision += 1
        if avg_score <= max_score: passed_score += 1

    print(f"\n  RETRIEVAL SUMMARY: "
          f"{passed_precision}/{total} queries met Precision@3 ≥ 0.5  |  "
          f"{passed_score}/{total} met score threshold")


# 
# 2. LLM RESPONSE QUALITY  (Agent C — Synthesizer)
# 

# Golden answers: query → keywords the answer MUST contain
LLM_GOLDEN = [
    {
        "query":    "What is RAG and why is it useful?",
        "must_contain": ["retrieval", "generation", "LLM", "document"],
        "must_not_contain": ["not available", "cannot determine"],
        "source_expected": "LLM",
    },
    {
        "query":    "How many employees are eligible for LTA as per the latest policy?",
        "must_contain": ["9"],
        "must_not_contain": ["not available", "cannot determine"],
        "source_expected": "DB",
    },
    {
        "query":    "List employees in Engineering and explain the department structure",
        "must_contain": ["Alice", "Bob", "Karen", "Engineering"],
        "must_not_contain": ["not available"],
        "source_expected": "DB",
    },
]

def evaluate_llm_responses():
    section("2. LLM RESPONSE QUALITY  (Agent C — Synthesizer)")

    if not PIPELINE_AVAILABLE:
        print("  [SKIP] Pipeline not available — showing metric definitions only.\n")
        print("  Metrics used:")
        print("    • Keyword Coverage  — answer contains expected facts")
        print("    • Faithfulness      — answer does not contradict sources")
        print("    • No Refusal        — no 'not available' when data exists")
        return

    pipeline = build_pipeline()
    passed = 0

    for test in LLM_GOLDEN:
        query        = test["query"]
        must         = test["must_contain"]
        must_not     = test["must_not_contain"]

        subsection(f"Query: '{query}'")

        state = {
            "user_query": query, "sub_queries": [], "agents_to_run": [],
            "rag_results": [], "sql_results": [], "web_results": [],
            "reason_result": "", "final_answer": "", "execution_log": [],
            "timestamp": datetime.now().isoformat(), "tool_failures": {},
        }
        result_state = pipeline.invoke(state)
        answer = result_state.get("final_answer", "")

        # Check keyword coverage
        hits     = [kw for kw in must     if kw.lower() in answer.lower()]
        misses   = [kw for kw in must_not if kw.lower() in answer.lower()]

        coverage   = len(hits) / len(must)
        faithful   = len(misses) == 0
        test_pass  = coverage >= 0.5 and faithful

        print(f"    Answer preview: {answer[:120].strip()}...")
        result("Keyword coverage", f"{coverage:.0%}  (found: {hits})",   coverage >= 0.5)
        result("Faithfulness",     "PASS — no refusal phrases" if faithful
               else f"FAIL — found: {misses}",                            faithful)
        result("Overall",          "PASS" if test_pass else "FAIL",       test_pass)

        if test_pass:
            passed += 1

    print(f"\n  LLM RESPONSE SUMMARY: {passed}/{len(LLM_GOLDEN)} queries passed")


# 
# 3. SQL AGENT QUALITY  (Agent B)
# 

import os
from pathlib import Path
DB_PATH = Path(__file__).resolve().parent / "company.db"

# Each test: SQL to run + expected result value to find in rows
SQL_TESTS = [
    {
        "name":     "Count LTA-eligible employees",
        "sql":      "SELECT COUNT(*) as count FROM employees WHERE is_permanent=1 AND years_of_service>=1.0",
        "expected": {"count": 9},
        "safe":     True,
    },
    {
        "name":     "List Engineering employees",
        "sql":      """SELECT e.name FROM employees e
                       JOIN departments d ON e.department_id = d.id
                       WHERE d.name = 'Engineering'""",
        "expected_names": ["Alice Johnson", "Bob Smith", "Karen Clark"],
        "safe":     True,
    },
    {
        "name":     "Block DROP TABLE (injection test)",
        "sql":      "DROP TABLE employees",
        "expected": "BLOCKED",
        "safe":     False,
    },
    {
        "name":     "Block multi-statement (injection test)",
        "sql":      "SELECT * FROM employees; DROP TABLE employees",
        "expected": "BLOCKED",
        "safe":     False,
    },
]

def evaluate_sql_agent():
    section("3. SQL AGENT QUALITY  (Agent B)")

    try:
        from multi_agent_rag import sql_query_tool, validate_sql_safety
    except ImportError:
        print("  [SKIP] Could not import sql tools.")
        return

    passed = 0

    for test in SQL_TESTS:
        subsection(test["name"])
        sql = test["sql"].strip()
        print(f"    SQL: {sql[:80]}{'...' if len(sql)>80 else ''}")

        # Safety check first
        is_safe, reason = validate_sql_safety(sql)

        if not test["safe"]:
            # Expect this to be BLOCKED
            blocked = not is_safe
            result("Injection blocked", "YES — " + reason if blocked else "NO — ran when it shouldn't",
                   blocked)
            if blocked:
                passed += 1
            continue

        # Safe query — run it
        if not is_safe:
            result("Safety check", f"FAILED unexpectedly: {reason}", False)
            continue

        rows = sql_query_tool(sql)

        if rows and "error" in rows[0]:
            result("Execution", f"ERROR: {rows[0]['error']}", False)
            continue

        # Validate result
        if "expected" in test:
            expected_val = list(test["expected"].values())[0]
            actual_val   = list(rows[0].values())[0] if rows else None
            match = (actual_val == expected_val)
            result("Result correctness",
                   f"Expected={expected_val}  Got={actual_val}", match)
            if match:
                passed += 1

        elif "expected_names" in test:
            actual_names = [r.get("name", "") for r in rows]
            match = set(test["expected_names"]) == set(actual_names)
            result("Result correctness",
                   f"Expected={test['expected_names']}  Got={actual_names}", match)
            if match:
                passed += 1

    print(f"\n  SQL SUMMARY: {passed}/{len(SQL_TESTS)} tests passed")


# 
# 4. ORCHESTRATION LOGIC
# 

# Each test: query → which agents MUST be selected
ROUTING_TESTS = [
    {
        "query":          "How many employees are eligible for LTA?",
        "must_include":   ["B"],       # Must use SQL
        "must_exclude":   [],
        "description":    "Count query → must route to SQL (Agent B)",
    },
    {
        "query":          "What is RAG and why is it useful?",
        "must_include":   ["E"],       # Must use Reasoning
        "must_exclude":   ["B"],       # Should NOT use SQL
        "description":    "Conceptual question → must route to Reasoning (Agent E)",
    },
    {
        "query":          "Compare our remote work policy with industry best practices",
        "must_include":   ["A"],       # Must use RAG
        "must_exclude":   [],
        "description":    "Policy comparison → must route to RAG (Agent A)",
    },
    {
        "query":          "What is the latest EV adoption trend in the US?",
        "must_include":   ["D"],       # Must use Web search
        "must_exclude":   [],
        "description":    "External/market question → must route to Web (Agent D)",
    },
]

def evaluate_orchestration():
    section("4. ORCHESTRATION LOGIC")

    if not PIPELINE_AVAILABLE:
        print("  [SKIP] Pipeline not available — showing test cases only.\n")
        for t in ROUTING_TESTS:
            print(f"  Query: '{t['query']}'")
            print(f"    Expected: must_include={t['must_include']}  "
                  f"must_exclude={t['must_exclude']}")
            print(f"    Reason: {t['description']}\n")
        return

    pipeline = build_pipeline()
    passed = 0

    for test in ROUTING_TESTS:
        query = test["query"]
        subsection(f"Query: '{query}'")
        print(f"    Expectation: {test['description']}")

        state = {
            "user_query": query, "sub_queries": [], "agents_to_run": [],
            "rag_results": [], "sql_results": [], "web_results": [],
            "reason_result": "", "final_answer": "", "execution_log": [],
            "timestamp": datetime.now().isoformat(), "tool_failures": {},
        }
        result_state = pipeline.invoke(state)
        agents_used = result_state.get("agents_to_run", [])

        # Check must_include
        include_ok = all(a in agents_used for a in test["must_include"])
        # Check must_exclude
        exclude_ok = all(a not in agents_used for a in test["must_exclude"])
        test_pass  = include_ok and exclude_ok

        result("Agents selected",  str(agents_used))
        result("Must include",     f"{test['must_include']} → {'OK' if include_ok else 'MISSING'}", include_ok)
        if test["must_exclude"]:
            result("Must exclude", f"{test['must_exclude']} → {'OK' if exclude_ok else 'WRONGLY INCLUDED'}", exclude_ok)
        result("Routing decision", "CORRECT" if test_pass else "WRONG", test_pass)

        if test_pass:
            passed += 1

    print(f"\n  ORCHESTRATION SUMMARY: {passed}/{len(ROUTING_TESTS)} routing decisions correct")


# 
# 5. HALLUCINATION & GROUNDING
# 

# Test cases: answer + context → check grounding
HALLUCINATION_TESTS = [
    {
        "name":    "DB answer matches SQL result",
        "answer":  "There are 9 employees eligible for LTA. [DB]",
        "sql":     [{"COUNT(*)": 9}],
        "rag":     [],
        "check":   "numeric_match",
        "expected_number": 9,
    },
    {
        "name":    "Phantom chunk citation detected",
        "answer":  "Policy says X [CHUNK_abc999]",
        "sql":     [],
        "rag":     [{"chunk_id": "realchunk1", "text": "some text", "source": "doc", "score": 1.0}],
        "check":   "phantom_chunk",
    },
    {
        "name":    "No citation when LLM-only answer",
        "answer":  "RAG stands for Retrieval Augmented Generation. [LLM]",
        "sql":     [],
        "rag":     [],
        "check":   "citation_present",
    },
    {
        "name":    "Refusal despite DB data (faithfulness failure)",
        "answer":  "The information is not available in sources.",
        "sql":     [{"COUNT(*)": 9}],
        "rag":     [],
        "check":   "refusal_with_data",
    },
]

def check_numeric_match(answer, sql_rows, expected_number):
    """Check if number in answer matches DB result."""
    numbers_in_answer = re.findall(r'\b\d+\b', answer)
    return str(expected_number) in numbers_in_answer

def check_phantom_chunks(answer, rag_results):
    """Check if answer cites chunk IDs that weren't actually retrieved."""
    cited_ids   = set(re.findall(r'\[CHUNK_([^\]]+)\]', answer))
    real_ids    = set(r["chunk_id"] for r in rag_results)
    phantom_ids = cited_ids - real_ids
    return phantom_ids   # empty set = no phantoms

def check_refusal_with_data(answer, sql_rows, rag_results):
    """Detect if answer says 'not available' despite having data."""
    refusal_phrases = ["not available", "cannot determine", "no information"]
    has_refusal = any(p in answer.lower() for p in refusal_phrases)
    has_data    = bool(sql_rows or rag_results)
    return has_refusal and has_data  # True = hallucination failure

def evaluate_hallucination():
    section("5. HALLUCINATION & GROUNDING")

    passed = 0

    for test in HALLUCINATION_TESTS:
        subsection(test["name"])

        answer = test["answer"]
        sql    = test.get("sql", [])
        rag    = test.get("rag", [])
        check  = test["check"]

        print(f"    Answer: \"{answer}\"")

        if check == "numeric_match":
            expected = test["expected_number"]
            match    = check_numeric_match(answer, sql, expected)
            result("Number in answer matches DB", f"Expected {expected}", match)
            if match: passed += 1

        elif check == "phantom_chunk":
            phantoms = check_phantom_chunks(answer, rag)
            no_phantom = len(phantoms) == 0
            result("Phantom chunk IDs",
                   "None detected ✓" if no_phantom else f"PHANTOM IDs: {phantoms}",
                   no_phantom)
            if no_phantom: passed += 1

        elif check == "citation_present":
            has_citation = bool(re.search(r'\[(DB|CHUNK_|Web|LLM)', answer))
            result("Citation present in answer", "YES" if has_citation else "NO",
                   has_citation)
            if has_citation: passed += 1

        elif check == "refusal_with_data":
            is_failure = check_refusal_with_data(answer, sql, rag)
            result("Faithfulness (no refusal with data)",
                   "FAIL — model refused despite DB data" if is_failure else "PASS",
                   not is_failure)
            if not is_failure: passed += 1

    print(f"\n  HALLUCINATION SUMMARY: {passed}/{len(HALLUCINATION_TESTS)} grounding checks passed")


# 6. COST & LATENCY

LATENCY_QUERIES = [
    "How many employees are eligible for LTA?",
    "What is RAG and why is it useful?",
    "List employees in Engineering department",
]

# Rough token estimation (4 chars ≈ 1 token)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# Groq pricing (as of 2024) for allam-2-7b — approximate
COST_PER_1K_INPUT_TOKENS  = 0.00007   # $0.07 per 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.00007

def evaluate_cost_latency():
    section("6. COST OPTIMIZATION & LATENCY")

    if not PIPELINE_AVAILABLE:
        print("  [SKIP] Pipeline not available — showing metric definitions only.\n")
        print("  Metrics tracked:")
        print("    • Wall-clock time per query (seconds)")
        print("    • Estimated input + output tokens")
        print("    • Estimated API cost per query (USD)")
        print("    • Latency target: < 10 seconds per query")
        return

    pipeline = build_pipeline()
    total_cost    = 0.0
    total_latency = 0.0

    for query in LATENCY_QUERIES:
        subsection(f"Query: '{query}'")

        state = {
            "user_query": query, "sub_queries": [], "agents_to_run": [],
            "rag_results": [], "sql_results": [], "web_results": [],
            "reason_result": "", "final_answer": "", "execution_log": [],
            "timestamp": datetime.now().isoformat(), "tool_failures": {},
        }

        start = time.perf_counter()
        result_state = pipeline.invoke(state)
        elapsed = time.perf_counter() - start

        answer       = result_state.get("final_answer", "")
        input_tokens = estimate_tokens(query) + 500   # query + prompt overhead
        output_tokens = estimate_tokens(answer)

        cost = (input_tokens  / 1000 * COST_PER_1K_INPUT_TOKENS +
                output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)

        total_cost    += cost
        total_latency += elapsed

        result("Latency",       f"{elapsed:.2f}s",              elapsed < 10.0)
        result("Input tokens",  f"~{input_tokens}")
        result("Output tokens", f"~{output_tokens}")
        result("Est. cost",     f"${cost:.6f} USD")

    print(f"\n  COST & LATENCY SUMMARY:")
    print(f"    Total latency : {total_latency:.2f}s  "
          f"({'OK' if total_latency/len(LATENCY_QUERIES) < 10 else 'SLOW'} avg)")
    print(f"    Total est. cost : ${total_cost:.5f} USD  "
          f"for {len(LATENCY_QUERIES)} queries")

    # Optimization tips printed as part of evaluation output
    print("\n  OPTIMIZATION STRATEGIES APPLIED IN THIS SYSTEM:")
    print("     Parallel agent execution (ThreadPoolExecutor)")
    print("     RAG context compression (max_chars=2000)")
    print("     SQL cohesion guard (skips LLM decomposition for simple DB queries)")
    print("     Relevance threshold (skips irrelevant chunks early)")
    print("     Retry decorator (max 2 retries, avoids cascading failures)")

# MAIN — RUN ALL EVALUATIONS

def main():
    print("\n" + "=" * 65)
    print("  AI EVALUATION STRATEGY — SECTION 3")
    print("  Multi-Agent RAG System")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    evaluate_retrieval()
    evaluate_llm_responses()
    evaluate_sql_agent()
    evaluate_orchestration()
    evaluate_hallucination()
    evaluate_cost_latency()

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
