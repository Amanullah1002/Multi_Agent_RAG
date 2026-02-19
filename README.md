# Multi-Agent RAG System

A production-grade multi-agent pipeline that combines **Retrieval-Augmented Generation (RAG)**, **SQL database querying**, **web search**, and **LLM reasoning** to answer complex HR and company policy questions.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         ORCHESTRATOR                │
│  • Decomposes multi-intent queries  │
│  • Routes to correct agent(s)       │
│  • Runs agents in parallel          │
└──────┬──────┬──────┬────────┬───────┘
       │      │      │        │
       ▼      ▼      ▼        ▼
  ┌────────┐ ┌────┐ ┌────┐ ┌──────┐
  │Agent A │ │ B  │ │ D  │ │  E   │
  │  RAG   │ │SQL │ │Web │ │ LLM  │
  └────────┘ └────┘ └────┘ └──────┘
       │      │      │        │
       └──────┴──────┴────────┘
                    │
                    ▼
          ┌──────────────────┐
          │    Agent C       │
          │  Synthesizer     │
          │ (cited answer)   │
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  Feedback Memory │  ← Section 4
          │ (good/bad tags)  │
          └──────────────────┘
```

---

## Project Structure

```
multi_agent_rag/
├── multi_agent_rag.py          # Main pipeline (Sections 1)
├── eval_strategy.py            # Evaluation framework (Section 3)
├── feedback_demo.py            # Feedback memory demo (Section 4)
├── requirements.txt
├── .env                        # GROQ_API_KEY + TAVILY_API_KEY goes here
│
├── documents/                  # Source documents (Agent A reads these)
│   ├── doc1_hr_leave_policy.txt
│   ├── doc2_remote_work_policy.txt
│   ├── doc3_employee_benefits.txt
│   ├── doc4_performance_review.txt
│   ├── doc5_expense_policy.txt
│   ├── doc6_onboarding_process.txt
│   └── doc7_data_privacy_policy.txt
│
├── feedback/                   # Section 4 — Feedback Memory
│   ├── __init__.py
│   ├── feedback_memory.py      # Store + retrieve feedback
│   └── feedback_reranker.py    # Re-rank RAG with feedback scores
│
├── execution_logs/             # Auto-generated per query
│   └── sample_execution_log.json
│
├── faiss_index/                # Auto-generated vector index
└── company.db                  # SQLite database (Agent B reads this)
```

---

## Agents

| Agent | Role | Tool |
|-------|------|------|
| **A — RAG Retriever** | Retrieves relevant document chunks using FAISS vector search | `retrieve_docs()` |
| **B — SQL Agent** | Queries employee database with safe, validated SQL | `sql_query_tool()` |
| **C — Synthesizer** | Merges all agent outputs into a grounded, cited answer | LLM (Groq) |
| **D — Web Search** | Retrieves real-time external and industry knowledge for benchmarking and comparisons | `web_search() · Tavily Search API` |
| **E — Reasoning** | Uses LLM prior knowledge for conceptual questions | `llm_reason()` |

---

## Orchestration Logic

The orchestrator decides which agents to invoke based on the query:

| Query Type | Agents Selected | Example |
|-----------|----------------|---------|
| Policy + count | A + B | "How many employees eligible for LTA?" |
| Conceptual | E | "What is RAG?" |
| Policy comparison | A + D | "Compare our WFH policy with industry trends" |
| DB listing | B | "List employees in Engineering" |

Routing uses `heuristic_fallback()` — a keyword-based rule set that is fast, reliable, and avoids LLM hallucination in routing decisions.

---

## Section 3 — Evaluation Strategy

Six dimensions evaluated (`eval_strategy.py`):

## Evaluation Framework

To ensure production reliability and answer quality, the system is evaluated across six key dimensions covering retrieval, reasoning, orchestration, grounding, and performance.

---

### 1. Retrieval Quality (RAG)

**Evaluation Methods**

- **Precision@K** — Measures relevance of top retrieved chunks.
- **Similarity Threshold Filtering** — Low-relevance chunks are discarded using L2 distance thresholds.
- **Minimum Chunk Requirement** — Ensures sufficient grounding context exists before synthesis.
- **Feedback Reranking** — Historical user feedback adjusts retrieval ranking scores.

**Purpose**

Ensures only relevant document evidence is passed to the LLM, reducing noise and hallucination risk.

---

### 2. LLM Response Quality

**Evaluation Methods**

- Keyword coverage vs query intent
- Faithfulness to retrieved evidence
- Answer completeness validation
- Output length and clarity constraints
- Guardrail prompts for acronym/domain disambiguation

**Purpose**

Maintains answer accuracy, conciseness, and alignment with grounded data sources.

---

### 3. SQL Agent Task Accuracy

**Validation Mechanisms**

- Schema whitelist enforcement
- SELECT-only query restriction
- SQL injection keyword blocking
- Auto-correction of hallucinated table names
- Execution success and row-count validation

**Purpose**

Ensures database interactions remain secure, valid, and factually correct.

---

### 4. Agent Orchestration Logic

**Evaluation Criteria**

- Routing accuracy against a golden query test set
- Multi-intent query decomposition correctness
- Parallel execution reliability
- Fallback routing behavior on tool failure

**Design Choice**

Heuristic routing is used instead of LLM routing to prevent hallucinated tool selection and reduce latency/cost.

---

### 5. Hallucination & Grounding Control

**Guardrails Implemented**

- Citation enforcement using chunk IDs
- Phantom citation detection and removal
- Grounding validation before synthesis
- Tool-priority context ordering:



---

## Section 4 — Feedback Memory

After each answer, users rate it **good** or **bad**.

- `feedback_memory.py` — Stores feedback with query embeddings to disk (`feedback_store.json`)
- `feedback_reranker.py` — Re-ranks future RAG retrieval: `score = 0.7 × retrieval_sim + 0.3 × reputation`
- `find_similar()` — Uses cosine similarity to warn when a similar query previously got a bad answer

Run the standalone demo:
```bash
python feedback_demo.py
```

---

## Setup

### 1. Clone and install
```bash
git clone 
cd multi-agent-rag
pip install -r requirements.txt
```

### 2. Set API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
Or export it:
```bash
export GROQ_API_KEY=your_groq_api_key_here
export TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Ensure files are in place
- `company.db` — SQLite database with employees/departments tables
- `documents/` — 7 HR policy `.txt` files (included)

### 4. Run
```bash
# Interactive CLI (main system)
python multi_agent_rag.py

# Evaluation suite
python eval_strategy.py

# Feedback memory demo
python feedback_demo.py
```

---

## Demo Questions

### Tests Agent A (RAG)
```
What is the remote work policy?
What are the maternity leave rules?
How many days of sick leave do employees get?
What is the expense approval process for amounts above $2000?
```

### Tests Agent B (SQL)
```
List all employees in the Engineering department
What is the average salary of permanent employees?
How many departments does the company have?
```

### Tests Agent A + B (RAG + SQL — most important)
```
How many employees are eligible for LTA as per the policy?
List employees in Engineering and explain the department structure
```

### Tests Agent D (Web Search)
```
Compare our remote work policy with industry best practices for 2024
What are the latest EV adoption trends in the US?
```

### Tests Agent E (Reasoning)
```
What is RAG and why is it useful?
Explain what cosine similarity is
What is a vector database?
```

### Tests Multi-Agent (A + B + D)
```
Compare our LTA policy with global industry standards
How does our health insurance compare to market benchmarks?
```

---

## Database Schema

```sql
employees(id, name, department_id, years_of_service, is_permanent, basic_salary)
departments(id, name)
leave_requests(id, employee_id, leave_type, status, year)
```

**LTA eligibility rule:** `is_permanent = 1 AND years_of_service >= 1.0`

---

## Key Design Decisions

- **Parallel execution** via `ThreadPoolExecutor` — all agents run simultaneously
- **Heuristic routing** instead of LLM routing — faster, more reliable, no hallucinated agent names
- **Score thresholding** — RAG chunks below L2=5.0 are filtered out before synthesis
- **SQL injection protection** — keyword blocklist + SELECT-only enforcement + schema whitelist
- **Citation enforcement** — synthesizer receives pre-formatted DB/chunk data and explicit chunk ID whitelist
- **Feedback loop** — good/bad ratings adjust future chunk ranking (70% retrieval + 30% reputation)

---

## Assumptions & Limitations

- Web search uses `Tavily API` for real-time external grounding. An active `TAVILY_API_KEY` is required. Results depend on external search availability and API limits
- LLM used: `openai/gpt-oss-120b` via Groq API — smaller model, may need stronger model for complex synthesis
- FAISS index is rebuilt on first run; subsequent runs use cached index
- Feedback is stored in a local JSON file; production would use a database
- SQL agent only handles `SELECT` queries — no write operations allowed


# Section 2

## Dependency Resolution Approach

This solution uses **Kahn’s Algorithm (BFS-based Topological Sorting)** to determine a valid execution order of tasks based on their dependencies.

### Why Kahn’s Algorithm?

- Efficient for Directed Acyclic Graphs (DAGs)
- Runs in **O(V + E)** time complexity
- Naturally detects cycles
- Produces a valid dependency-safe execution order

### How It Works

1. Build a directed graph from task dependencies.
2. Compute in-degree (number of incoming edges) for each task.
3. Initialize a queue with tasks having in-degree = 0 (no dependencies).
4. Iteratively:
   - Remove a task from the queue
   - Add it to the execution order
   - Reduce in-degree of its dependent tasks
5. If not all tasks are processed, a cycle exists.

### Guarantees

- Returns a valid topological order if possible.
- Raises an error if cyclic dependencies are detected.
- Handles multiple independent task chains.
