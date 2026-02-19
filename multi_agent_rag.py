# ----------------------------  Standard library ----------------------------
import os
# Silence HF + Transformers noise
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Optional: suppress hub auth warning
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import re
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from time import sleep

from dotenv import load_dotenv
load_dotenv()

import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from tavily import TavilyClient
# ---------------------------- LangChain / LangGraph -------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)



# ---------------------------Absolute project path ------------------------------
from pathlib import Path

# Project root = folder where this file exists
ROOT_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------------------
# 0.  CONFIGURATION
# -------------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in .env file."
    )

DOCS_DIR    = ROOT_DIR / "documents"
DB_PATH     = ROOT_DIR / "company.db"
FAISS_INDEX = ROOT_DIR / "faiss_index"

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# RAG tuning
CHUNK_SIZE           = 600
CHUNK_OVERLAP        = 120
RELEVANCE_THRESHOLD  = 5   # FAISS L2 distance threshold (lower = more similar)
MIN_RELEVANT_CHUNKS  = 1     # Minimum chunks that must pass threshold

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 1  # seconds

# Parallelism
MAX_WORKERS = 4

# SQL Schema whitelist
ALLOWED_TABLES = {"employees", "departments", "leave_requests"}
ALLOWED_COLUMNS = {
    "employees": {"id", "name", "department_id", "years_of_service", "is_permanent", "basic_salary"},
    "departments": {"id", "name"},
    "leave_requests": {"id", "employee_id", "leave_type", "status", "year"},
}

# Configure structured logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MultiAgentRAG")

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found. Web search will fail.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# LLM shared across agents
llm = ChatGroq(
    api_key     = GROQ_API_KEY,
    model       = "allam-2-7b",
    # model       = "llama-3.1-8b-instant",
    temperature = 0,
)

def deduplicate_logs(logs):
    seen = set()
    unique = []

    for l in logs:
        key = (l.get("node"), l.get("action"))

        if key not in seen:
            unique.append(l)
            seen.add(key)

    return unique

# ------------------------------------------------------------------------------
# 1.  PIPELINE STATE
# ------------------------------------------------------------------------------

class PipelineState(TypedDict):
    user_query:       str
    sub_queries:      List[Dict[str, Any]]  # decomposed queries
    agents_to_run:    List[str]
    rag_results:      List[Dict[str, Any]]
    sql_results:      List[Dict[str, Any]]
    web_results:      List[str]
    reason_result:    str
    final_answer:     str
    execution_log:    List[Dict[str, Any]]
    timestamp:        str
    tool_failures:    Dict[str, int]  # track failures per tool


# ------------------------------------------------------------------------------
# 2.  RETRY DECORATOR FOR TOOL FAILURE RECOVERY
# ------------------------------------------------------------------------------

def retry_on_failure(max_attempts: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator to retry tool calls on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying...")
                        sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            # Return safe fallback instead of crashing
            return {"error": str(last_exception), "retries_exhausted": True}
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# 3.  VECTOR STORE WITH METADATA
# ------------------------------------------------------------------------------

def chunk_id(text: str) -> str:
    """Generate deterministic chunk ID."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def build_or_load_vector_store(docs_dir: Path, force_rebuild: bool = False) -> FAISS:
    """
    Build or load vector store from individual document files.
    Each .txt file in docs_dir is treated as one document.
    Source name is taken from the filename.
    """
    index_path = Path(FAISS_INDEX)

    if index_path.exists() and not force_rebuild:
        logger.info("Loading existing FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded {vector_store.index.ntotal} vectors")
        return vector_store

    logger.info(f"Building new vector store from: {docs_dir}/")

    doc_files = sorted(Path(docs_dir).glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " "],
    )

    all_chunks = []

    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Use filename (without extension) as source name

        filename = doc_file.stem  # e.g. doc1_hr_leave_policy

        # Try to extract title from first line of file
        first_line = raw_text.strip().splitlines()[0]
        match = re.search(r"DOCUMENT \d+ .{1,3} (.+)", first_line)
        source = match.group(1).strip() if match else filename

        chunks = splitter.split_text(raw_text)
        logger.info(f"  {doc_file.name} -> {len(chunks)} chunks  (source: {source})")

        char_pos = 0
        for chunk_text in chunks:
            metadata = {
                "chunk_id":  chunk_id(chunk_text),
                "source":    source,
                "file":      doc_file.name,
                "char_start": char_pos,
                "char_end":   char_pos + len(chunk_text),
                "length":     len(chunk_text),
            }
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
            char_pos += len(chunk_text)

    logger.info(f"Total chunks created: {len(all_chunks)}")

    embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local(str(index_path))
    logger.info(f"Saved FAISS index to {index_path}/")

    return vector_store


VECTOR_STORE = build_or_load_vector_store(DOCS_DIR, force_rebuild=True)


# Addon with Section 4 feedback feature
@retry_on_failure(max_attempts=2)
def retrieve_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    RAG retrieval with:
    - Relevance filtering
    - Feedback reranking (Section 4)
    """


    # VECTOR RETRIEVAL
    results = VECTOR_STORE.similarity_search_with_score(
        query,
        k=top_k
    )

    # THRESHOLD FILTERING
    relevant = [
        {
            "text":     doc.page_content,
            "source":   doc.metadata.get("source", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score":    float(score),
        }
        for doc, score in results
        if score <= RELEVANCE_THRESHOLD
    ]

    # LOW RELEVANCE WARNING
    if len(relevant) < MIN_RELEVANT_CHUNKS:

        logger.warning(
            f"Only {len(relevant)}/{top_k} chunks passed "
            f"threshold ({RELEVANCE_THRESHOLD})."
        )

    if not relevant:
        return []

    # FEEDBACK MEMORY RERANKING
    try:# Lazy importing
        from feedback.feedback_reranker import rerank_with_feedback
        reranked = rerank_with_feedback(
            query,
            relevant
        )

        logger.info(
            f"Reranked chunk order: "
            f"{[c['chunk_id'] for c in reranked]}"
        )

        relevant = reranked

    except Exception as e:

        logger.warning(
            f"Feedback reranking skipped: {e}"
        )

    # -------------------------------------
    # RETURN FINAL CHUNKS
    # -------------------------------------
    return relevant


#   SQL Schema Validation 

SQL_INJECTION_KEYWORDS = [
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE",
    "EXEC", "EXECUTE", "UNION", "--", "/*", "*/", "xp_", "sp_"
]

DB_SCHEMA = """
Tables:
  employees(id, name, department_id, years_of_service, is_permanent, basic_salary)
  departments(id, name)
  leave_requests(id, employee_id, leave_type, status, year)

Notes:
  - is_permanent=1 means permanent employee
  - LTA eligibility: is_permanent=1 AND years_of_service >= 1.0
"""

from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword


def _extract_from_token(token):
    """
    Extract table names from Identifier / IdentifierList.
    Removes aliases automatically.
    """
    tables = set()

    if isinstance(token, IdentifierList):
        for identifier in token.get_identifiers():
            tables.add(identifier.get_real_name())

    elif isinstance(token, Identifier):
        tables.add(token.get_real_name())

    return tables


def extract_tables(sql: str):
    """
    Extract real table names from SQL.
    Handles aliases + JOINs safely.
    """

    parsed = sqlparse.parse(sql)

    if not parsed:
        return set()

    stmt = parsed[0]

    tables = set()
    from_seen = False

    for token in stmt.tokens:

        # Detect FROM clause
        if from_seen:

            if token.ttype is Keyword:
                from_seen = False
                continue

            tables |= _extract_from_token(token)

        if token.ttype is Keyword and token.value.upper() == "FROM":
            from_seen = True

        # Handle JOIN tables
        if token.ttype is Keyword and "JOIN" in token.value.upper():
            next_token = stmt.token_next(stmt.token_index(token))
            if next_token:
                tables |= _extract_from_token(next_token)

    return {t for t in tables if t}

ALLOWED_TABLES = {
    "employees",
    "departments",
    "leave_requests"
}


def validate_sql_schema(sql: str) -> Tuple[bool, str]:
    """
    Robust schema validation using SQL parser.
    Avoids false blocking on aliases, COUNT(*), etc.
    """
    try:
        tables = extract_tables(sql)

        invalid_tables = tables - ALLOWED_TABLES

        if invalid_tables:
            return False, f"Invalid tables: {invalid_tables}"

        return True, ""

    except Exception as e:
        return False, f"SQL parse failure: {e}"
    

import re

def sanitize_sql(sql: str) -> str:
    """
    Remove SQL comments + normalize whitespace.
    Prevent false injection blocking.
    """

    if not sql:
        return sql

    # Remove block comments /* ... */
    sql = re.sub(
        r"/\*.*?\*/",
        "",
        sql,
        flags=re.DOTALL
    )

    # Remove inline comments --
    sql = re.sub(
        r"--.*?$",
        "",
        sql,
        flags=re.MULTILINE
    )

    # Normalize whitespace
    sql = " ".join(sql.split())

    return sql.strip()


def validate_sql_safety(sql: str) -> Tuple[bool, str]:
    """
    Enforce SELECT-only + safe schema usage.
    Now with comment sanitization.
    """

    if not sql:
        return False, "Empty SQL query"

    # -------------------------------------
    # SANITIZE SQL FIRST
    # -------------------------------------
    clean_sql = sanitize_sql(sql)
    clean_sql = autocorrect_table_names(clean_sql)

    sql_upper = clean_sql.upper()

    # -------------------------------------
    # SELECT-ONLY ENFORCEMENT
    # -------------------------------------
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries allowed"

    # -------------------------------------
    # BLOCK DANGEROUS KEYWORDS
    # -------------------------------------
    for keyword in SQL_INJECTION_KEYWORDS:

        if keyword in sql_upper:

            return False, f"Blocked keyword: {keyword}"

    # -------------------------------------
    # MULTI-STATEMENT BLOCK
    # -------------------------------------
    if clean_sql.count(";") > 1:
        return False, "Multiple statements not allowed"

    # -------------------------------------
    # SCHEMA VALIDATION
    # -------------------------------------
    schema_valid, schema_error = validate_sql_schema(clean_sql)

    if not schema_valid:
        return False, schema_error

    return True, ""

############
def extract_sql_only(text: str) -> str:
    """
    Extract ONLY SQL from LLM output.
    Removes explanations.
    """

    match = re.search(
        r"(SELECT[\s\S]+?;)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return text.strip()
###############

@retry_on_failure(max_attempts=2)
def sql_query_tool(sql: str) -> List[Dict[str, Any]]:
    """Execute SQL with full safety + schema validation."""

    if not sql:
        return [{"error": "Empty SQL"}]

    # Sanitize comments / whitespace
    clean_sql = sanitize_sql(sql)

    # AUTOCORRECT TABLE NAMES  ← ADD HERE
    clean_sql = autocorrect_table_names(clean_sql)

    logger.info(f"🧹 Sanitized SQL:\n{clean_sql}")

    # Validate safety + schema
    is_safe, error = validate_sql_safety(clean_sql)

    if not is_safe:
        logger.warning(f" SQL blocked: {error}")
        return [{"error": f"Security/Schema Error: {error}"}]
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        logger.info(f"Sanitized SQL:\n{clean_sql}")

        cur.execute(clean_sql)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows if rows else []
    except sqlite3.Error as e:
        logger.error(f"SQL execution error: {e}")
        return [{"error": f"SQL Error: {str(e)}"}]


#  Web Search & Reasoning 

# MOCK Version
# @retry_on_failure(max_attempts=2)
# def web_search(query: str, top_k: int = 3) -> List[str]:
#     """Mocked web search (replace with Tavily in production)."""
#     MOCK_WEB_DATA = {
#         "LTA policy india": [
#             "LTA (Leave Travel Allowance) in India is exempt under Section 10(5) of the Income Tax Act.",
#             "LTA can be claimed for travel within India. Airfare, train, bus tickets qualify.",
#             "Employees can claim LTA twice in a block of 4 years as per Indian tax rules.",
#         ],
#         "RAG retrieval augmented generation": [
#             "RAG combines retrieval with LLM generation to ground answers in documents.",
#             "Introduced in 2020 paper by Lewis et al. at Facebook AI Research.",
#             "Reduces hallucinations by grounding outputs in retrieved knowledge.",
#         ],
#         "EV adoption trends US 2024": [
#             "EV sales in US hit 1.2M units in 2023, up 40% YoY.",
#             "Tesla, GM, Ford lead US EV market as of 2024.",
#             "Charging infrastructure is the main bottleneck for growth.",
#         ],
#         "default": [
#             "General information about the topic.",
#             "Related facts from web sources.",
#             "Expert opinions and developments.",
#         ],
#     }
    
#     for key in MOCK_WEB_DATA:
#         if any(word in query.lower() for word in key.split()):
#             return MOCK_WEB_DATA[key][:top_k]
#     return MOCK_WEB_DATA["default"][:top_k]


# ---------------- Web Search (Tavily Production Version) ----------------

@retry_on_failure(max_attempts=2)
def web_search(query: str, top_k: int = 3) -> List[str]:
    """
    Real web search using Tavily API.
    Returns clean text snippets for grounding.
    """

    if not tavily_client:
        raise ValueError("Tavily client not initialized")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",   # better factual quality
        max_results=top_k,
        include_answer=False,
        include_raw_content=False,
    )

    if "results" not in response:
        return []

    snippets = []

    for result in response["results"]:
        content = result.get("content", "").strip()
        if content:
            snippets.append(content)

    return snippets[:top_k]


@retry_on_failure(max_attempts=2)
def llm_reason(query: str) -> str:
    """
    Guarded + domain-resolved reasoning.
    Prevents acronym hallucinations like RAG.
    """

    query_lower = query.lower()

    # -------------------------------------
    # ACRONYM DOMAIN RESOLUTION — RAG
    # -------------------------------------
    if re.search(r"\brag\b", query_lower):

        logger.info(
            "🧠 Reasoning override: Resolving RAG as AI architecture"
        )

        guard_prompt = f"""
You are an AI systems expert.

User Query:
{query}

Interpret "RAG" STRICTLY as:

Retrieval Augmented Generation — an AI architecture
that combines document retrieval with LLM generation.

Explain clearly:

1. What RAG is
2. Why it is useful
3. Where it is applied

DO NOT interpret RAG as:
- Red Amber Green
- Rapid Application Development
- Any non-AI meaning

Provide a concise technical explanation.
"""

    # -------------------------------------
    # DEFAULT REASONING (non-acronym)
    # -------------------------------------
    else:

        guard_prompt = f"""
You are answering a conceptual question.

Provide a clear, factual explanation.

Query:
{query}
"""

    # -------------------------------------
    # LLM Invocation
    # -------------------------------------

    response = llm.invoke([
        SystemMessage(
            content="You are a careful AI expert."
        ),
        HumanMessage(content=guard_prompt),
    ])

    return response.content




# ------------------------------------------------------------------------------
# 5. QUERY DECOMPOSITION FOR MULTI-INTENT QUERIES
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# QUERY DECOMPOSITION FOR MULTI-INTENT QUERIES (PRODUCTION-GRADE)
# ------------------------------------------------------------------------------

import json
import re
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

logger = logging.getLogger("MultiAgentRAG")

# ------------------------------------------------------------------------------
# SAFE JSON EXTRACTION
# ------------------------------------------------------------------------------

def _safe_json_load(text: str) -> Dict[str, Any]:
    """
    Safely extract JSON object from LLM output.
    Handles markdown / prefix / suffix noise.
    """

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("No JSON object found in LLM output")

        json_text = match.group()
        return json.loads(json_text)

    except Exception as e:
        raise ValueError(f"JSON parsing error: {e}")


# ------------------------------------------------------------------------------
# SQL COHESION DETECTOR
# Prevents over-splitting single database queries
# ------------------------------------------------------------------------------

SQL_INTENT_KEYWORDS = [
    "average", "avg", "count", "sum", "total",
    "max", "min",
    "salary", "employees", "department",
    "eligible", "permanent", "years_of_service",
    "list employees", "employee list",
    "list", "how many", "headcount"
]


def _is_sql_cohesive(query: str) -> bool:
    """
    Detect if query is a single cohesive SQL intent.
    Returns False (skip SQL-only routing) if query also mentions
    policy/document context — those need Agent A too.
    """

    query_lower = query.lower()

    # If query references policy/docs, it needs Agent A — not SQL-only
    doc_keywords = ["policy", "as per", "according to", "guideline", "document"]
    if any(kw in query_lower for kw in doc_keywords):
        return False

    keyword_hits = sum(
        1 for k in SQL_INTENT_KEYWORDS if k in query_lower
    )

    return keyword_hits >= 2


# ------------------------------------------------------------------------------
# ACRONYM DISAMBIGUATION — SAFE (WORD-BOUNDARY)
# ------------------------------------------------------------------------------

def _apply_acronym_resolution(query: str, subqs: List[Dict]):

    query_lower = query.lower()

    # Word-boundary safe detection
    if re.search(r"\brag\b", query_lower):

        logger.info(
            "Acronym detected: RAG — applying AI domain resolution"
        )

        rag_meaning = "Retrieval Augmented Generation"

        logger.info(f"RAG resolved to: {rag_meaning}")

        for sq in subqs:

            sq["text"] += (
                " (in context of AI Retrieval Augmented Generation)"
            )

            if "agents" not in sq or sq["agents"] == ["auto"]:
                sq["agents"] = ["E"]

    return subqs


# ------------------------------------------------------------------------------
# MAIN DECOMPOSITION FUNCTION
# ------------------------------------------------------------------------------

def decompose_query(query: str) -> List[Dict[str, Any]]:
    """
    Production-grade multi-intent decomposition with:

    ✔ Strict JSON contract
    ✔ Retry repair
    ✔ SQL cohesion override
    ✔ Acronym disambiguation
    ✔ Observability logging
    ✔ Safe fallback
    """

    # -------------------------------------
    # SQL COHESION OVERRIDE (PRE-LLM GUARD)
    # -------------------------------------

    if _is_sql_cohesive(query):

        logger.info(
            "SQL cohesion detected — skipping decomposition"
        )

        return [
            {
                "text": query,
                "agents": ["B"]
            }
        ]

    # -------------------------------------
    # LLM PROMPT
    # -------------------------------------

    prompt = f"""
You are a query analyzer.

Determine if this query requires multiple information sources.

User Query:
"{query}"

Return ONLY valid JSON in this format:

{{
  "multi_intent": true/false,
  "sub_queries": [
    {{"text": "sub-query 1", "agents": ["A"]}},
    {{"text": "sub-query 2", "agents": ["D"]}}
  ]
}}

Agents:
- A → Company documents (policies, HR, internal structure)
- B → Employee database (counts, lists, salaries)
- D → Web search (external / industry info)
- E → General knowledge / reasoning

CRITICAL RULES:

1. DO NOT split SQL-style queries.
   Example:
     - "Average salary of permanent employees"
     - "Count employees eligible for LTA"

   These are SINGLE database intents → Agent B.

2. Split ONLY if query combines different sources.
   Example:
     - Internal policy + industry comparison
     - Database + external research

3. Company structure → Agent A (NOT Web).

4. Output ONLY JSON.
5. No markdown.
6. No explanations.
7. No text outside JSON.

If single intent return:

{{
  "multi_intent": false,
  "sub_queries": [
    {{"text": "{query}", "agents": ["auto"]}}
  ]
}}
"""

    # -------------------------------------
    # FIRST LLM CALL
    # -------------------------------------

    try:

        response = llm.invoke([HumanMessage(content=prompt)])
        raw_output = response.content.strip()

        logger.info(f"Decomposition raw output:\n{raw_output}")

        parsed = _safe_json_load(raw_output)

    except Exception as e:

        logger.warning(f"Initial decomposition parsing failed: {e}")

        # -------------------------------------─
        # RETRY REPAIR
        # -------------------------------------─

        repair_prompt = f"""
Fix this into valid JSON only:

{raw_output}
"""

        try:

            repair_response = llm.invoke(
                [HumanMessage(content=repair_prompt)]
            )

            repaired_output = repair_response.content.strip()

            logger.info(
                f"🛠️ Repaired decomposition output:\n{repaired_output}"
            )

            parsed = _safe_json_load(repaired_output)

        except Exception as repair_error:

            logger.warning(
                f"❌ Decomposition failed after retry: {repair_error}"
            )

            return [
                {
                    "text": query,
                    "agents": ["auto"]
                }
            ]

    # -------------------------------------
    # VALIDATION + POST-PROCESSING
    # -------------------------------------

    try:

        if parsed.get("multi_intent", False):

            subqs = parsed.get("sub_queries", [])

            if not subqs:
                raise ValueError("Empty sub_queries list")

            # Acronym resolution patch
            subqs = _apply_acronym_resolution(query, subqs)

            logger.info(
                f"Query decomposed into {len(subqs)} sub-queries"
            )

            return subqs

        else:

            return [
                {
                    "text": query,
                    "agents": ["auto"]
                }
            ]

    except Exception as e:

        logger.warning(
            f"Decomposition structure invalid: {e}"
        )

        return [
            {
                "text": query,
                "agents": ["auto"]
            }
        ]


# ------------------------------------------------------------------------------
# 6.  ENHANCED ORCHESTRATOR
# ------------------------------------------------------------------------------

def heuristic_fallback(query: str) -> List[str]:
    """Rule-based fallback routing."""
    query_lower = query.lower()
    agents = []
    
    rag_keywords = ["policy", "leave", "lta", "benefit", "remote", "wfh"]
    if any(kw in query_lower for kw in rag_keywords):
        agents.append("A")
    
    sql_keywords = ["how many", "count", "list", "employees", "department"]
    if any(kw in query_lower for kw in sql_keywords):
        agents.append("B")
    
    web_keywords = [
    "market trend",
    "industry trend",
    "latest news",
    "adoption rate",
    "global",
    "external",
    "benchmark",
    ]
    if any(kw in query_lower for kw in web_keywords):
        agents.append("D")
    
    reason_keywords = ["what is", "explain", "define"]
    if any(kw in query_lower for kw in reason_keywords) and not agents:
        agents.append("E")
    
    return list(set(agents)) if agents else ["E"]


def orchestrator(state: PipelineState) -> dict:
    """
    Enhanced orchestrator with query decomposition.
    """
    query = state["user_query"]
    exec_log = state.get("execution_log", [])
    
    logger.info(f"🎯 ORCHESTRATOR: Analyzing query")
    
    # Step 1: Decompose query if multi-intent
    sub_queries = decompose_query(query)
    
    # Step 2: Route each sub-query using heuristic (reliable, no LLM call needed)
    all_agents = set()
    for sq in sub_queries:
        if sq["agents"] == ["auto"]:
            sq_agents = heuristic_fallback(sq["text"])
        else:
            sq_agents = sq["agents"]
        all_agents.update(sq_agents)
    
    agents = list(all_agents)
    agents = [a for a in agents if a in ["A", "B", "D", "E"]]
    
    if not agents:
        agents = ["E"]
    
    exec_log.append({
        "timestamp": datetime.now().isoformat(),
        "node": "orchestrator",
        "action": "routing",
        "sub_queries": sub_queries,
        "agents_selected": agents,
    })
    
    logger.info(f"   → Sub-queries: {len(sub_queries)}")
    logger.info(f"   → Agents: {agents}")
    
    return {
        "agents_to_run": agents,
        "sub_queries": sub_queries,
        "execution_log": exec_log,
        "tool_failures": {},
    }


# ------------------------------------------------------------------------------
# 7.  PARALLEL AGENTS WITH FAILURE RECOVERY
# ------------------------------------------------------------------------------

def agent_a_rag_parallel(state: PipelineState) -> Dict[str, Any]:
    """Agent A with failure fallback."""
    if "A" not in state["agents_to_run"]:
        return {}
    
    query = state["user_query"]
    logger.info(f"[Agent A] Retrieving docs")
    
    chunks = retrieve_docs(query, top_k=5)
    
    #Failure recovery: if RAG returns empty, try reasoning fallback
    # if not chunks or (isinstance(chunks, dict) and "error" in chunks):
    #     logger.warning("[Agent A] RAG failed or empty, triggering reasoning fallback")
    #     reason_answer = llm_reason(f"Based on general knowledge about HR policies: {query}")
    #     return {
    #         "rag_results": [],
    #         "reason_result": f"[RAG unavailable, using general knowledge]\n{reason_answer}",
    #         "execution_log": [{
    #             "timestamp": datetime.now().isoformat(),
    #             "node": "agent_a",
    #             "action": "retrieve_failed_fallback_to_reasoning",
    #         }]
    #     }
    if not chunks:
        logger.warning("[Agent A] No relevant chunks found")

        return {
            "rag_results": [],
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "agent_a",
                "action": "retrieve_empty_no_fallback",
            }]
        }

    
    return {
        "rag_results": chunks,
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "node": "agent_a",
            "action": "retrieve",
            "chunks_retrieved": len(chunks),
        }]
    }


def agent_b_sql_parallel(state: PipelineState) -> Dict[str, Any]:
    """Agent B with failure fallback."""
    if "B" not in state["agents_to_run"]:
        return {}
    
    query = state["user_query"]
    logger.info(f"[Agent B] Generating SQL")
    tables_hint = ", ".join(ALLOWED_TABLES)

    sql_prompt = f"""
        You are a senior SQL expert.

        Use ONLY the schema below:

        {DB_SCHEMA}

        CRITICAL RULES:

        1. Write ONLY a SELECT query.
        2. DO NOT add explanations.
        3. DO NOT add markdown.
        4. DO NOT invent columns.

        BUSINESS LOGIC DEFINITIONS:

        • LTA eligibility =
            is_permanent = 1
            AND years_of_service >= 1.0

        • Leave requests table does NOT define eligibility.
        • Eligibility must be computed from employees table.

        Examples:

        Q: Count employees eligible for LTA
        SQL:
        SELECT COUNT(*)
        FROM employees
        WHERE is_permanent = 1
        AND years_of_service >= 1.0;

        Q: List eligible employees
        SQL:
        SELECT name
        FROM employees
        WHERE is_permanent = 1
        AND years_of_service >= 1.0;

        Now write SQL for:

        "{query}"
        """

    sql_response = llm.invoke([HumanMessage(content=sql_prompt)])

    sql = extract_sql_only(sql_response.content)

    
    logger.info(f"   → SQL: {sql}")
    
    rows = sql_query_tool(sql)
    
    #Failure recovery: if SQL fails, log and return gracefully
    if rows and isinstance(rows, list) and len(rows) > 0 and "error" in rows[0]:
        logger.warning(f"[Agent B] SQL execution failed: {rows[0]['error']}")
        return {
            "sql_results": [],
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "agent_b",
                "action": "sql_query_failed",
                "error": rows[0]["error"],
            }]
        }
    
    return {
        "sql_results": rows,
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "node": "agent_b",
            "action": "sql_query",
            "sql": sql,
            "rows_returned": len(rows),
        }]
    }


def agent_d_web_parallel(state: PipelineState) -> Dict[str, Any]:
    """Agent D with failure fallback."""
    if "D" not in state["agents_to_run"]:
        return {}
    
    query = state["user_query"]
    logger.info(f"[Agent D] Web search")
    
    snippets = web_search(query, top_k=3)
    
    #Failure recovery
    if isinstance(snippets, dict) and "error" in snippets:
        logger.warning("[Agent D] Web search failed")
        return {
            "web_results": [],
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "agent_d",
                "action": "web_search_failed",
            }]
        }
    
    return {
        "web_results": snippets,
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "node": "agent_d",
            "action": "web_search",
            "results_count": len(snippets),
        }]
    }


def agent_e_reason_parallel(state: PipelineState) -> Dict[str, Any]:
    """Agent E."""
    if "E" not in state["agents_to_run"]:
        return {}
    
    query = state["user_query"]
    logger.info(f"[Agent E] LLM reasoning")
    
    answer = llm_reason(query)
    
    return {
        "reason_result": answer,
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "node": "agent_e",
            "action": "reasoning",
        }]
    }


def execute_agents_parallel(state: PipelineState) -> dict:
    """Run all agents in parallel with failure recovery."""
    agents_map = {
        "A": agent_a_rag_parallel,
        "B": agent_b_sql_parallel,
        "D": agent_d_web_parallel,
        "E": agent_e_reason_parallel,
    }
    
    agents_to_run = state["agents_to_run"]
    
    if not agents_to_run:
        return {}
    
    logger.info(f"⚡ Executing {len(agents_to_run)} agents in parallel...")
    
    results = {
        "rag_results":    [],
        "sql_results":    [],
        "web_results":    [],
        "reason_result":  "",
        "execution_log":  [],
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(agents_map[agent], state): agent 
            for agent in agents_to_run if agent in agents_map
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                for key, value in result.items():
                    if key == "execution_log":
                        results["execution_log"].extend(value)
                    elif key in results and value:
                        results[key] = value
                logger.info(f"   ✓ Agent {agent_name} completed")
            except Exception as e:
                logger.error(f"   ✗ Agent {agent_name} failed: {e}")
                results["execution_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "node": f"agent_{agent_name.lower()}",
                    "action": "catastrophic_failure",
                    "error": str(e),
                })
    
    return results


# ------------------------------------------------------------------------------
# 8. ENFORCED CITATION + GROUNDING GUARDRAILS (FIXED)
# ------------------------------------------------------------------------------

import json
from datetime import datetime
from langchain_core.messages import HumanMessage


# ------------------------------------------------------------------------------
# Grounding verification (post-answer audit)
# ------------------------------------------------------------------------------

def verify_grounding(answer, sql, rag):
    """
    Detect if answer ignored available grounded data.
    Adds evaluator-visible warnings.
    """

    if sql and "cannot determine" in answer.lower():
        return (
            answer
            + "\n\n[WARNING: SQL data was available but not fully used]"
        )

    if rag and "not available in documents" in answer.lower():
        return (
            answer
            + "\n\n[WARNING: Documents contained relevant data]"
        )

    return answer


# ------------------------------------------------------------------------------
# SQL sanitizer (removes error payloads)
# ------------------------------------------------------------------------------
TABLE_NAME_FIXES = {
    "department": "departments",
    "employee": "employees",
    "leave_request": "leave_requests",
}

def autocorrect_table_names(sql: str) -> str:
    """
    Auto-fix common LLM table hallucinations.
    """

    for wrong, correct in TABLE_NAME_FIXES.items():

        sql = re.sub(
            rf"\b{wrong}\b",
            correct,
            sql,
            flags=re.IGNORECASE
        )

    return sql


def sanitize_sql_rows(rows):
    """
    Remove SQL error payloads before synthesis.
    Prevents hallucinated DB answers.
    """

    if not rows:
        return []

    if isinstance(rows, list) and "error" in rows[0]:
        logger.warning(
            f"🚫 SQL failure detected: {rows[0]['error']}"
        )
        return []

    return rows


# ------------------------------------------------------------------------------
# Grounding validation (pre-synthesis guardrail)
# ------------------------------------------------------------------------------

def validate_grounding(chunks, sql_rows, web_rows, reasoning):
    """
    Ensure synthesis has at least one verified source.
    Blocks hallucinated fallback answers.
    """

    has_chunks = bool(chunks)
    has_sql    = bool(sql_rows)
    has_web    = bool(web_rows)
    has_llm    = bool(reasoning)

    if not (has_chunks or has_sql or has_web or has_llm):
        return False, "No verified sources available"

    return True, ""


# ------------------------------------------------------------------------------
# Tool priority context builder
# ------------------------------------------------------------------------------

def enforce_tool_priority(sql, rag, reasoning):
    """
    Ensure factual tools override LLM reasoning.
    """

    priority_context = []

    if sql:
        priority_context.append(
            "DATABASE RESULTS (HIGHEST PRIORITY):\n"
            + json.dumps(sql, indent=2)
        )

    if rag:
        rag_text = "\n".join([r["text"] for r in rag])
        priority_context.append(
            "DOCUMENT EVIDENCE:\n" + rag_text
        )

    if reasoning and not sql and not rag:
        priority_context.append(
            "LLM KNOWLEDGE (LOW PRIORITY):\n" + reasoning
        )

    return "\n\n".join(priority_context)

##########################
# ------------------------------------------------------------------------------─
# Context compression + deduplication (prevents repetition)
# ------------------------------------------------------------------------------─

def deduplicate_chunks(chunks):
    """
    Remove duplicate chunk texts.
    Prevents repeated policy paragraphs.
    """
    seen = set()
    unique = []

    for c in chunks:
        text_hash = hash(c["text"])

        if text_hash not in seen:
            unique.append(c)
            seen.add(text_hash)

    return unique


def compress_rag_context(chunks, max_chars=2000):
    """
    Limit total RAG context size.
    Prevents LLM over-generation.
    """

    total = 0
    compressed = []

    for c in chunks:
        length = len(c["text"])

        if total + length > max_chars:
            break

        compressed.append(c)
        total += length

    return compressed

def clean_answer_output(text: str) -> str:
    """
    Removes excessive blank space + trailing gaps.
    """

    if not text:
        return text

    # Collapse >2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace
    text = text.strip()

    return text
# 
# FINAL CONTEXT BUILDER (STRICT TOOL PRIORITY)
# 

def build_context(sql, rag, web, reasoning):
    """
    Enforces tool priority:

    DB > Documents > Web > LLM
    """

    blocks = []

    #  DATABASE ─
    if sql:
        blocks.append(
            "=== DATABASE RESULTS (HIGHEST PRIORITY) ===\n"
            + json.dumps(sql, indent=2)
        )

    #  DOCUMENTS 
    if rag:
        rag_text = "\n\n".join([
            f"[CHUNK_{r['chunk_id']}] ({r['source']})\n{r['text']}"
            for r in rag
        ])

        blocks.append(
            "=== COMPANY DOCUMENTS ===\n" + rag_text
        )

    #  WEB ─
    if web:
        blocks.append(
            "=== WEB SEARCH ===\n"
            + "\n".join(f"• {w}" for w in web)
        )

    #  LLM (ONLY if nothing else exists) ─
    if reasoning and not (sql or rag or web):
        blocks.append(
            "=== LLM KNOWLEDGE ===\n" + reasoning
        )

    return "\n\n".join(blocks)


def strip_hallucinated_citations(answer: str, sql: list, rag: list, web: list) -> str:
    """
    Remove citation tags from answer that have no backing data.
    Prevents the model inventing [CHUNK_1], [Database] etc.
    """
    # If no SQL data, remove [DB] and [Database] tags
    if not sql:
        answer = re.sub(r'\[DB\]', '', answer)
        answer = re.sub(r'\[Database\]', '', answer, flags=re.IGNORECASE)

    # If no RAG chunks, remove all [CHUNK_*] tags
    if not rag:
        answer = re.sub(r'\[CHUNK_[^\]]*\]', '', answer)

    # If no web results, remove [Web] tags
    if not web:
        answer = re.sub(r'\[Web\]', '', answer, flags=re.IGNORECASE)

    # Clean up empty References sections
    answer = re.sub(r'References:\s*\n(\s*\n)+', '', answer)

    return answer.strip()


# ------------------------------------------------------------------------------
# Agent-C Synthesizer
# ------------------------------------------------------------------------------

def clean_references(answer: str, sql: list, rag: list, web: list) -> str:
    """
    Remove hallucinated content from the References section.
    - Strip fake SQL from [DB] lines
    - Remove [CHUNK_x] if no RAG chunks were retrieved
    - Remove [Web] if no web results exist
    - Remove [LLM] if reasoning agent wasn't used
    """

    # Remove any SQL statements that appear after [DB]
    # e.g.  [DB] SELECT COUNT(*) FROM ...  → [DB]
    answer = re.sub(r'\[DB\]\s+SELECT[^\n]*', '[DB]', answer, flags=re.IGNORECASE)

    # Remove [CHUNK_x] tags if no RAG data was retrieved
    if not rag:
        answer = re.sub(r'\[CHUNK_[^\]]*\]', '', answer)

    # Remove [Web] if no web results
    if not web:
        answer = re.sub(r'\[Web\]', '', answer, flags=re.IGNORECASE)

    # Clean up blank lines left behind in References section
    answer = re.sub(r'References:\s*\n(\s*\n)+', 'References:\n', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    return answer.strip()

def agent_c_synthesizer(state: PipelineState) -> dict:
    """
    Agent C — Grounded synthesis with enforced citations
    + Hallucination guardrails
    + Source validation
    """

    logger.info(
        "[Agent C] Synthesizing with enforced citations"
    )

    # -------------------------------------
    # Extract state inputs
    # -------------------------------------

    query       = state["user_query"]
    rag         = state.get("rag_results", [])
    sql_raw     = state.get("sql_results", [])
    web         = state.get("web_results", [])
    reasoning   = state.get("reason_result", "")

    if state.get("sql_results") or state.get("rag_results") or state.get("web_results"):
        reasoning = ""

    #  NEW: Deduplicate + compress RAG context 
    rag = deduplicate_chunks(rag)
    rag = compress_rag_context(rag)


    # -------------------------------------
    # Sanitize SQL errors
    # -------------------------------------

    sql = sanitize_sql_rows(sql_raw)

    # -------------------------------------
    # Grounding validation BEFORE synthesis
    # -------------------------------------

    is_grounded, grounding_error = validate_grounding(
        rag, sql, web, reasoning
    )

    if not is_grounded:

        logger.warning(
            "Synthesis blocked — no verified grounding sources"
        )

        fallback_answer = (
            "I could not retrieve verified information to answer "
            "this query. Please refine the query or check the data sources."
        )

        exec_log = state.get("execution_log", [])
        exec_log.append({
            "timestamp": datetime.now().isoformat(),
            "node": "agent_c",
            "action": "synthesis_blocked_no_grounding"
        })

        return {
            "final_answer": fallback_answer,
            "execution_log": exec_log
        }



    all_context = build_context(sql, rag, web, reasoning)

    # Pre-format DB result and RAG facts for direct injection into prompt
    db_line  = ""
    rag_facts = ""

    if sql:
        # Convert DB rows into a simple readable string
        db_line = "DATABASE ANSWER: " + ", ".join(
            f"{k}={v}" for row in sql for k, v in row.items()
        )

    if rag:
        valid_chunk_ids = [r["chunk_id"] for r in rag]
        rag_facts = "\n".join(
            f"[CHUNK_{r['chunk_id']}]: {r['text'][:200]}"
            for r in rag[:3]
        )
    else:
        valid_chunk_ids = []

    synth_prompt = f"""Answer this question using ONLY the data below.

QUESTION: {query}

{db_line}

DOCUMENT EXCERPTS:
{rag_facts if rag_facts else "None"}

INSTRUCTIONS:
- Start your answer with the database number if one exists (e.g. "9 employees...").
- Cite [DB] for database facts, [CHUNK_<id>] for document facts.
- Only use these chunk IDs: {", ".join(valid_chunk_ids) if valid_chunk_ids else "none"}
- Keep answer under 80 words.
- End with: References: [DB] or [CHUNK_id] as used.

Answer:"""
    # -------------------------------------
    # Invoke LLM
    # -------------------------------------

    response = llm.invoke(
        [HumanMessage(content=synth_prompt)],
        max_tokens=512
    )

    final_answer = response.content
    final_answer = clean_references(final_answer, sql, rag, web)
    final_answer = strip_hallucinated_citations(final_answer, sql, rag, web)
    # -------------------------------------
    # Post-answer grounding audit
    # -------------------------------------

    final_answer = verify_grounding(
        final_answer, sql, rag
    )

    #  NEW: Clean whitespace + repetition artifacts 
    final_answer = clean_answer_output(final_answer)
    # -------------------------------------
    # Execution logging
    # -------------------------------------

    exec_log = state.get("execution_log", [])
    exec_log.append({
        "timestamp": datetime.now().isoformat(),
        "node": "agent_c",
        "action": "synthesis_with_enforced_citations",
        "chunks_used": len(rag),
        "sql_rows_used": len(sql),
        "web_results_used": len(web)
    })

    exec_log = deduplicate_logs(exec_log)

    return {
        "final_answer": final_answer,
        "execution_log": exec_log
    }



# ------------------------------------------------------------------------------
# 9.  BUILD GRAPH
# ------------------------------------------------------------------------------

def should_run_agents(state: PipelineState) -> str:
    if state.get("agents_to_run"):
        return "agents"
    return "synthesizer"


def build_pipeline():
    graph = StateGraph(PipelineState)
    
    graph.add_node("orchestrator",    orchestrator)
    graph.add_node("parallel_agents", execute_agents_parallel)
    graph.add_node("synthesizer",     agent_c_synthesizer)
    
    graph.set_entry_point("orchestrator")
    
    graph.add_conditional_edges(
        "orchestrator",
        should_run_agents,
        {
            "agents":      "parallel_agents",
            "synthesizer": "synthesizer",
        }
    )
    
    graph.add_edge("parallel_agents", "synthesizer")
    graph.add_edge("synthesizer",     END)
    
    return graph.compile()


# ------------------------------------------------------------------------------
# 10. EXECUTION
# ------------------------------------------------------------------------------
def detect_sources(state, answer):

    sources = []

    if "[DB]" in answer:
        sources.append("Database")

    if "[CHUNK_" in answer:
        sources.append("Company Documents")

    if "[Web]" in answer:
        sources.append("Web Search")

    if "[LLM]" in answer:
        sources.append("LLM Knowledge")

    return sources



def extract_sql_from_logs(logs):

    for log in logs:
        if log.get("node") == "agent_b" and log.get("action") == "sql_query":
            return log.get("sql")

    return None



# ======================================= Run Query Function ================================
# ------------------------------------------------------------------------------
# Display helpers  (plain text — no ANSI colors)
# ------------------------------------------------------------------------------
import os
import json
from datetime import datetime

# All helpers just return the text unchanged — clean plain output
def BOLD(t):    return t
def CYAN(t):    return t
def GREEN(t):   return t
def YELLOW(t):  return t
def MAGENTA(t): return t
def BLUE(t):    return t
def RED(t):     return t
def DIM(t):     return t

SEP_HEAVY = "═" * 70
SEP_LIGHT = "─" * 70
SEP_DOT   = "·" * 70


# ------------------------------------------------------------------------------
# HELPER — map internal node names → human labels
# ------------------------------------------------------------------------------

_AGENT_LABELS = {
    "agent_a": "Agent A  — RAG Retriever",
    "agent_b": "Agent B  — SQL Agent",
    "agent_d": "Agent D  — Web Search",
    "agent_e": "Agent E  — Reasoning (LLM)",
    "agent_c": "Agent C  — Answer Synthesizer",
    "orchestrator": "Orchestrator",
}

def _label(node: str) -> str:
    return _AGENT_LABELS.get(node, node.replace("_", " ").title())


# ------------------------------------------------------------------------------
# SECTION PRINTERS
# ------------------------------------------------------------------------------

def print_header(query: str):
    print()
    print(BOLD(CYAN(SEP_HEAVY)))
    print(BOLD(CYAN("  MULTI-AGENT RAG PIPELINE  —  EXECUTION REPORT")))
    print(BOLD(CYAN(SEP_HEAVY)))
    print(f"  {BOLD('Query :')} {query}")
    print(f"  {BOLD('Time  :')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(BOLD(CYAN(SEP_HEAVY)))


def print_routing(logs: list):
    """Show orchestrator routing decision."""
    print()
    print(BOLD(YELLOW("▌ STEP 1 — ORCHESTRATOR  (Routing Decision)")))
    print(DIM(SEP_LIGHT))

    for log in logs:
        if log.get("node") != "orchestrator":
            continue

        agents_sel = log.get("agents_selected", [])
        sub_qs     = log.get("sub_queries", [])

        # Agent map
        label_map = {
            "A": "Agent A  (RAG)",
            "B": "Agent B  (SQL)",
            "D": "Agent D  (Web Search)",
            "E": "Agent E  (Reasoning)",
        }

        print(f"  {BOLD('Agents selected :')} "
              + ", ".join(label_map.get(a, a) for a in agents_sel))

        if len(sub_qs) > 1:
            print(f"  {BOLD('Query decomposed into')} {len(sub_qs)} sub-queries:")
            for i, sq in enumerate(sub_qs, 1):
                print(f"    [{i}] {sq['text']}")
                print(f"        → routed to: {sq.get('agents', ['auto'])}")
        else:
            print(f"  {BOLD('Single-intent query')} — no decomposition needed")

        break   # only first orchestrator log needed


def print_rag(chunks: list):
    """Print retrieved document chunks."""
    if not chunks:
        print()
        print(BOLD(BLUE("▌ STEP 2 — AGENT A  (RAG Retriever)")))
        print(DIM(SEP_LIGHT))
        print(f"  {RED('No relevant chunks retrieved')} "
              "(below relevance threshold or empty)")
        return

    print()
    print(BOLD(BLUE(f"▌ STEP 2 — AGENT A  (RAG Retriever)  "
                    f"— {len(chunks)} chunk(s) retrieved")))
    print(DIM(SEP_LIGHT))

    for i, chunk in enumerate(chunks, 1):
        score  = chunk.get("score", "N/A")
        source = chunk.get("source", "Unknown")
        cid    = chunk.get("chunk_id", "???")
        text   = chunk.get("text", "")

        # Truncate preview to 300 chars
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "…"

        print(f"  {BOLD(f'Chunk [{i}]')}  "
              f"ID={CYAN(cid)}  "
              f"Source={GREEN(source)}  "
              f"Score={YELLOW(str(round(float(score), 4)) if score != 'N/A' else 'N/A')}")
        print(f"  {DIM('Preview:')} {preview}")
        if i < len(chunks):
            print(f"  {DIM(SEP_DOT)}")


def print_sql(logs: list, sql_results: list):
    """Print SQL query and raw results."""
    # Find SQL log entry
    sql_log = next(
        (l for l in logs
         if l.get("node") == "agent_b" and l.get("action") == "sql_query"),
        None
    )

    print()
    print(BOLD(GREEN("▌ STEP 3 — AGENT B  (SQL Agent)")))
    print(DIM(SEP_LIGHT))

    if sql_log is None and not sql_results:
        print(f"  {DIM('SQL agent not invoked for this query.')}")
        return

    if sql_log:
        sql_text = sql_log.get("sql", "N/A")
        rows_n   = sql_log.get("rows_returned", len(sql_results))
        print(f"  {BOLD('SQL query executed:')}")
        # indent each line
        for line in sql_text.strip().splitlines():
            print(f"    {CYAN(line)}")
        print(f"  {BOLD('Rows returned:')} {rows_n}")

    # Error case
    if sql_results and "error" in sql_results[0]:
        print(f"  {RED('SQL Error:')} {sql_results[0]['error']}")
        return

    if sql_results:
        print(f"  {BOLD('Raw results:')}")
        display_rows = sql_results[:10]
        for line in json.dumps(display_rows, indent=4).splitlines():
            print(f"    {line}")
        if len(sql_results) > 10:
            print(f"    (... and {len(sql_results)-10} more rows)")
    else:
        print(f"  Query returned 0 rows.")


def print_web(web_results: list):
    """Print web search snippets."""
    print()
    print(BOLD(MAGENTA("▌ STEP 4 — AGENT D  (Internet Search)")))
    print(DIM(SEP_LIGHT))

    if not web_results:
        print(f"  {DIM('Web search agent not invoked for this query.')}")
        return

    print(f"  {BOLD(f'{len(web_results)} snippet(s) retrieved:')}")
    for i, snippet in enumerate(web_results, 1):
        print(f"  [{i}] {snippet}")


def print_reasoning(reason_result: str):
    """Print LLM reasoning output."""
    print()
    print(BOLD(RED("▌ STEP 5 — AGENT E  (Reasoning Agent)")))
    print(DIM(SEP_LIGHT))

    if not reason_result:
        print(f"  {DIM('Reasoning agent not invoked for this query.')}")
        return

    print(f"  {BOLD('LLM Knowledge output:')}")
    for line in reason_result.strip().splitlines():
        print(f"  {line}")


def print_answer(final_answer: str):
    """Print the synthesized final answer — deduplicated."""
    print()
    print(BOLD(CYAN("▌ STEP 6 — AGENT C  (Answer Synthesizer)")))
    print(DIM(SEP_LIGHT))
    print(f"  {BOLD('FINAL ANSWER:')}")
    print()
    # Deduplicate consecutive repeated lines (model sometimes echoes output)
    lines = final_answer.strip().splitlines()
    seen_lines = set()
    deduped = []
    for line in lines:
        stripped = line.strip()
        if stripped not in seen_lines:
            deduped.append(line)
            if stripped:  # only track non-blank lines
                seen_lines.add(stripped)
    for line in deduped:
        print(f"  {line}")


def print_sources(final_answer: str, logs: list):
    """Detect and display source breakdown."""
    sources = []
    if "[DB]"    in final_answer: sources.append(("DB",   "SQL Database"))
    if "[CHUNK_" in final_answer: sources.append(("DOCS", "Company Documents (RAG)"))
    if "[Web]"   in final_answer: sources.append(("Web",  "Web Search"))
    if "[LLM]"   in final_answer: sources.append(("LLM",  "LLM Internal Knowledge"))

    print()
    print(BOLD(YELLOW("▌ SOURCE BREAKDOWN")))
    print(DIM(SEP_LIGHT))
    if sources:
        for tag, label in sources:
            print(f"  {BOLD(f'[{tag}]')}  →  {label}")
    else:
        print(f"  {DIM('No explicit citation tags found in answer.')}")


def print_execution_path(logs: list):
    """Show condensed execution path."""
    print()
    print(BOLD(DIM("▌ EXECUTION PATH (internal log)")))
    print(DIM(SEP_LIGHT))

    seen = set()
    for log in logs:
        node   = log.get("node", "?")
        action = log.get("action", "?")
        ts     = log.get("timestamp", "")
        key    = (node, action)
        if key in seen:
            continue
        seen.add(key)
        label = _label(node)
        print(f"  {DIM(ts[:19])}  {BOLD(label):45s}  → {action}")


def print_footer():
    print()
    print(BOLD(CYAN(SEP_HEAVY)))
    print()


# ------------------------------------------------------------------------------
# MAIN run_query FUNCTION
# ------------------------------------------------------------------------------

def run_query(pipeline, question: str) -> dict:
    """
    Run the multi-agent pipeline and print a structured execution report.

    Output sections:
      1. Orchestrator routing decision
      2. Agent A  — RAG retrieved chunks
      3. Agent B  — SQL query + raw results
      4. Agent D  — Web search snippets
      5. Agent E  — LLM reasoning output
      6. Agent C  — Final synthesized answer
      7. Source breakdown
      8. Execution path log
    """

    initial_state = {
        "user_query":    question,
        "sub_queries":   [],
        "agents_to_run": [],
        "rag_results":   [],
        "sql_results":   [],
        "web_results":   [],
        "reason_result": "",
        "final_answer":  "",
        "execution_log": [],
        "timestamp":     datetime.now().isoformat(),
        "tool_failures": {},
    }

    result = pipeline.invoke(initial_state)

    # Deduplicate logs before printing (LangGraph can accumulate duplicates)
    raw_logs = result.get("execution_log", [])
    seen_keys = set()
    logs = []
    for entry in raw_logs:
        key = (entry.get("node"), entry.get("action"))
        if key not in seen_keys:
            logs.append(entry)
            seen_keys.add(key)

    rag_chunks   = result.get("rag_results",   [])
    sql_results  = result.get("sql_results",   [])
    web_results  = result.get("web_results",   [])
    reason       = result.get("reason_result", "")
    final_answer = result.get("final_answer",  "")

    #  Print all sections
    print_header(question)
    print_routing(logs)
    print_rag(rag_chunks)
    print_sql(logs, sql_results)
    print_web(web_results)
    print_reasoning(reason)
    print_answer(final_answer)
    print_sources(final_answer, logs)
    print_execution_path(logs)
    print_footer()

    # ── Dump execution log to working directory ────────────────────────────
    log_dir = ROOT_DIR / "execution_logs"
    log_dir.mkdir(exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_q = re.sub(r"[^a-zA-Z0-9]", "_", question[:40])
    log_file = log_dir / f"log_{ts}_{safe_q}.json"

    # Map agent letters to full names for readability
    agent_label_map = {
        "A": "Agent A — RAG Retriever",
        "B": "Agent B — SQL Agent",
        "D": "Agent D — Web Search",
        "E": "Agent E — Reasoning (LLM)",
    }
    agents_used = result.get("agents_to_run", [])

    # Extract SQL query string from logs
    sql_query_str = next(
        (e.get("sql", "") for e in logs if e.get("node") == "agent_b" and e.get("action") == "sql_query"),
        None
    )

    log_payload = {
        "query": question,
        "timestamp": datetime.now().isoformat(),

        # 1. Which agents were invoked
        "agents_invoked": {
            letter: agent_label_map.get(letter, letter)
            for letter in agents_used
        },

        # 2. Retrieved documents / chunks (with preview)
        "retrieved_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "score": round(float(c.get("score", 0)), 4),
                "preview": c.get("text", "")[:200].replace("\n", " "),
            }
            for c in rag_chunks
        ],

        # 3. SQL query executed + results
        "sql_agent": {
            "query_executed": sql_query_str,
            "sample_results": sql_results[:5],
            "rows_returned": len(sql_results),
        } if sql_query_str else None,

        # 4. Web search queries + snippets
        "web_search": {
            "snippets_retrieved": len(web_results),
            "results": web_results,
        } if web_results else None,

        # 5. Final response
        "final_answer": final_answer,

        # Internal execution trace
        "execution_trace": [
            {
                "timestamp": e.get("timestamp", ""),
                "agent": agent_label_map.get(
                    e.get("node", "").replace("agent_", "").upper(),
                    e.get("node", "")
                ),
                "action": e.get("action", ""),
            }
            for e in logs
        ],
    }

    with open(log_file, "w") as f:
        json.dump(log_payload, f, indent=2, default=str)

    print(f"  [Log saved -> execution_logs/{log_file.name}]")

    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT_DIR))

    # ── Try loading feedback memory (Section 4) ────────────────────────────
    try:
        from feedback.feedback_memory import FeedbackMemory
        fm = FeedbackMemory()
        FEEDBACK_ON = True
    except Exception as e:
        print(f"[WARN] Feedback memory not available: {e}")
        FEEDBACK_ON = False

    pipeline = build_pipeline()

    print()
    print(BOLD(CYAN("=" * 65)))
    print(BOLD(CYAN("  MULTI-AGENT RAG SYSTEM  —  INTERACTIVE CLI")))
    print(BOLD(CYAN("=" * 65)))
    print(f"  Type your question and press Enter.")
    print(f"  Type {BOLD('exit')} or {BOLD('quit')} to stop.")
    print(BOLD(CYAN("=" * 65)))

    while True:
        # ── Get question from user ─────────────────────────────────────────
        print()
        try:
            question = input(BOLD("  Your question: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not question:
            print("  Please enter a question.")
            continue

        if question.lower() in ("exit", "quit"):
            print("\n  Goodbye!\n")
            break

        # ── Run pipeline ───────────────────────────────────────────────────
        result = run_query(pipeline, question)

        # ── Collect feedback (Section 4) ───────────────────────────────────
        if FEEDBACK_ON:
            print()
            print(BOLD(YELLOW("▌ FEEDBACK  (Section 4 — Feedback Memory)")))
            print(DIM("─" * 65))
            print(f"  Was this answer helpful?  "
                  f"{BOLD(GREEN('[g]'))} Good  |  "
                  f"{BOLD(RED('[b]'))} Bad   |  "
                  f"{BOLD(DIM('[s]'))} Skip")
            print()

            try:
                fb = input("  Your choice (g/b/s): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                fb = "s"

            if fb in ("g", "b"):
                tag       = "good" if fb == "g" else "bad"
                answer    = result.get("final_answer", "")
                rag       = result.get("rag_results", [])
                chunk_ids = [c["chunk_id"] for c in rag if "chunk_id" in c]

                fm.store_feedback(
                    query     = question,
                    answer    = answer,
                    tag       = tag,
                    chunk_ids = chunk_ids,
                )

                # ── Show similar past feedback ─────────────────────────────
                similar = fm.find_similar(question, top_k=2)
                if similar:
                    print()
                    print(f"  {BOLD('Similar past feedback found:')}")
                    for i, entry in enumerate(similar, 1):
                        print(f"  [{i}] sim={entry['similarity']:.3f} "
                              f"\"{entry['query'][:60]}\"")

                # ── Show chunk reputation summary ──────────────────────────
                rep = fm.get_chunk_reputation()
                if rep:
                    trusted = sum(1 for v in rep.values() if v["score"] >= 0.7)
                    demoted = sum(1 for v in rep.values() if v["score"] <  0.4)
                    print(f"  {BOLD('Chunk reputation:')} "
                          f"{GREEN(str(trusted))} trusted  |  "
                          f"{RED(str(demoted))} demoted  "
                          f"(across {len(rep)} chunks)")

            elif fb == "s":
                print(f"  {DIM('Feedback skipped.')}")
            else:
                print(f"  {DIM('Invalid input — feedback skipped.')}")

        print()
        print(BOLD(CYAN("─" * 65)))
        print(f"  {DIM('Ask another question or type exit to quit.')}")
