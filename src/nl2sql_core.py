import os
import re
import math
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from dotenv import load_dotenv
import chromadb
from crewai.llm import LLM
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from collections import defaultdict
from crewai import Agent, Task, Crew, Process


# =========================
# 0) General Settings + ENV
# =========================

load_dotenv()

@dataclass
class Settings:
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "sakila")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASS: str = os.getenv("DB_PASS")
    SQLALCHEMY_ECHO: bool = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"
    MODEL_NAME: str = os.getenv("MODEL_NAME", "groq/llama-3.3-70b-versatile")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    AGENTOPS_API_KEY: Optional[str] = os.getenv("AGENTOPS_API_KEY")

cfg = Settings()

MYSQL_URL = URL.create(
    drivername="mysql+pymysql",
    username=cfg.DB_USER,
    password=cfg.DB_PASS,
    host=cfg.DB_HOST,
    port=int(cfg.DB_PORT),
    database=cfg.DB_NAME,
)

Engine: Engine = create_engine(MYSQL_URL, echo=cfg.SQLALCHEMY_ECHO, pool_pre_ping=True)

# Embeddings & Chroma settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "./vector_sql_schema3"

# =========================
# 1) General Helper Functions
# =========================

def _norm_name(name: str) -> str:
    """Normalize a table/column name to lowercase and remove backticks."""
    return (name or "").strip().strip("`").lower()


def extract_sql_from_text(s: str) -> str:
    """
    Try to extract a SQL statement from a text that may contain ```sql ... ``` fences.
    If nothing is found, return the input text stripped.
    """
    s = str(s).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


# =========================
# 2) ACL: User Permissions
# =========================

def get_user_acl(user_id: str) -> dict:
    """
    Read table-level permissions for a given user from acl_user_tables in MySQL.
    Expected columns:
      - user_id
      - table_schema
      - table_name
      - is_allowed (0/1)

    We constrain it to the same schema used in cfg.DB_NAME.
    """
    with Engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT table_name, is_allowed
                FROM acl_user_tables
                WHERE user_id = :uid AND table_schema = :schema
            """),
            {"uid": user_id, "schema": cfg.DB_NAME}
        ).mappings().all()

    allowed_tables = sorted(
        [r["table_name"] for r in rows if int(r["is_allowed"]) == 1]
    )

    return {
        "tables": allowed_tables,
        "columns": {},
        "row_filters": {},
    }


def get_allowed_tables_set(user_id: str) -> Set[str]:
    """
    Return a set of normalized table names that the user is allowed to query.
    """
    acl = get_user_acl(user_id)
    return {_norm_name(t) for t in acl.get("tables", [])}


# =========================
# 3) ChromaDB + Embedding Setup
# =========================

class LocalSentenceTransformerEF(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def __call__(self, inputs):
        return self.model.encode(inputs).tolist()


client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    name="vector_DB",
    embedding_function=LocalSentenceTransformerEF(EMBEDDING_MODEL_NAME),
    metadata={"hnsw:space": "cosine"},
)


# =========================
# 4) Find Related Tables (Vector + ACL)
# =========================

def find_related_tables_with_distances(
    collection,
    user_query: str,
    top_k_tables: int = 10,
    per_query_k: int = 2000,
    allowed_tables_acl: Optional[Set[str]] = None,
    ignore_views: bool = True,
    user_id: Optional[str] = None,
    strict_acl: bool = True,
    use_vector_where: bool = True,
):
    """
    Query ChromaDB for the closest tables/columns to the question, with ACL support.

    - If allowed_tables_acl = None → no ACL filter (free mode).
    - If allowed_tables_acl = set() → user has effectively no table permissions → returns [].
    """
    COLUMN_BOOST = 1.45
    VIEW_PENALTY = 0.6
    LEX_WEIGHT = 0.22
    MEAN_TOPK_K = 4
    COL_KW_WEIGHT = 0.06

    _STOP = set(
        (
            "the of a an and or by to for with in on at from as is are was were be been being "
            "into over under between within without about against through during before after "
            "than then this that these those here there it its their his her your our you we they"
        ).split()
    )

    COL_KW = {
        "amount",
        "payment",
        "payment_id",
        "payment_date",
        "rental",
        "rental_id",
        "rental_date",
        "inventory",
        "inventory_id",
        "film_id",
        "length",
        "duration",
        "price",
        "rate",
    }

    def _lexical_overlap(q: str, text: str) -> float:
        def toks(s):
            return {
                w
                for w in re.findall(r"[a-zA-Z0-9_]+", s.lower())
                if w not in _STOP and len(w) > 2
            }

        qq, tt = toks(q), toks(text)
        if not qq or not tt:
            return 0.0
        inter = len(qq & tt)
        union = len(qq | tt)
        return inter / union if union else 0.0

    def _is_view(doc_text: str, tname: str) -> bool:
        head = (doc_text or "").strip().upper()
        if head.startswith("DESCRIPTION: VIEW"):
            return True
        return any(x in tname for x in ("_list", "sales_by_", "vm_", "nicer_but_slower"))

    # If no allowed_tables_acl but user_id is provided, fetch from ACL
    if allowed_tables_acl is None and user_id is not None:
        allowed_tables_acl = get_allowed_tables_set(user_id)

    # Distinguish clearly between None and an empty set
    if allowed_tables_acl is None:
        allowed_norm = None
    else:
        allowed_norm = {_norm_name(t) for t in allowed_tables_acl}
        if not allowed_norm:
            # ACL exists but is empty → no allowed tables
            return []

    where = None
    if use_vector_where and allowed_norm is not None:
        # Let Chroma filter results by allowed tables
        where = {"table": {"$in": list(allowed_norm)}}

    res = collection.query(
        query_texts=[user_query],
        n_results=per_query_k,
        include=["metadatas", "distances", "documents"],
        where=where,
    )

    mds = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    if not mds or not dists:
        return []

    per_table_scores = defaultdict(list)
    per_table_best = {}
    table_cols_seen = defaultdict(set)

    for md, d, doc in zip(mds, dists, docs):
        if not isinstance(md, dict):
            continue
        t = md.get("table")
        if not t:
            continue

        t_norm = _norm_name(t)

        # Extra ACL filtering (defensive)
        if allowed_norm is not None and t_norm not in allowed_norm:
            if strict_acl:
                continue

        if ignore_views and _is_view(doc or "", t):
            continue

        try:
            dv = float(d)
        except (TypeError, ValueError):
            continue
        if not (dv == dv) or math.isinf(dv):
            continue

        sim = max(0.0, 1.0 - dv)

        if md.get("type") == "column":
            sim *= COLUMN_BOOST
            c = md.get("column")
            if isinstance(c, str):
                table_cols_seen[t_norm].add(c.lower())

        if _is_view(doc or "", t):
            sim *= VIEW_PENALTY

        lex = _lexical_overlap(user_query, (doc or "") + " " + (md.get("description") or ""))
        sim = sim + LEX_WEIGHT * lex

        per_table_scores[t_norm].append(sim)

        if (t_norm not in per_table_best) or (dv < per_table_best[t_norm]["dist"]):
            per_table_best[t_norm] = {"dist": dv, "doc": (doc or ""), "name": t}

    table_boosts = {}
    for t_norm, cols in table_cols_seen.items():
        hits = sum(1 for c in cols for kw in COL_KW if kw in c)
        table_boosts[t_norm] = min(3, hits) * COL_KW_WEIGHT

    ranked = []
    for t_norm, sims in per_table_scores.items():
        sims.sort(reverse=True)
        topk = sims[: max(1, min(MEAN_TOPK_K, len(sims)))]
        score = float(sum(topk) / len(topk)) + table_boosts.get(t_norm, 0.0)
        ranked.append((t_norm, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    results = []
    for t_norm, _ in ranked:
        if allowed_norm is not None and t_norm not in allowed_norm:
            if strict_acl:
                continue
        rec = per_table_best.get(
            t_norm, {"dist": float("inf"), "doc": "", "name": t_norm}
        )
        results.append(
            {
                "table": rec["name"],
                "distance": rec["dist"],
                "description": rec["doc"][:200],
            }
        )
        if len(results) >= top_k_tables:
            break

    return results


# =========================
# 5) Load schema.json
# =========================

with open("schema.json", "r") as file:
    schema_info = json.load(file)


# =========================
# 6) Groq LLM for Manager & Agents
# =========================

manager_llm = LLM(
    model=cfg.MODEL_NAME,
    api_key=cfg.GROQ_API_KEY,
    temperature=0.0
)


# =========================
# 7) Agents & Tasks with CrewAI (MySQL)
# =========================

# Natural Language Processor Agent
nlp_agent = Agent(
    role="Natural Language Processor",
    goal=(
        "Interpret the following natural language input: {natural_language_query}. "
        "You are working in a strict, ACL-enforced environment. "
        "The provided schema_info already represents ONLY the subset of tables and columns "
        "that this user is allowed to query, and that are relevant to the question. "
        "You must NEVER assume the existence of any table or column outside schema_info. "
        "Your goal is to extract the relevant information (intent, filters, groupings, time ranges, limits) "
        "that will be used later to build a MySQL-optimized SQL query strictly over schema_info."
    ),
    verbose=False,
    memory=True,
    allow_delegation=False,
    backstory=(
        "You are skilled at understanding and interpreting natural language in a secure, access-controlled "
        "analytics environment. You always respect access control limits and never reference data outside "
        "the provided schema_info."
    ),
    llm=manager_llm,
    max_iter=10,
)

interpret_nl_task = Task(
    description=(
        "Interpret the given natural language input: {natural_language_query}. "
        "Use ONLY the provided schema: {schema_info}. "
        "Extract intent, main entity (table), filters, groupings, and any time constraints. "
        "Do not invent any tables or columns; assume that schema_info is the complete universe of data "
        "available to the user. "
        "Return a clear JSON dictionary with keys: 'intent', 'parameters'."
    ),
    expected_output="A dictionary with 'intent' and 'parameters'.",
    agent=nlp_agent,
)

# SQL Generator Agent
sql_agent = Agent(
    role="SQL Generator",
    goal=(
        "Generate STRICT MySQL-optimized SQL queries based on the following natural language input: "
        "{natural_language_query}. "
        "The provided schema_info already represents ONLY the tables and columns that the user "
        "is allowed to query and that are considered relevant to the question. "
        "You MUST NOT reference ANY table or column that does not exist in schema_info. "
        "If the schema_info is insufficient to fully answer the question, you MUST still restrict yourself "
        "to schema_info and avoid guessing or reconstructing data from other imaginary tables. "
        "Always limit this SQL query to {n_rows} rows using 'LIMIT {n_rows}'."
    ),
    verbose=False,
    memory=True,
    allow_delegation=False,
    backstory=(
        "You are an expert in SQL query formation, particularly for MySQL databases in a secure environment. "
        "You NEVER bypass ACL or invent tables/columns. "
        "You generate accurate and optimized SQL queries based ONLY on structured data and the given schema_info."
    ),
    llm=manager_llm,
    max_iter=10,
)

generate_sql_task = Task(
    description=(
        "Generate a single STRICT MySQL-optimized SQL query string based on the following natural language input: "
        "{natural_language_query}. "
        "Use only the tables and columns present in the provided schema: {schema_info}. "
        "Do NOT reference any table or column outside this schema. "
        "Your final answer MUST be a valid MySQL SQL query string. "
        "Always limit this SQL query to {n_rows} rows using 'LIMIT {n_rows}'."
    ),
    expected_output="A valid and optimized SQL query string for MySQL database.",
    agent=sql_agent,
)

# Manager Agent (MySQL DBA)
DBA = Agent(
    role="MySQL DBA",
    goal=(
        "Oversee the process of interpreting natural language and generating STRICT, ACL-compliant, optimized "
        "SQL queries for a MySQL database. "
        "Coordinate between agents and ensure the final response is ONLY a validated and optimized SQL query string "
        "that uses ONLY tables and columns present in schema_info. "
        "You must also enforce semantic correctness as much as possible: "
        "if the question explicitly mentions or hints at using certain domain concepts or tables "
        "(for example: rentals, payments, customers), and these tables exist in schema_info, "
        "the final SQL must actually use them instead of approximating the answer from unrelated columns. "
        "If the available schema_info does not contain the necessary tables to respect these hints, "
        "you must still stay within schema_info and accept that a higher layer may reject the query as insufficient. "
        "Never fabricate or infer extra tables beyond schema_info."
    ),
    verbose=False,
    memory=True,
    backstory=(
        "You are an experienced MySQL DBA with deep expertise in SQL tuning and data access control. "
        "You strictly respect ACL and the semantic intent of the question, "
        "and you verify that the generated SQL truly reflects those hints when the corresponding tables are available."
    ),
    allow_delegation=True,
    llm=manager_llm,
    max_iter=10,
)

sql_query_crew = Crew(
    agents=[nlp_agent, sql_agent],
    tasks=[interpret_nl_task, generate_sql_task],
    manager_agent=DBA,
    process=Process.hierarchical,
    manager_llm=manager_llm,
)

# =========================
# 8) Main Function: NL2SQL + ACL + Vector (Strict)
# =========================

def nl2sql_with_acl(
    user_id: str,
    natural_language_query: str,
    n_rows: int = 50,
    strict_acl: bool = True,
):
    """
    Strict pipeline:
      1) Fetch user's table permissions from ACL (MySQL).
      2) Use ChromaDB (without ACL) to find ideal tables for the question.
      3) Compute the intersection between ideal tables and ACL:
         - If empty → question cannot be answered with current permissions → reject.
         - If non-empty (even one table) → restrict to those tables only, no alternative tables.
      4) Build a filtered schema on top of the intersected tables only (filtered_tables).
      5) Run CrewAI (NLP + SQL Generator + MySQL DBA Manager).
      6) Clean the SQL and finally verify that all used tables are within filtered_tables.
    """

    # 1) ACL
    allowed_tables_acl = get_allowed_tables_set(user_id)
    if not allowed_tables_acl:
        raise Exception("This user has no table permissions configured.")

    # 2) Vector DB (no ACL): ideal tables for the question
    ideal_related = find_related_tables_with_distances(
        collection=collection,
        user_query=natural_language_query,
        top_k_tables=20,
        per_query_k=2000,
        allowed_tables_acl=None,   # no ACL here
        strict_acl=False,
        use_vector_where=False,
    )

    if not ideal_related:
        raise Exception("Could not find any tables related to this question in the Vector DB.")

    # Ideal tables (no ACL)
    ideal_tables_norm = {_norm_name(r["table"]) for r in ideal_related}

    # 3) Intersection between ideal tables and ACL
    covered_tables_norm = ideal_tables_norm & allowed_tables_acl

    if not covered_tables_norm:
        # The question depends only on tables that the user is not allowed to access
        # and we do not try to approximate an answer from other tables.
        raise Exception(
            "This question cannot be answered under your current permissions, "
            "because the tables relevant to the question are not within your ACL."
        )

    # Filter ideal_related by the intersection (keep relevance order)
    related = [
        r for r in ideal_related
        if _norm_name(r["table"]) in covered_tables_norm
    ][:10]  # top 10 tables only

    if not related:
        raise Exception(
            "Failed to select allowed tables from among the tables related to this question."
        )

    question_tables = {r["table"] for r in related}
    question_tables_norm = {_norm_name(t) for t in question_tables}

    # 4) Build filtered schema on these tables only
    base_tables = schema_info.get("tables") or schema_info

    filtered_tables: Dict[str, Any] = {}
    for tname, tdef in base_tables.items():
        if _norm_name(tname) in question_tables_norm:
            filtered_tables[tname] = tdef

    if not filtered_tables:
        raise Exception(
            "Failed to build a filtered schema for the allowed tables; "
            "check schema.json or table names."
        )

    schema_filtered = {"tables": filtered_tables}

    # 5) Run the Crew
    input_data = {
        "natural_language_query": natural_language_query,
        "schema_info": schema_filtered,
        "n_rows": n_rows,
    }

    result = sql_query_crew.kickoff(inputs=input_data)

    # 6) Clean result and extract SQL
    sql_query = extract_sql_from_text(str(result)).strip()

    if not sql_query:
        raise Exception("Failed to generate a valid SQL statement.")

    # Extra guard: ensure all tables used in SQL are within filtered_tables only
    allowed_table_names_norm = {_norm_name(t) for t in filtered_tables.keys()}

    # Simple regex to capture tables after FROM or JOIN
    table_pattern = re.compile(
        r"\bFROM\s+([`\"\w\.]+)|\bJOIN\s+([`\"\w\.]+)",
        re.IGNORECASE
    )

    found_tables_norm: Set[str] = set()
    for m in table_pattern.finditer(sql_query):
        t1, t2 = m.groups()
        t = t1 or t2
        if not t:
            continue
        # Remove possible alias: schema.table AS t
        t_clean = t.split()[0]
        t_clean = t_clean.split(".")[-1]  # drop schema if present
        t_norm = _norm_name(t_clean)
        found_tables_norm.add(t_norm)

    # Reject if any table is not allowed
    for t_norm in found_tables_norm:
        if t_norm not in allowed_table_names_norm:
            raise Exception(
                f"Generated SQL contains a table that is not allowed or not present in the filtered schema: '{t_norm}'. "
                "The query has been rejected to protect access permissions."
            )

    return sql_query
