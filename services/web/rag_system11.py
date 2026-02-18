"""
CMC Thesis Chatbot — v9
========================
Changes from v8:
  - FIXED:   person_lookup now runs two separate SQL queries (as advisor, as author)
             so it correctly counts ALL theses advised, not just ones where the
             person appears in the same row as author
  - FIXED:   respond() for topic_search now summarizes EACH thesis individually
             instead of writing one generic summary at the top
  - FIXED:   title_lookup typing a bare thesis title now correctly routes to
             title_lookup (bare title detection moved into classify() prompt)
  - CHANGED: list view response format — each thesis gets its own mini-summary
             drawn from its keywords/abstract/title rather than a generic header

Architecture: classify → fetch → respond (unchanged)
Vector index: unchanged, no need to re-run --build-index.

Install:
    pip install chromadb sentence-transformers groq numpy pyyaml

Usage:
    python rag_system17.py --build-index   # only if not yet indexed
    python rag_system17.py
"""

import sqlite3, logging, argparse, json, os, re, yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DB_PATH         = "theses2.db"
CHROMA_DIR      = "./chroma_store"
COLLECTION_NAME = "theses"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CONFIG_PATH     = Path("config.yaml")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("📦 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model ready.\n")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ACRONYMS = {
    "rag": "retrieval augmented generation RAG vector search embeddings",
    "llm": "large language model LLM GPT artificial intelligence",
    "nlp": "natural language processing NLP text machine learning",
    "ml" : "machine learning ML artificial intelligence neural networks",
    "ai" : "artificial intelligence AI machine learning deep learning",
    "cv" : "computer vision CV image recognition deep learning",
    "rl" : "reinforcement learning RL reward optimization",
    "dl" : "deep learning DL neural networks",
    "api": "application programming interface API software engineering",
    "ui" : "user interface UI design front end",
    "ux" : "user experience UX design usability",
    "etf": "exchange traded fund ETF finance investing",
    "esg": "environmental social governance ESG investing",
    "gdp": "gross domestic product GDP economics",
    "fed": "federal reserve monetary policy interest rates",
    "gan": "generative adversarial network GAN image synthesis deep learning",
    "var": "value at risk VaR financial risk quantitative finance",
    "cpi": "consumer price index CPI inflation economics",
    "eeg": "electroencephalography EEG neuroscience brain signals",
}

def load_acronyms() -> dict:
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as f:
            yaml.dump({"acronyms": DEFAULT_ACRONYMS}, f, sort_keys=True)
        return DEFAULT_ACRONYMS
    loaded = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {**DEFAULT_ACRONYMS, **loaded.get("acronyms", {})}

ACRONYMS = load_acronyms()


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_columns() -> list[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(theses)").fetchall()]
        conn.close()
        return cols
    except Exception:
        return []

COLUMNS = get_columns()

def run_sql(query: str, params: list = []) -> list[dict]:
    """Run SQL and return list of dicts keyed by column name."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA case_sensitive_like = OFF")
        rows = conn.execute(query, params).fetchall()
        conn.close()
        if not rows or not COLUMNS:
            return []
        return [dict(zip(COLUMNS, row)) for row in rows]
    except Exception as e:
        logger.error(f"SQL error: {e}")
        return []

def clean_row(row: dict, keep_abstract: bool = False) -> dict:
    """Map raw DB column names to clean keys, drop empty values."""
    mapping = {
        "Title"            : "title",
        "author"           : "author",
        "author 2"         : "author2",
        "advisor1"         : "advisor1",
        "advisor2"         : "advisor2",
        "advisor3"         : "advisor3",
        "department"       : "department",
        "second_department": "second_department",
        "keywords"         : "keywords",
        "disciplines"      : "disciplines",
        "abstract"         : "abstract",
        "award"            : "award",
        "season"           : "season",
        "First published"  : "first_published",
        "publication_date" : "publication_date",
        "URL"              : "url",
    }
    result = {}
    for db_key, clean_key in mapping.items():
        if not keep_abstract and clean_key == "abstract":
            continue
        val = row.get(db_key) or row.get(clean_key)
        if val and str(val).strip():
            result[clean_key] = str(val).strip()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Vector search  (uses existing ChromaDB index — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def vector_search(query: str, topics: list[str], n: int = 20) -> list[dict]:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    enriched   = query + (" " + " ".join(topics[:3]) if topics else "")
    embedding  = embedder.encode([enriched])[0].tolist()
    results    = collection.query(
        query_embeddings=[embedding], n_results=n,
        include=["metadatas", "distances"]
    )
    hits = []
    if results and results["metadatas"]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            hits.append({**meta, "_similarity": round(1 - dist, 3)})
    return hits

def rerank(candidates: list[dict], query: str, topics: list[str], top_n: int = 15) -> list[dict]:
    if not candidates:
        return []
    enriched = query + (" " + " ".join(topics[:3]) if topics else "")
    texts = [
        " | ".join(filter(None, [
            c.get("title",""), c.get("keywords","")[:200], c.get("abstract","")[:300]
        ])) for c in candidates
    ]
    q_emb  = embedder.encode([enriched])[0]
    c_embs = embedder.encode(texts)
    scores = [
        float(np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e) + 1e-9))
        for e in c_embs
    ]
    for score, c in zip(scores, candidates):
        c["_similarity"] = round(score, 3)
    return sorted(candidates, key=lambda x: x["_similarity"], reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Year filter builder
# ─────────────────────────────────────────────────────────────────────────────
def year_clause(years: list[str], operator: str) -> tuple[str, list]:
    if not years:
        return "", []
    yr = 'CAST(SUBSTR("First published", LENGTH("First published")-1, 2) AS INTEGER)'
    ops = {
        "after" : (f"{yr} > ?",             [int(years[0]) % 100]),
        "before": (f"{yr} < ?",             [int(years[0]) % 100]),
        "range" : (f"{yr} BETWEEN ? AND ?", [int(years[0]) % 100, int(years[-1]) % 100])
                   if len(years) >= 2 else ("", []),
    }
    if operator in ops:
        return ops[operator]
    placeholders = ",".join("?" * len(years[:4]))
    return f"{yr} IN ({placeholders})", [int(y) % 100 for y in years[:4]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────
def classify(question: str) -> dict:
    """Single LLM call: intent + entities. Prompt is the only place we tune."""
    acronym_hints = "\n".join(
        f"  {k.upper()} → \"{v}\"" for k, v in list(ACRONYMS.items())[:20]
    )

    prompt = f"""You are a query parser for a college senior thesis database.
Classify the query and extract entities. Return ONLY valid JSON, no markdown.

Query: "{question}"

━━━ INTENT RULES ━━━
title_lookup  → user references a SPECIFIC thesis by title OR types a title directly.
                A bare title: 5+ words, mostly title-cased, no question/search word at start.
                e.g. "what is [title] about", "summarize [title]", "tell me about [title]",
                     "Containing Compounding Container Congestion",
                     "A New Experiment on Rational Behavior"
topic_search  → searching by subject/concept, or asking for guidance/ideas
                e.g. "theses about climate change", "give me ideas about AI"
person_lookup → searching by a person's name (advisor or author)
                e.g. "who is Mark Huber", "mark huber", "theses by Jane Smith"
aggregation   → counting or ranking
                e.g. "who advised the most theses", "how many in 2021",
                     "which department has the most"

━━━ MODE RULES ━━━
retrieval → user wants to SEE real records (default)
guidance  → user wants ADVICE or IDEAS
            keywords: "give me ideas", "suggest", "recommend", "who should I ask",
            "help me", "brainstorm", "what should I write"

━━━ PERSON DETECTION ━━━
If the query is just a person's name (1-3 words, no verbs, proper-cased), classify as person_lookup.
e.g. "mark huber", "Jane Smith", "David Bjerk" → person_lookup

━━━ ACRONYM EXPANSION (expand in topics, NOT for title_lookup) ━━━
{acronym_hints}

━━━ OUTPUT FORMAT ━━━
{{
  "intent": "title_lookup|topic_search|person_lookup|aggregation",
  "mode": "retrieval|guidance",
  "topics": ["expanded keyword strings — for title_lookup put title words AS-IS"],
  "names": ["person names only"],
  "years": ["4-digit years"],
  "year_op": "after|before|range|exact|none",
  "seasons": ["Spring or Fall if mentioned"],
  "award": true/false,
  "department": "department name or null",
  "role": "advisor|author|both|unknown"
}}

━━━ EXAMPLES ━━━
"mark huber"
  → intent: person_lookup, names: ["Mark Huber"], role: advisor

"what is Containing Compounding Container Congestion about"
  → intent: title_lookup, topics: ["Containing Compounding Container Congestion"]

"Revisiting the Minimum Wage-Employment Debate Using Univariate Regressions"
  → intent: title_lookup, topics: ["Revisiting the Minimum Wage-Employment Debate Using Univariate Regressions"]

"theses about RAG systems after 2020"
  → intent: topic_search, topics: ["retrieval augmented generation RAG vector search"], years: ["2020"], year_op: after

"who should I ask to advise my LLM thesis"
  → intent: topic_search, mode: guidance, topics: ["large language model LLM GPT"]

"who advised the most behavioral economics theses"
  → intent: aggregation, topics: ["behavioral economics decision making"]

"award-winning climate theses after 2019"
  → intent: topic_search, topics: ["climate change environment"], award: true, years: ["2019"], year_op: after"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=350
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        m    = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group(0) if m else raw)
    except Exception as e:
        logger.error(f"classify() failed: {e}")
        data = {}

    return {
        "intent"    : data.get("intent", "topic_search"),
        "mode"      : data.get("mode", "retrieval"),
        "topics"    : data.get("topics", []),
        "names"     : data.get("names", []),
        "years"     : data.get("years", []),
        "year_op"   : data.get("year_op", "none"),
        "seasons"   : data.get("seasons", []),
        "award"     : bool(data.get("award", False)),
        "department": data.get("department") or None,
        "role"      : data.get("role", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FETCH
# Pure data retrieval — zero LLM calls.
# ─────────────────────────────────────────────────────────────────────────────
def fetch(question: str, p: dict) -> tuple[str, list]:
    """Route to right data source. Returns (intent, records)."""
    intent = p["intent"]

    # ── title_lookup ──────────────────────────────────────────────────────────
    if intent == "title_lookup":
        title_words = " ".join(p["topics"]) if p["topics"] else question
        noise = {"what","is","about","tell","me","summarize","explain","describe",
                 "the","a","an","thesis","paper","study","this","please"}
        clean = " ".join(w for w in title_words.lower().split() if w not in noise)

        rows = run_sql(
            'SELECT * FROM theses WHERE Title LIKE ? ORDER BY "First published" DESC LIMIT 5',
            [f"%{clean}%"]
        )
        if rows and p["names"]:
            name     = p["names"][0].lower()
            filtered = [r for r in rows if name in str(r).lower()]
            rows     = filtered or rows

        if rows:
            return "title_lookup", [clean_row(r, keep_abstract=True) for r in rows[:3]]

        # Fallback: vector search
        hits = vector_search(question, p["topics"], n=3)
        return "title_lookup", hits

    # ── person_lookup ─────────────────────────────────────────────────────────
    # FIX: run two separate queries so we get ALL theses where this person
    # appears as advisor (any of advisor1/2/3) AND as author separately.
    # Previous version merged them in one query which missed many records.
    if intent == "person_lookup":
        if not p["names"]:
            return "topic_search", vector_search(question, p["topics"])

        name = p["names"][0]
        like = f"%{name}%"

        advised = run_sql(
            'SELECT * FROM theses WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? '
            'ORDER BY "First published" DESC',
            [like, like, like]
        )
        authored = run_sql(
            'SELECT * FROM theses WHERE author LIKE ? OR "author 2" LIKE ? '
            'ORDER BY "First published" DESC',
            [like, like]
        )

        # Return both sets packed together with a separator so respond() can split them
        return "person_lookup", {
            "name"    : name,
            "advised" : [clean_row(r, keep_abstract=True) for r in advised],
            "authored": [clean_row(r, keep_abstract=True) for r in authored],
        }

    # ── aggregation ───────────────────────────────────────────────────────────
    if intent == "aggregation":
        result = _aggregate(question, p)
        return "aggregation", [result]

    # ── topic_search (+ optional SQL filters) ─────────────────────────────────
    where, params = [], []
    if p["award"]:
        where.append("(award IS NOT NULL AND TRIM(award) != '')")
    if p["years"]:
        clause, yp = year_clause(p["years"], p["year_op"])
        if clause:
            where.append(f"({clause})")
            params.extend(yp)
    if p["seasons"]:
        where.append("season LIKE ?")
        params.append(f"%{p['seasons'][0]}%")
    if p["department"]:
        where.append("(department LIKE ? OR second_department LIKE ?)")
        params.extend([f"%{p['department']}%"] * 2)

    if where:
        sql_rows   = run_sql(
            f'SELECT * FROM theses WHERE {" AND ".join(where)} ORDER BY "First published" DESC LIMIT 300',
            params
        )
        candidates = [clean_row(r, keep_abstract=True) for r in sql_rows]
        results    = rerank(candidates, question, p["topics"], top_n=15)
    else:
        # Pure semantic — fetch with abstracts for richer summaries
        raw_hits = vector_search(question, p["topics"], n=20)
        # Hydrate abstracts from SQL for vector hits (meta only stores 500 chars)
        results  = _hydrate_abstracts(raw_hits)

    return "topic_search", results


def _hydrate_abstracts(hits: list[dict]) -> list[dict]:
    """
    Vector search metadata truncates abstracts at 500 chars.
    For list views where we want per-thesis summaries, fetch full abstracts from SQL.
    """
    if not hits:
        return hits
    hydrated = []
    for h in hits:
        title = h.get("title", "")
        if title:
            rows = run_sql(
                "SELECT * FROM theses WHERE Title LIKE ? LIMIT 1",
                [f"%{title[:60]}%"]   # partial match in case of minor formatting differences
            )
            if rows:
                full = clean_row(rows[0], keep_abstract=True)
                full["_similarity"] = h.get("_similarity", 0)
                hydrated.append(full)
                continue
        hydrated.append(h)
    return hydrated


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (no LLM during data fetch — LLM only for final formatting)
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(question: str, p: dict) -> str:
    q       = question.lower()
    yr_expr = 'CAST(SUBSTR("First published", LENGTH("First published")-1, 2) AS INTEGER)'

    # ── Advisor ranking ───────────────────────────────────────────────────────
    if any(w in q for w in ["advis","professor","faculty","supervised","most theses"]):
        if p["topics"]:
            hits  = vector_search(question, p["topics"], n=100)
            stats : dict = defaultdict(lambda: {"n":0,"yr":0,"depts":set(),"titles":[]})
            for h in hits:
                m  = re.search(r'\b(19|20)\d{2}\b', h.get("first_published",""))
                yr = int(m.group(0)) if m else 0
                for k in ("advisor1","advisor2","advisor3"):
                    adv = h.get(k,"").strip()
                    if adv:
                        stats[adv]["n"] += 1
                        stats[adv]["yr"] = max(stats[adv]["yr"], yr)
                        if h.get("department"): stats[adv]["depts"].add(h["department"])
                        t = h.get("title","")
                        if t and t not in stats[adv]["titles"]: stats[adv]["titles"].append(t)
            cur  = datetime.now().year
            data = sorted([{
                "advisor":a,"thesis_count":s["n"],"latest_year":s["yr"],
                "still_active":(cur-s["yr"])<=3 if s["yr"] else False,
                "departments":list(s["depts"]),"sample_theses":s["titles"][:2],
            } for a,s in stats.items()], key=lambda x:x["thesis_count"], reverse=True)[:15]
            label = ", ".join(p["topics"][:2])
        else:
            dept_c, dept_p = "", []
            if p["department"]:
                dept_c = "AND (department LIKE ? OR second_department LIKE ?)"
                dept_p = [f"%{p['department']}%"]*2
            rows = run_sql(f"""
SELECT advisor1, COUNT(*) as cnt, MAX("First published") as latest,
       GROUP_CONCAT(DISTINCT department) as depts
FROM theses WHERE advisor1 IS NOT NULL AND TRIM(advisor1)!='' {dept_c}
GROUP BY advisor1 ORDER BY cnt DESC LIMIT 15""", dept_p)
            cur  = datetime.now().year
            data = []
            for r in rows:
                adv,cnt,latest,depts = r.get("advisor1"),r.get("cnt"),r.get("latest",""),r.get("depts","")
                m  = re.search(r'\b(19|20)\d{2}\b', latest or "")
                yr = int(m.group(0)) if m else 0
                data.append({"advisor":adv,"thesis_count":cnt,"latest_year":yr,
                              "still_active":(cur-yr)<=3 if yr else False,
                              "departments":(depts or "").split(",")[:3],"sample_theses":[]})
            label = p["department"] or "all departments"

        if not data:
            return "No advisors found for that query."
        return llm(f"""Question: {question}

CRITICAL: Use ONLY advisors in the JSON. Do not add names from outside this list.
Do NOT draw on knowledge of famous researchers from your training data.

Advisors for "{label}":
{json.dumps(data, indent=2)}

Conversational answer using ONLY names above:
1. One sentence summarising the finding.
2. Ranked list (up to 10): name, count, ✅ active or ⚠️ last seen [year], departments, sample thesis.
3. Whether the top advisor is currently active.""", max_tokens=900)

    # ── Department count ──────────────────────────────────────────────────────
    if any(w in q for w in ["department","dept","field","major"]):
        rows = run_sql("""SELECT department, COUNT(*) as cnt FROM theses
WHERE department IS NOT NULL AND TRIM(department)!=''
GROUP BY department ORDER BY cnt DESC LIMIT 15""")
        data = [{"department":r.get("department"),"count":r.get("cnt")} for r in rows]
        return llm(f"Question: {question}\nCRITICAL: Use ONLY these departments.\n{json.dumps(data)}\nRanked list.", max_tokens=500)

    # ── Year count ────────────────────────────────────────────────────────────
    years = p["years"]
    if len(years) >= 2:
        counts = {}
        for yr in years[:4]:
            r = run_sql(f"SELECT COUNT(*) as cnt FROM theses WHERE {yr_expr}=?", [int(yr)%100])
            counts[yr] = r[0].get("cnt",0) if r else 0
        return llm(f"Question: {question}\nCounts: {json.dumps(counts)}\nCompare directly.", max_tokens=300)

    def to_full(y): return 2000+y if y<=25 else 1900+y
    rows = run_sql(f"""SELECT {yr_expr} as y, COUNT(*) as cnt FROM theses
WHERE "First published" IS NOT NULL AND LENGTH("First published")>=5
GROUP BY y ORDER BY y DESC LIMIT 15""")
    data = [{"year":to_full(r.get("y",0)),"count":r.get("cnt",0)} for r in rows if r.get("y") is not None]
    return llm(f"Question: {question}\nCounts by year: {json.dumps(data)}\nTrending up or down?", max_tokens=400)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RESPOND
# ─────────────────────────────────────────────────────────────────────────────
def respond(question: str, intent: str, mode: str, records) -> str:
    """Format natural language answer from fetched records."""

    # Aggregation already pre-formatted
    if intent == "aggregation":
        return records[0] if records else "No results found."

    # ── person_lookup: full profile with per-thesis summaries ─────────────────
    if intent == "person_lookup":
        # records is a dict with "name", "advised", "authored"
        if not isinstance(records, dict):
            return "Could not retrieve profile."

        name    = records["name"]
        advised = records["advised"]
        authored= records["authored"]

        if not advised and not authored:
            return f"I couldn't find **{name}** in the thesis database."

        # Build compact per-thesis entries including abstract snippet
        def summarise_list(theses: list[dict], limit: int = 20) -> list[dict]:
            return [
                {
                    "title"   : t.get("title",""),
                    "year"    : t.get("first_published","") or t.get("publication_date",""),
                    "dept"    : t.get("department",""),
                    "award"   : t.get("award",""),
                    "abstract": (t.get("abstract","") or "")[:300],
                    "keywords": (t.get("keywords","") or "")[:150],
                    "author"  : t.get("author",""),
                    "advisor1": t.get("advisor1",""),
                }
                for t in theses[:limit]
            ]

        profile_data = {
            "name"         : name,
            "advised_count": len(advised),
            "authored_count": len(authored),
            "advised_theses": summarise_list(advised),
            "authored_theses": summarise_list(authored, limit=5),
        }

        return llm(
f"""User searched for: "{question}"

Database profile for {name}:
{json.dumps(profile_data, indent=2)}

CRITICAL: Use ONLY data from the JSON. Do not add outside knowledge.

Write a conversational profile:
1. One-sentence summary: advisor, author, or both — and total counts.
2. If advisor (advised_count > 0):
   - Date range of advising
   - Departments covered
   - For EACH thesis in advised_theses, write one sentence about what it covers
     (use abstract if available, otherwise infer from title/keywords)
   - Note any award-winning theses with 🏆
3. If author (authored_count > 0):
   - List authored theses with year and one sentence on each
4. Close with whether they appear to be currently active (check most recent year).""",
            max_tokens=1800
        )

    # ── title_lookup: full summary with abstract ──────────────────────────────
    if intent == "title_lookup":
        if not records:
            return f"I couldn't find that thesis in the database."

        record = records[0]
        others = [r.get("title","") for r in records[1:] if r.get("title")]
        note   = f"\n\nAlso matched: {'; '.join(others)}. Ask me about any of them." if others else ""

        return llm(
f"""User asked: "{question}"

Thesis record:
{json.dumps(record, indent=2)}

Write a 3-5 sentence summary:
- What it investigates or argues (use abstract if available)
- Methodology or approach used
- Key findings or significance
- Author, advisor, department, year
- Note 🏆 if award-winning
Use ONLY the data above. Do not invent details.{note}""",
            max_tokens=450
        )

    # ── guidance (topic_search in guidance mode) ──────────────────────────────
    if mode == "guidance":
        advisor_stats: dict = defaultdict(lambda: {"n":0,"yr":0,"depts":set(),"titles":[]})
        for h in records:
            m  = re.search(r'\b(19|20)\d{2}\b', h.get("first_published",""))
            yr = int(m.group(0)) if m else 0
            for k in ("advisor1","advisor2","advisor3"):
                adv = h.get(k,"").strip()
                if adv:
                    advisor_stats[adv]["n"] += 1
                    advisor_stats[adv]["yr"] = max(advisor_stats[adv]["yr"], yr)
                    if h.get("department"): advisor_stats[adv]["depts"].add(h["department"])
                    t = h.get("title","")
                    if t and t not in advisor_stats[adv]["titles"]: advisor_stats[adv]["titles"].append(t)

        cur      = datetime.now().year
        advisors = sorted([{
            "advisor":a,"count":s["n"],"latest_year":s["yr"],
            "is_active":(cur-s["yr"])<=3 if s["yr"] else False,
            "departments":list(s["depts"]),"theses":s["titles"][:3],
        } for a,s in advisor_stats.items()], key=lambda x:x["count"], reverse=True)[:10]

        related = [{"title":r.get("title"),"department":r.get("department"),
                    "advisor":r.get("advisor1"),"abstract":(r.get("abstract","") or "")[:200]}
                   for r in records[:8]]

        q = question.lower()
        if any(w in q for w in ["who should","who can","who to ask","best advisor","advise my"]):
            task = """Recommend advisors using ONLY names from the advisors JSON.
For each: name (exact), ✅ active or ⚠️ last seen [year], count, departments, 1-2 thesis titles.
End with: suggest emailing top 2-3 active advisors.
CRITICAL: Do not invent advisor names not in the JSON."""
        else:
            task = """Generate 3-5 thesis ideas inspired by related_theses.
Format each as:
**Title**: ...
**Overview**: 2-3 sentences
**Research Questions**: 3 bullet points
**Methodology**: achievable in ONE semester by an undergrad
**Significance**: why it matters
**Potential Advisor**: EXACT name from advisors JSON only
**Timeline**: One semester (3-4 months)
CRITICAL: Advisor must be verbatim from advisors JSON. If none fits, write "Consult department"."""

        return llm(
f"""User asked: "{question}"

related_theses (real DB data):
{json.dumps(related, indent=2)}

advisors (real DB data — ONLY names you may use):
{json.dumps(advisors, indent=2)}

{task}""",
            max_tokens=1400, temperature=0.3
        )

    # ── topic_search: per-thesis summaries ────────────────────────────────────
    # FIX: instead of one generic header, summarise each thesis individually
    if not records:
        return llm(
            f'No results for: "{question}". Briefly acknowledge and suggest broader terms.',
            max_tokens=120
        )

    # Build a compact record for each thesis including abstract snippet
    display = []
    for r in records[:10]:
        entry = {
            "title"   : r.get("title",""),
            "author"  : r.get("author",""),
            "advisor" : " / ".join(filter(None,[r.get("advisor1"),r.get("advisor2"),r.get("advisor3")])),
            "year"    : r.get("first_published","") or r.get("publication_date",""),
            "dept"    : r.get("department",""),
            "abstract": (r.get("abstract","") or "")[:300],
            "keywords": (r.get("keywords","") or "")[:150],
        }
        if r.get("award"):
            entry["award"] = r["award"]
        if r.get("_similarity"):
            entry["relevance"] = f"{r['_similarity']:.0%}"
        display.append({k:v for k,v in entry.items() if v})

    return llm(
f"""User asked: "{question}"

Found {len(records)} results. Here are the top {len(display)}:
{json.dumps(display, indent=2)}

For EACH thesis in the list above, write a numbered entry with:
- **Title** (mark 🏆 if award field is present)
- Author, advisor, year, department
- 1-2 sentences on what it's about (use abstract if available, otherwise infer from title/keywords)

Do NOT write a single summary for all of them. Each thesis gets its own entry.
Use ONLY data from the JSON above.""",
        max_tokens=1400
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────
def llm(prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=temperature, max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "I'm having trouble responding right now. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# Index builder — run once with --build-index
# ─────────────────────────────────────────────────────────────────────────────
def build_vector_index():
    print("🔨 Building vector index from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM theses").fetchall()
    conn.close()

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"}
    )

    ids, docs, metas = [], [], []
    for i, row in enumerate(rows):
        d = dict(row)
        parts = [
            f"Title: {d['Title'].strip()}"               if d.get("Title")      else "",
            f"Department: {d['department'].strip()}"     if d.get("department") else "",
            f"Disciplines: {d['disciplines'].strip()}"   if d.get("disciplines")else "",
            f"Keywords: {d['keywords'][:300].strip()}"   if d.get("keywords")   else "",
            f"Abstract: {d['abstract'][:400].strip()}"   if d.get("abstract")   else "",
            f"Full Text: {d['full_text'][:600].strip()}" if d.get("full_text")  else "",
        ]
        doc_text = " | ".join(p for p in parts if p) or d.get("Title") or f"Thesis {i}"

        meta = {k: str(v or "")[:500] for k,v in {
            "title": d.get("Title"), "author": d.get("author"),
            "advisor1": d.get("advisor1"), "advisor2": d.get("advisor2"),
            "advisor3": d.get("advisor3"), "department": d.get("department"),
            "second_department": d.get("second_department"),
            "keywords": d.get("keywords","")[:300], "abstract": d.get("abstract","")[:500],
            "award": d.get("award"), "season": d.get("season"),
            "first_published": d.get("First published"), "url": d.get("URL"),
            "disciplines": d.get("disciplines","")[:300],
        }.items()}

        ids.append(str(i)); docs.append(doc_text); metas.append(meta)

        if len(ids) == 256:
            embs = embedder.encode(docs, show_progress_bar=False).tolist()
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            print(f"  ↳ {i+1}/{len(rows)}")
            ids, docs, metas = [], [], []

    if ids:
        embs = embedder.encode(docs, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    print(f"✅ Indexed {len(rows)} theses.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        count = chroma_client.get_collection(COLLECTION_NAME).count()
        print(f"✅ Vector index loaded — {count} theses indexed.")
    except Exception:
        print("⚠️  No vector index. Run: python rag_system17.py --build-index")
        return

    print(f"\n🎓 CMC Thesis Chatbot (v9)")
    print(f"📋 {len(ACRONYMS)} acronyms loaded from config.yaml")
    print("=" * 55)
    print("  'mark huber'                      → full advisor profile")
    print("  'what is [title] about'            → thesis summary")
    print("  'theses about climate after 2019'  → per-thesis summaries")
    print("  'who advised the most ML theses'   → rankings")
    print("  'give me ideas about ESG'          → guided suggestions")
    print("=" * 55)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("💬 You: ").strip()
            if not question or question.lower() in ("exit","quit","q"):
                print("👋 Goodbye!")
                break

            parsed         = classify(question)
            intent, records= fetch(question, parsed)
            answer         = respond(question, intent, parsed["mode"], records)

            print(f"\n   [{intent} → {parsed['mode']}]")
            print("═" * 65)
            print(answer)
            print("═" * 65 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("❌ Something went wrong. Please try again.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()
    build_vector_index() if args.build_index else main()
