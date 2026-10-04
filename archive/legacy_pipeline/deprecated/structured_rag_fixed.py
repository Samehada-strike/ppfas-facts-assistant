# ----------------------------
# ENV SETUP
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH")

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
AMC_NAME = "PPFAS Mutual Fund"
SOURCE_LINK = "https://groww.in/mutual-funds/amc/ppfas-mutual-funds"
JSON_PATH = Path(r"E:\ML Projects\RAG_Chatbot_PP_MF\project_files\data\json_files\chunkable_ppfas.json")

# Dynamic Date Fetching
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        _data = json.load(f)
        LAST_UPDATED_DATE = _data[0].get("overview", {}).get("nav", {}).get("as_on", "Unknown Date")
        print(f"✅ LAST_UPDATED_DATE = {LAST_UPDATED_DATE}")
except Exception as e:
    print(f"⚠️ Could not load LAST_UPDATED_DATE: {e}")
    LAST_UPDATED_DATE = "Unknown Date"

# ----------------------------
# DATABASE
# ----------------------------
# Ensure this run is only if paths are set
if SQLITE_DB_PATH:
    db = SQLDatabase.from_uri(f"sqlite:///{SQLITE_DB_PATH}")
else:
    db = None
    print("⚠️ SQLITE_DB_PATH not set. Database features will fail.")

# ----------------------------
# LLM (INTENT ONLY)
# ----------------------------
llm = OpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# ----------------------------
# PII MIDDLEWARE
# ----------------------------
PII_PATTERNS = [
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                  # PAN
    r"\b[2-9]{1}[0-9]{3}\s[0-9]{4}\s[0-9]{4}\b",  # Aadhaar (fmt 1)
    r"\b[2-9]{1}[0-9]{3}[0-9]{4}[0-9]{4}\b",      # Aadhaar (fmt 2)
    r"\b\d{10}\b",                                # Phone
    r"\b(\+91[\-\s]?)?[0]?(91)?[789]\d{9}\b",     # Phone (India)
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b",                # Email
    r"\b\d{9,18}\b",                              # Account numbers
    r"\b\d{4,6}\b"                                # OTP/PIN
]

PII_REFUSAL_MESSAGE = (
    "For your privacy and security, I can’t process questions that include sensitive personal information.\n"
    "Please remove any personal identifiers and ask your question again in a general form.\n\n"
    "I answer only factual mutual fund queries related to Parag Parikh Mutual Fund schemes using official Groww information:\n"
    f"{SOURCE_LINK}"
)

def detect_pii(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in PII_PATTERNS)

# ----------------------------
# SCHEME LOADING & BM25
# ----------------------------
def load_schemes(database: SQLDatabase):
    if not database: return []
    engine = database._engine
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT scheme_id, scheme_name FROM schemes")
        ).fetchall()
    return [{"scheme_id": r[0], "scheme_name": r[1]} for r in rows]

SCHEMES = load_schemes(db)

ALIAS_MAP = {
    "ppfas": "parag parikh",
    "elss": "tax saver",
    "taxsaver": "tax saver",
    "long term": "flexi cap",
}

def canonicalize(text: str) -> str:
    t = text.lower()
    for k, v in ALIAS_MAP.items():
        t = t.replace(k, v)
    return t

if SCHEMES:
    bm25_corpus = [s["scheme_name"].lower().split() for s in SCHEMES]
    bm25 = BM25Okapi(bm25_corpus)
else:
    bm25 = None

# ----------------------------
# INTENTS
# ----------------------------
SINGLE_SCHEME_INTENTS = {
    "NAV", "MIN_SIP", "EXPENSE_RATIO", "EXIT_LOAD", "FUND_MANAGER",
    "RETURNS", "HOLDINGS", "RISK_RATIOS"
}

STRUCTURED_INTENTS = {
    "NAV", "MIN_SIP", "EXPENSE_RATIO", "EXIT_LOAD", "FUND_MANAGER",
    "RETURNS", "HOLDINGS", "RISK_RATIOS"
}

def resolve_schemes(query: str, intent: str, top_k: int = 3):
    if not bm25: return []
    q = canonicalize(query)
    query_tokens = q.split()
    scores = bm25.get_scores(query_tokens)
    
    ranked = sorted(zip(SCHEMES, scores), key=lambda x: x[1], reverse=True)
    
    if "all" in q or "schemes" in q:
        return SCHEMES
        
    if intent in SINGLE_SCHEME_INTENTS:
        best_scheme, best_score = ranked[0]
        if best_score == 0: return []
        return [best_scheme]
        
    return [scheme for scheme, score in ranked[:top_k] if score > 0]

def get_text_from_response(response) -> str:
    if isinstance(response, str): return response
    return response.content

INTENT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Classify the user's question into ONE intent.

STRUCTURED (factual, numeric, database):
- NAV
- MIN_SIP
- EXPENSE_RATIO
- EXIT_LOAD
- FUND_MANAGER
- RETURNS
- HOLDINGS
- RISK_RATIOS

UNSTRUCTURED (explanatory, descriptive):
- INVESTMENT_PHILOSOPHY
- INVESTMENT_APPROACH
- GENERAL_INFO

OTHER:
- OPINIONATED
- VAGUE
- UNKNOWN

User question:
{question}

Return ONLY the intent label.
"""
)

def detect_intent(question: str) -> str:
    response = llm.invoke(INTENT_PROMPT.format(question=question))
    return get_text_from_response(response).strip().upper()

# ----------------------------
# SQL HANDLERS
# ----------------------------
def run_query(query: str, params: dict):
    if not db: return []
    engine = db._engine
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        return result.fetchall()

def make_in_clause(values):
    placeholders = ",".join([f":v{i}" for i in range(len(values))])
    params = {f"v{i}": v for i, v in enumerate(values)}
    return placeholders, params

def fetch_min_sip(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"SELECT s.scheme_name, m.min_sip FROM minimum_investment m JOIN schemes s ON m.scheme_id = s.scheme_id WHERE m.scheme_id IN ({placeholders})"
    return run_query(q, params)

def fetch_expense_ratio(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"SELECT s.scheme_name, e.expense_ratio FROM expense_exit_tax e JOIN schemes s ON e.scheme_id = s.scheme_id WHERE e.scheme_id IN ({placeholders})"
    return run_query(q, params)

def fetch_exit_load(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"SELECT s.scheme_name, e.exit_load FROM expense_exit_tax e JOIN schemes s ON e.scheme_id = s.scheme_id WHERE e.scheme_id IN ({placeholders})"
    return run_query(q, params)

def fetch_nav(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"SELECT s.scheme_name, n.nav_value_raw, n.nav_as_on FROM nav_summary n JOIN schemes s ON n.scheme_id = s.scheme_id WHERE n.scheme_id IN ({placeholders})"
    return run_query(q, params)

def fetch_fund_managers(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"SELECT s.scheme_name, f.manager_name FROM fund_managers f JOIN schemes s ON f.scheme_id = s.scheme_id WHERE f.scheme_id IN ({placeholders})"
    return run_query(q, params)

def fetch_returns(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"""
    SELECT s.scheme_name, r.period_label, r.value
    FROM return_values r
    JOIN schemes s ON r.scheme_id = s.scheme_id
    WHERE r.scheme_id IN ({placeholders})
    AND r.metric_label = 'Fund returns'
    ORDER BY s.scheme_name, r.period_label
    """
    return run_query(q, params)

def fetch_holdings(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"""
    SELECT s.scheme_name, h.name, h.assets
    FROM holdings h
    JOIN schemes s ON h.scheme_id = s.scheme_id
    WHERE h.scheme_id IN ({placeholders})
    ORDER BY s.scheme_name, CAST(REPLACE(h.assets, '%', '') AS FLOAT) DESC
    LIMIT 10
    """
    return run_query(q, params)

def fetch_ratios(scheme_ids):
    placeholders, params = make_in_clause(scheme_ids)
    q = f"""
    SELECT s.scheme_name, a.alpha, a.beta, a.sharpe_ratio, a.standard_deviation
    FROM advanced_ratios a
    JOIN schemes s ON a.scheme_id = s.scheme_id
    WHERE a.scheme_id IN ({placeholders})
    """
    return run_query(q, params)

# ----------------------------
# RESPONSE FORMATTING
# ----------------------------
OPINIONATED_RESPONSE = (
    "I can’t help with investment advice, recommendations, or performance comparisons. "
    "I’m designed to answer factual questions about Parag Parikh mutual fund schemes only. "
    f"Last updated from sources: {SOURCE_LINK}\n"
    f"Data as of: {LAST_UPDATED_DATE}"
)

def format_answer(rows, intent):
    footer = f"\nLast updated from sources: {SOURCE_LINK}\nData as of: {LAST_UPDATED_DATE}"
    
    if not rows:
        return f"I could not find the requested information in the structured database. " + footer

    if intent == "NAV":
        scheme, nav, date = rows[0]
        return f"The latest NAV for {scheme} is {nav} as of {date}. " + footer

    # General Single-Row Handler
    if len(rows) == 1 and len(rows[0]) == 2:
        scheme, value = rows[0]
        return f"The requested information for {scheme} is {value}. " + footer
        
    # List Handler
    lines = [f"- {r[0]}: {r[1]}" + (f" ({r[2]})" if len(r) > 2 else "") for r in rows]
    return "Here is the requested information:\n" + "\n".join(lines) + footer

# ----------------------------
# MAIN EXECUTION CHAIN
# ----------------------------
def structured_rag_chain(query: str) -> str | None:
    if detect_pii(query):
        return PII_REFUSAL_MESSAGE

    intent = detect_intent(query)
    
    if intent not in STRUCTURED_INTENTS:
        return None

    if intent == "OPINIONATED":
        return OPINIONATED_RESPONSE

    schemes = resolve_schemes(query, intent)
    if not schemes:
        return None

    scheme_ids = [s["scheme_id"] for s in schemes]
    
    if intent == "MIN_SIP": rows = fetch_min_sip(scheme_ids)
    elif intent == "EXPENSE_RATIO": rows = fetch_expense_ratio(scheme_ids)
    elif intent == "EXIT_LOAD": rows = fetch_exit_load(scheme_ids)
    elif intent == "NAV": rows = fetch_nav(scheme_ids)
    elif intent == "FUND_MANAGER": rows = fetch_fund_managers(scheme_ids)
    elif intent == "RETURNS": rows = fetch_returns(scheme_ids)
    elif intent == "HOLDINGS": rows = fetch_holdings(scheme_ids)
    elif intent == "RISK_RATIOS": rows = fetch_ratios(scheme_ids)
    else: return None

    return format_answer(rows, intent)
