"""Central configuration: every path, model name, and constant lives here.

Paths are derived from this file's location, so the project runs from any
folder or machine (no hard-coded E:\\ paths).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# === PATHS ===
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "project_files" / "data"

# Offline inputs/outputs (ingest)
SCHEMES_CONFIG_PATH = DATA_DIR / "json_files" / "schemes_config.json"
RAW_HTML_DIR = DATA_DIR / "raw" / "HTML"
RAW_MD_DIR = DATA_DIR / "raw" / "Markdown"
CLEAN_MD_DIR = DATA_DIR / "raw" / "Markdown_cleaned"
CRAWL_MANIFEST_PATH = DATA_DIR / "json_files" / "crawl_manifest.json"
STRUCTURED_JSON_PATH = DATA_DIR / "json_files" / "parag_parikh_structured_db.json"
CHUNKABLE_JSON_PATH = DATA_DIR / "json_files" / "chunkable_ppfas.json"
SQLITE_PATH = DATA_DIR / "structured_data" / "structured.sqlite"

# Knowledge pages (Groww help-centre MF FAQs + blogs): a separate crawl job
SUPPORT_SITEMAP_URL = "https://groww.in/support-sitemap.xml"
KNOWLEDGE_URL_PREFIX = "https://groww.in/help/mutual-funds/"
EXTRA_KNOWLEDGE_URLS = [
    "https://groww.in/blog/what-are-mutual-funds",
    "https://groww.in/blog/mutual-funds-things-you-should-know-as-a-beginner",
]
KNOWLEDGE_URLS_PATH = DATA_DIR / "json_files" / "knowledge_urls.json"
RAW_KNOWLEDGE_DIR = DATA_DIR / "raw" / "knowledge"

# Extraction outputs (rebuilt from raw files; replace the notebook-era JSONs above)
PROCESSED_DIR = DATA_DIR / "processed"
SCHEME_FACTS_PATH = PROCESSED_DIR / "scheme_facts.json"
SCHEME_FAQS_PATH = PROCESSED_DIR / "scheme_faqs.json"
KNOWLEDGE_PATH = PROCESSED_DIR / "knowledge.json"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"  # hand-off to Phase 2 (chunking + embedding)
SOURCES_CSV_PATH = REPO_ROOT / "sources.csv"         # deliverable: every source URL used

# Offline outputs (index), loaded by the online app
FAISS_INDEX_DIR = DATA_DIR / "index" / "faiss"
INDEX_INFO_PATH = DATA_DIR / "index" / "index_info.json"
CHUNKS_PATH = DATA_DIR / "index" / "chunks.jsonl"  # exactly what gets embedded (readable)

# === CHUNKING (sizes in tokens, measured with the embedding model's tokenizer) ===
TOKENIZER = "cl100k_base"     # tokenizer used by text-embedding-3-small
CHUNK_MAX_TOKENS = 400        # a document at or under this stays one chunk
CHUNK_TARGET_TOKENS = 350     # size to aim for when a document must be split
CHUNK_OVERLAP_TOKENS = 50     # overlap between consecutive prose chunks
GLOSSARY_MIN_ROWS = 30        # a 2-column table this long is split one row (term) per chunk
EMBED_BATCH_SIZE = 100        # chunks per embeddings API call

# === MODELS ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

# === RETRIEVAL ===
RETRIEVAL_K = 5

# === PRODUCT / SCOPE ===
AMC_NAME = "PPFAS Mutual Fund"
AMC_SOURCE_URL = "https://groww.in/mutual-funds/amc/ppfas-mutual-funds"

# === ANSWER FORMAT ===
MAX_ANSWER_SENTENCES = 3
DISCLAIMER = "Facts-only. No investment advice."
