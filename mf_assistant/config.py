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

# Offline outputs (index), loaded by the online app
FAISS_INDEX_DIR = DATA_DIR / "index" / "faiss"

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
