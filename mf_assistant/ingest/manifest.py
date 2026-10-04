"""Crawl manifest: one record per crawled URL, saying when and how it was fetched.

This file is the single source of truth for provenance dates. Documents built
later inherit `scraped_at` from here, and sources.csv is generated from it.
"""

import json
from datetime import datetime

from mf_assistant import config


def load_manifest() -> dict[str, dict]:
    """Return {url: record}. Empty if no crawl has run yet."""
    if not config.CRAWL_MANIFEST_PATH.exists():
        return {}
    records = json.loads(config.CRAWL_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {r["url"]: r for r in records}


def record_fetch(manifest: dict[str, dict], *, url: str, job: str, status: int, raw_file: str | None):
    """Add or replace the record for one URL (a re-crawl overwrites the old entry)."""
    manifest[url] = {
        "url": url,
        "job": job,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "raw_file": raw_file,
    }


def save_manifest(manifest: dict[str, dict]):
    config.CRAWL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(manifest.values(), key=lambda r: (r["job"], r["url"]))
    config.CRAWL_MANIFEST_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
