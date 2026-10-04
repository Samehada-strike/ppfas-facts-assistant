"""Offline: build the frozen URL list for the `knowledge_pages` crawl job.

Reads Groww's support sitemap (listed in robots.txt), keeps the mutual-fund
help articles, and adds the hand-picked blog posts. The list is written once
and reviewed; the crawl then reads this file instead of the live sitemap, so
the corpus only changes when we decide to re-discover.

Run:  python -m mf_assistant.ingest.discover_knowledge
"""

import json
import re
import urllib.request
from datetime import datetime

from mf_assistant import config


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    req = urllib.request.Request(sitemap_url, headers={"User-Agent": "Mozilla/5.0"})
    xml = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def main():
    sitemap_urls = fetch_sitemap_urls(config.SUPPORT_SITEMAP_URL)
    help_urls = sorted({u for u in sitemap_urls if u.startswith(config.KNOWLEDGE_URL_PREFIX)})

    entries = (
        [{"url": u, "kind": "help_faq", "topic": u.removeprefix(config.KNOWLEDGE_URL_PREFIX).split("/")[0]}
         for u in help_urls]
        + [{"url": u, "kind": "blog", "topic": "blog"} for u in config.EXTRA_KNOWLEDGE_URLS]
    )
    payload = {
        "discovered_at": datetime.now().isoformat(timespec="seconds"),
        "sitemap": config.SUPPORT_SITEMAP_URL,
        "urls": entries,
    }
    config.KNOWLEDGE_URLS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Sitemap: {len(sitemap_urls)} URLs, {len(help_urls)} under {config.KNOWLEDGE_URL_PREFIX}")
    topics = {}
    for e in entries:
        topics[e["topic"]] = topics.get(e["topic"], 0) + 1
    for topic, n in sorted(topics.items(), key=lambda t: -t[1]):
        print(f"  {n:4}  {topic}")
    print(f"Wrote {len(entries)} URLs → {config.KNOWLEDGE_URLS_PATH}")


if __name__ == "__main__":
    main()
