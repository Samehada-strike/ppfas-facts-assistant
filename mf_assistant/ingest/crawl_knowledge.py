"""Offline job `knowledge_pages`: crawl the frozen help-FAQ + blog URL list.

Reads knowledge_urls.json (made by discover_knowledge.py) and saves each page's
raw HTML and Markdown. This job runs on its own schedule, separately from
`scheme_pages`, because this content changes rarely.

Politeness: robots.txt is checked for every URL, at most 2 pages are fetched at
once, and each request waits 1-2 s; HTTP 429/503 responses back off and retry.

Run:  python -m mf_assistant.ingest.crawl_knowledge
"""

import asyncio
import json
import sys

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, RateLimiter, SemaphoreDispatcher

from mf_assistant import config
from mf_assistant.ingest.manifest import load_manifest, record_fetch, save_manifest

JOB = "knowledge_pages"


def raw_name(url: str) -> str:
    """'https://groww.in/help/mutual-funds/order/what-is-exit-load--71' -> 'help__order__what-is-exit-load--71'."""
    path = url.split("groww.in/", 1)[1]
    return path.replace("help/mutual-funds/", "help/").replace("/", "__")


async def main():
    entries = json.loads(config.KNOWLEDGE_URLS_PATH.read_text(encoding="utf-8"))["urls"]
    urls = [e["url"] for e in entries]
    config.RAW_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # always fetch fresh; our raw files are the cache
        check_robots_txt=True,
    )
    # A fixed cap of 2 pages at a time. (MemoryAdaptiveDispatcher pauses while system RAM
    # is above 90%, which stalls the crawl on a busy machine.)
    dispatcher = SemaphoreDispatcher(
        semaphore_count=2,
        rate_limiter=RateLimiter(base_delay=(1.0, 2.0), max_delay=30.0, max_retries=3),
    )

    ok, failed = 0, []
    print(f"Crawling {len(urls)} knowledge pages (job={JOB})")
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(urls, config=run_config, dispatcher=dispatcher)
        for result in results:
            name = raw_name(result.url)
            if result.success and result.status_code == 200:
                (config.RAW_KNOWLEDGE_DIR / f"{name}.html").write_text(result.html, encoding="utf-8")
                (config.RAW_KNOWLEDGE_DIR / f"{name}.md").write_text(result.markdown, encoding="utf-8")
                record_fetch(manifest, url=result.url, job=JOB, status=200,
                             raw_file=f"raw/knowledge/{name}.html")
                ok += 1
            else:
                record_fetch(manifest, url=result.url, job=JOB, status=result.status_code or 0, raw_file=None)
                failed.append((result.url, result.status_code, result.error_message))

    save_manifest(manifest)
    print(f"\n✓ {ok} saved, ✗ {len(failed)} failed → {config.RAW_KNOWLEDGE_DIR}")
    for url, status, err in failed:
        print(f"  ✗ {status} {url} {err or ''}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
