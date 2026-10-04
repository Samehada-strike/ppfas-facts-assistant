"""Offline job `knowledge_pages`: crawl the frozen help-FAQ + blog URL list.

Reads knowledge_urls.json (made by discover_knowledge.py) and saves each page's
raw HTML and Markdown. This job runs on its own schedule, separately from
`scheme_pages`, because this content changes rarely.

Politeness: robots.txt is checked for every URL, at most 2 pages are fetched at
once, and each request waits 1-2 s; HTTP 429/503 responses back off and retry.

Groww intermittently serves its error screen ("Some Error Occured") with HTTP
200. Those pages are retried in slower passes, one page at a time.

Run:  python -m mf_assistant.ingest.crawl_knowledge
"""

import asyncio
import json
import sys

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, RateLimiter, SemaphoreDispatcher

from mf_assistant import config
from mf_assistant.ingest.manifest import load_manifest, record_fetch, save_manifest

JOB = "knowledge_pages"
ERROR_MARKER = "Some Error Occured"  # Groww's error screen (their spelling)
RETRY_PASSES = 2


def raw_name(url: str) -> str:
    """'https://groww.in/help/mutual-funds/order/what-is-exit-load--71' -> 'help__order__what-is-exit-load--71'."""
    path = url.split("groww.in/", 1)[1]
    return path.replace("help/mutual-funds/", "help/").replace("/", "__")


def dispatcher(concurrency: int, delay: tuple[float, float]) -> SemaphoreDispatcher:
    # A fixed concurrency cap. (MemoryAdaptiveDispatcher pauses while system RAM is
    # above 90%, which stalls the crawl on a busy machine.)
    return SemaphoreDispatcher(
        semaphore_count=concurrency,
        rate_limiter=RateLimiter(base_delay=delay, max_delay=30.0, max_retries=3),
    )


async def crawl_pass(crawler, urls, manifest, disp, *, final: bool) -> list[str]:
    """Crawl `urls` once and save good pages. Returns URLs that got the error screen or failed.

    On the final pass, error-screen pages are saved anyway so extraction can report them.
    """
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # always fetch fresh; our raw files are the cache
        check_robots_txt=True,
    )
    retry = []
    for result in await crawler.arun_many(urls, config=run_config, dispatcher=disp):
        name = raw_name(result.url)
        fetched = result.success and result.status_code == 200
        error_screen = fetched and ERROR_MARKER in (result.markdown or "")
        if fetched and (not error_screen or final):
            (config.RAW_KNOWLEDGE_DIR / f"{name}.html").write_text(result.html, encoding="utf-8")
            (config.RAW_KNOWLEDGE_DIR / f"{name}.md").write_text(result.markdown, encoding="utf-8")
            record_fetch(manifest, url=result.url, job=JOB, status=200, raw_file=f"raw/knowledge/{name}.html")
        elif final:
            record_fetch(manifest, url=result.url, job=JOB, status=result.status_code or 0, raw_file=None)
        if not fetched or error_screen:
            retry.append(result.url)
    return retry


async def main():
    entries = json.loads(config.KNOWLEDGE_URLS_PATH.read_text(encoding="utf-8"))["urls"]
    urls = [e["url"] for e in entries]
    config.RAW_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    print(f"Crawling {len(urls)} knowledge pages (job={JOB})")
    async with AsyncWebCrawler() as crawler:
        pending = await crawl_pass(crawler, urls, manifest, dispatcher(2, (1.0, 2.0)), final=False)
        print(f"Pass 1: {len(urls) - len(pending)} good, {len(pending)} error screen or failed")
        for attempt in range(1, RETRY_PASSES + 1):
            if not pending:
                break
            await asyncio.sleep(10)  # give the server a moment before retrying
            pending = await crawl_pass(crawler, pending, manifest, dispatcher(1, (3.0, 5.0)),
                                       final=attempt == RETRY_PASSES)
            print(f"Retry pass {attempt}: {len(pending)} still failing")

    save_manifest(manifest)
    print(f"\n✓ {len(urls) - len(pending)} good pages, ✗ {len(pending)} still failing → {config.RAW_KNOWLEDGE_DIR}")
    for url in pending:
        print(f"  ✗ {url}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
