"""Offline job `scheme_pages`: crawl each Groww scheme page and save HTML + Markdown.

Moved from archive/legacy_pipeline/0-crawl_parag_parikh.py. Changes: paths come from
config, and every fetch is recorded in the crawl manifest.

Run:  python -m mf_assistant.ingest.crawl_schemes
"""

import asyncio
import json
import sys

from crawl4ai import AsyncWebCrawler

from mf_assistant import config
from mf_assistant.ingest.manifest import load_manifest, record_fetch, save_manifest

JOB = "scheme_pages"
DELAY_SECONDS = 2  # be polite: one page at a time, with a pause between pages


async def crawl_scheme(crawler: AsyncWebCrawler, scheme: dict, manifest: dict):
    scheme_id = scheme["scheme_id"]
    url = scheme["source_url"]
    print(f"\n=== Crawling scheme: {scheme_id} ===\nURL: {url}")

    html_file = config.RAW_HTML_DIR / f"{scheme_id}.html"
    md_file = config.RAW_MD_DIR / f"{scheme_id}.md"

    try:
        result = await crawler.arun(url=url)
        if not result.success:
            raise RuntimeError(result.error_message)

        html_file.write_text(result.html, encoding="utf-8")
        md_file.write_text(result.markdown, encoding="utf-8")
        record_fetch(manifest, url=url, job=JOB, status=result.status_code or 200,
                     raw_file=str(html_file.relative_to(config.DATA_DIR)))

        print(f"✓ HTML {len(result.html):,} chars, Markdown {len(result.markdown):,} chars")
    except Exception as e:
        # Keep the previous raw files; record the failure so it's visible in the manifest
        record_fetch(manifest, url=url, job=JOB, status=0, raw_file=None)
        print(f"✗ Error crawling {scheme_id}: {e}")


async def main():
    schemes = json.loads(config.SCHEMES_CONFIG_PATH.read_text(encoding="utf-8"))
    config.RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_MD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    print(f"Crawling {len(schemes)} scheme pages (job={JOB})")
    async with AsyncWebCrawler() as crawler:
        for i, scheme in enumerate(schemes):
            if i:
                await asyncio.sleep(DELAY_SECONDS)
            await crawl_scheme(crawler, scheme, manifest)

    save_manifest(manifest)
    print(f"\nManifest saved: {config.CRAWL_MANIFEST_PATH}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        # crawl4ai drives a browser subprocess; Windows needs the Proactor loop for that
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
