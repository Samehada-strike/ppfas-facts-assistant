import asyncio
import json
import sys
from pathlib import Path

from crawl4ai import AsyncWebCrawler


# === CONFIGURATION ===

# Project root (folder that contains schemes_config.json)
BASE_DIR = Path(r"E:\ML Projects\RAG_Chatbot_PP_MF")

# Paths
CONFIG_PATH = BASE_DIR / "project_files"/ "data"/ "json_files"/ "schemes_config.json"
HTML_DIR = BASE_DIR / "project_files" / "data" / "raw" / "HTML"
MD_DIR = BASE_DIR / "project_files" / "data" / "raw" / "Markdown"


async def crawl_scheme(crawler: AsyncWebCrawler, scheme: dict):
    """
    Crawl a single scheme URL and save HTML/Markdown.

    HTML:     data/raw/HTML/<scheme_id>.html
    Markdown: data/raw/Markdown/<scheme_id>.md
    """
    scheme_id = scheme["scheme_id"]
    url = scheme["source_url"]

    print(f"\n=== Crawling scheme: {scheme_id} ===")
    print(f"URL: {url}")

    # Ensure output dirs exist
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    md_file = MD_DIR / f"{scheme_id}.md"
    html_file = HTML_DIR / f"{scheme_id}.html"

    try:
        result = await crawler.arun(url=url)

        # Save markdown
        md_file.write_text(result.markdown, encoding="utf-8")

        # Save HTML
        html_file.write_text(result.html, encoding="utf-8")

        print(f"✓ Markdown saved to: {md_file}")
        print(f"✓ HTML saved to:     {html_file}")
        print(f"Markdown length: {len(result.markdown):,} chars")
        print(f"HTML length:     {len(result.html):,} chars")

    except Exception as e:
        print(f"✗ Error crawling {scheme_id}: {e}")


async def main():
    """Crawl all schemes defined in schemes_config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        schemes = json.load(f)

    print("Starting crawl4ai batch for Parag Parikh schemes...")
    print(f"Using config: {CONFIG_PATH}")
    print(f"Total schemes: {len(schemes)}")
    print("-" * 60)

    async with AsyncWebCrawler() as crawler:
        for scheme in schemes:
            await crawl_scheme(crawler, scheme)

    print("\nAll schemes processed.")


if __name__ == "__main__":
    # On Windows, make sure we use an event loop that supports subprocesses
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception as e:
            print(f"Warning: could not set WindowsProactorEventLoopPolicy: {e}")

    asyncio.run(main())