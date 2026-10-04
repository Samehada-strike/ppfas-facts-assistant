"""Offline: strip URLs from crawled Markdown, keeping link text and image alt text.

Moved from archive/legacy_pipeline/1-clean_markdown_files.py; paths now come from config.

Run:  python -m mf_assistant.ingest.clean
"""

import re

from mf_assistant import config


def remove_urls_from_markdown(content: str) -> str:
    """Removes URLs from markdown content while preserving link text and image alt text."""
    # [text](url) -> text
    content = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', content)
    # ![alt](url) -> alt
    content = re.sub(r'!\[([^\]]*)\]\(https?://[^\)]+\)', r'\1', content)
    # Remaining raw URLs
    content = re.sub(r'https?://[^\s\)]+', '', content)
    # Whitespace cleanup
    content = re.sub(r'  +', ' ', content)
    content = re.sub(r'\n\n\n+', '\n\n', content)
    return content


def main():
    md_files = list(config.RAW_MD_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files in {config.RAW_MD_DIR}")

    config.CLEAN_MD_DIR.mkdir(parents=True, exist_ok=True)
    for md_file in md_files:
        cleaned = remove_urls_from_markdown(md_file.read_text(encoding="utf-8"))
        (config.CLEAN_MD_DIR / md_file.name).write_text(cleaned, encoding="utf-8")
        print(f"✓ {md_file.name}")
    print(f"Cleaned {len(md_files)} files → {config.CLEAN_MD_DIR}")


if __name__ == "__main__":
    main()
