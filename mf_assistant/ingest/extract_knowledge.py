"""Offline: turn crawled help articles and blogs into clean, structured text.

- Help article: the question is the page's last H1; the answer is the text
  after it, up to Groww's "Was the answer helpful?" widget.
- Blog: the article is the Markdown from its H1 up to the disclaimer, split
  into heading sections. The page's embedded data gives the post's own
  `updated_at` date.

Extraction keeps structure (one record per article / section); sizing chunks
is Phase 2's job.

Unlike scheme pages, one malformed article shouldn't block the corpus: bad
pages are skipped and reported, and the run fails only if too many are bad.

Run:  python -m mf_assistant.ingest.extract_knowledge
"""

import json
import re

from mf_assistant import config
from mf_assistant.ingest.clean import remove_urls_from_markdown
from mf_assistant.ingest.crawl_knowledge import ERROR_MARKER, raw_name
from mf_assistant.ingest.extract_scheme import split_sections
from mf_assistant.ingest.manifest import load_manifest

MAX_FAILURE_RATE = 0.10


def clean_text(md: str) -> str:
    md = remove_urls_from_markdown(md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images with relative/empty URLs
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)  # remaining links -> text
    md = re.sub(r"^!.*$", "", md, flags=re.M)  # image alt-text leftovers ("!What are Mutual Funds?")
    md = re.sub(r"^\d+ min read$", "", md, flags=re.M)
    md = re.sub(r"^(Loading\.\.\.|```)\s*$", "", md, flags=re.M)  # page-loading placeholder, stray code fences
    md = re.sub(r"[ \t]*\n[ \t]*", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


class DeadPage(Exception):
    """The site serves no article at this URL (empty shell or error screen): a source problem, not ours."""


def extract_help_article(md: str) -> dict | None:
    """Return {"question", "answer"}; None if the article exists but couldn't be parsed."""
    h1s = list(re.finditer(r"^# (.+)$", md, re.M))
    # A real article page has two H1s: the "Customer Support" banner, then the question.
    # Broken articles render only the banner, or Groww's error screen even after retries.
    if len(h1s) < 2 or ERROR_MARKER in md:
        raise DeadPage
    question = h1s[-1].group(1).strip()
    rest = md[h1s[-1].end():]
    if "Was the answer helpful?" not in rest:
        return None  # our end-of-answer marker is gone: the page layout changed (our problem)
    answer = clean_text(rest.split("Was the answer helpful?")[0])
    if not answer:
        raise DeadPage  # question published with a blank answer (Groww's problem)
    return {"question": question, "answer": answer}


def blog_updated_at(html: str) -> str | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    blog = json.loads(m.group(1))["props"]["pageProps"].get("blogData") or {}
    return (blog.get("updated_at") or "")[:10] or None


def extract_blog(md: str) -> dict | None:
    """Return {"title", "sections": [{"heading", "parent", "text", "is_faq"}]}."""
    h1 = re.search(r"^# (.+)$", md, re.M)
    if not h1:
        return None
    article = md[h1.start():]
    # The article ends at whichever comes first: a Disclaimer heading, the feedback
    # prompt, or the "Recent Posts" sidebar (everything after is site navigation)
    article = re.split(r"^#{1,6} Disclaimer\s*$|^Do you like this edition\?|^Recent Posts\s*$",
                       article, flags=re.M)[0]

    sections, parent, in_faqs = [], None, False
    for level, heading, body in split_sections(article):
        heading = clean_text(heading).replace("**", "").strip()
        if level <= 2:
            parent = heading
            in_faqs = heading.lower() in ("faqs", "frequently asked questions")
        text = clean_text(body)
        if text:
            sections.append({"heading": heading, "parent": parent if level > 2 else None,
                             "text": text, "is_faq": in_faqs and level > 2})
    title = clean_text(h1.group(1))
    return {"title": title, "sections": sections} if sections else None


def main():
    entries = json.loads(config.KNOWLEDGE_URLS_PATH.read_text(encoding="utf-8"))["urls"]
    manifest = load_manifest()

    records, skipped, seen = [], [], set()
    for e in entries:
        url, kind = e["url"], e["kind"]
        md_path = config.RAW_KNOWLEDGE_DIR / f"{raw_name(url)}.md"
        if not md_path.exists():
            skipped.append((url, "not crawled"))
            continue
        md = md_path.read_text(encoding="utf-8")
        base = {"source_url": url, "topic": e["topic"], "scraped_at": manifest[url]["scraped_at"]}

        if kind == "help_faq":
            try:
                art = extract_help_article(md)
            except DeadPage:
                skipped.append((url, "dead page"))
                continue
            if not art:
                skipped.append((url, "no question/answer found"))
                continue
            key = (art["question"].lower(), art["answer"][:200].lower())
            if key in seen:  # the same article is listed under several topics
                skipped.append((url, "duplicate"))
                continue
            seen.add(key)
            records.append({"doc_type": "help_faq", **base, "title": art["question"], "text": art["answer"]})
        else:
            blog = extract_blog(md)
            if not blog:
                skipped.append((url, "no article found"))
                continue
            html = (config.RAW_KNOWLEDGE_DIR / f"{raw_name(url)}.html").read_text(encoding="utf-8")
            updated = blog_updated_at(html)
            for s in blog["sections"]:
                records.append({
                    "doc_type": "blog_faq" if s["is_faq"] else "blog_section", **base,
                    "page_title": blog["title"], "page_updated_at": updated,
                    "title": s["heading"], "parent_heading": s["parent"], "text": s["text"],
                })

    # Dead pages and duplicates are facts about the source; only parse failures count against us
    failures = [s for s in skipped if s[1] not in ("duplicate", "dead page")]
    for url, reason in skipped:
        if reason != "dead page":
            print(f"  skip ({reason}): {url}")
    dead = sum(1 for _, reason in skipped if reason == "dead page")
    rate = len(failures) / len(entries)
    if rate > MAX_FAILURE_RATE:
        raise SystemExit(f"{len(failures)}/{len(entries)} pages failed ({rate:.0%}); nothing written")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.KNOWLEDGE_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for r in records:
        counts[r["doc_type"]] = counts.get(r["doc_type"], 0) + 1
    print(f"✓ {len(records)} records {counts} → {config.KNOWLEDGE_PATH.name} "
          f"({len(failures)} failed, {dead} dead pages, {len(skipped) - len(failures) - dead} duplicates)")


if __name__ == "__main__":
    main()
