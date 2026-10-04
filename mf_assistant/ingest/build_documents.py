"""Offline: merge every text source into documents.jsonl, plus sources.csv.

documents.jsonl is the hand-off to Phase 2 (chunking + embedding): one JSON
object per line, {"id", "page_content", "metadata"}. Every document carries its
provenance in metadata (source_url, scraped_at), so any chunk cut from it can
cite its exact page.

Sources:
- scheme facts -> a few templated text docs per scheme (overview, objective,
  performance, fund managers, top holdings). SQL stays the primary path for
  exact facts; these let retrieval find them too.
- scheme FAQs, help-centre FAQs, blog sections/FAQs -> one doc each.

Run:  python -m mf_assistant.ingest.build_documents
"""

import csv
import hashlib
import json

from mf_assistant import config
from mf_assistant.ingest.manifest import load_manifest

TOP_HOLDINGS = 10


def doc(text: str, *, doc_type: str, section: str, title: str, source_url: str, scraped_at: str,
        scheme_id: str | None = None, scheme_name: str | None = None, part: int = 0, **extra) -> dict:
    # Stable id: same source + section + title (+ part, for headings repeated on one page)
    # -> same id across rebuilds, which enables incremental re-embedding later
    doc_id = hashlib.sha1(f"{source_url}|{section}|{title}|{part}".encode()).hexdigest()[:16]
    meta = {"doc_type": doc_type, "section": section, "title": title, "scheme_id": scheme_id,
            "scheme_name": scheme_name, "source_url": source_url, "scraped_at": scraped_at[:10], **extra}
    return {"id": doc_id, "page_content": text.strip(), "metadata": meta}


def _rupees(n) -> str:
    return f"₹{n:,}" if isinstance(n, (int, float)) else str(n)


def scheme_docs(f: dict) -> list[dict]:
    name = f["scheme_name"]
    common = {"doc_type": "scheme_fact", "scheme_id": f["scheme_id"], "scheme_name": name,
              "source_url": f["source_url"], "scraped_at": f["scraped_at"]}
    lock_in = f["lock_in"] or "no lock-in period"
    overview = (
        f"{name} is a {f['category']} fund ({f['sub_category']}) from {f['fund_house']}, {f['plan_type']} plan, "
        f"launched on {f['launch_date']}.\n"
        f"Riskometer: {f['riskometer']}. Benchmark: {f['benchmark']}.\n"
        f"Expense ratio: {f['expense_ratio']}. Exit load: {f['exit_load']}. Stamp duty: {f['stamp_duty']}.\n"
        f"Lock-in: {lock_in}.\n"
        f"Minimum SIP: {_rupees(f['min_sip'])}. Minimum first (lumpsum) investment: {_rupees(f['min_lumpsum'])}. "
        f"Minimum additional investment: {_rupees(f['min_additional'])}.\n"
        f"NAV: {_rupees(f['nav'])} as of {f['nav_date']}. Fund size (AUM): ₹{f['aum_cr']:,} Cr.\n"
        f"Tax implication: {f['tax_implication']}\n"
        f"Registrar & transfer agent (RTA): {f['registrar']} ({f['registrar_website']})."
    )
    r = f["returns"]
    perf_lines = [f"{label}: " + ", ".join(f"{p} {v}" for p, v in vals.items() if v) for label, vals in r["rows"].items()]
    ratios = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in f["risk_ratios"].items() if v is not None)
    analysis = "\n".join(f"- {'Pro' if a['kind'] == 'PROS' else 'Con'}: {a['text']}" for a in f["groww_analysis"])
    performance = (
        f"{name}: {r['basis']} returns and category rank as shown on Groww.\n" + "\n".join(perf_lines)
        + (f"\nRisk ratios: {ratios}." if ratios else "")
        + (f"\nGroww's analysis:\n{analysis}" if analysis else "")
    )
    managers = f"Fund managers of {name}:\n" + "\n".join(
        f"- {m['name']} (managing since {m['since']}). Education: {m['education']} Experience: {m['experience']}"
        for m in f["fund_managers"]
    )
    top = sorted(f["holdings"], key=lambda h: h["weight_pct"] or 0, reverse=True)[:TOP_HOLDINGS]
    holdings = (f"Top {len(top)} holdings of {name} (portfolio as of {f['holdings_as_of']}, "
                f"{len(f['holdings'])} holdings in total):\n"
                + "\n".join(f"- {h['name']} ({h['sector']}, {h['instrument']}): {h['weight_pct']}%" for h in top))
    return [
        doc(overview, section="overview", title=f"{name}: key facts", **common),
        doc(f"Investment objective of {name}: {f['objective']}", section="objective",
            title=f"{name}: investment objective", **common),
        doc(performance, section="performance", title=f"{name}: returns and rankings", **common),
        doc(managers, section="fund_managers", title=f"{name}: fund managers", **common),
        doc(holdings, section="holdings", title=f"{name}: top holdings", **common),
    ]


def main():
    facts = json.loads(config.SCHEME_FACTS_PATH.read_text(encoding="utf-8"))
    scheme_faqs = json.loads(config.SCHEME_FAQS_PATH.read_text(encoding="utf-8"))
    knowledge = json.loads(config.KNOWLEDGE_PATH.read_text(encoding="utf-8"))

    docs = [d for f in facts for d in scheme_docs(f)]
    docs += [
        doc(f"Question: {q['question']}\nAnswer: {q['answer']}", doc_type="scheme_faq", section="faq",
            title=q["question"], scheme_id=q["scheme_id"], scheme_name=q["scheme_name"],
            source_url=q["source_url"], scraped_at=q["scraped_at"])
        for q in scheme_faqs
    ]
    seen_titles = {}
    for k in knowledge:
        key = (k["source_url"], k["doc_type"], k["title"])
        part = seen_titles[key] = seen_titles.get(key, -1) + 1
        extra = {"topic": k["topic"], "part": part}
        if k["doc_type"] == "help_faq" or k["doc_type"] == "blog_faq":
            text = f"Question: {k['title']}\nAnswer: {k['text']}"
        else:
            text = f"{k['title']}\n{k['text']}"
        if k["doc_type"] != "help_faq":
            extra |= {"page_title": k["page_title"], "page_updated_at": k["page_updated_at"],
                      "parent_heading": k["parent_heading"]}
        docs.append(doc(text, doc_type=k["doc_type"], section=k["doc_type"], title=k["title"],
                        source_url=k["source_url"], scraped_at=k["scraped_at"], **extra))

    ids = [d["id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate document ids: two docs share source_url + section + title")

    with config.DOCUMENTS_PATH.open("w", encoding="utf-8") as out:
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_sources_csv(docs)

    by_type = {}
    for d in docs:
        t = d["metadata"]["doc_type"]
        by_type.setdefault(t, []).append(len(d["page_content"]))
    print(f"✓ {len(docs)} documents → {config.DOCUMENTS_PATH.name}")
    for t, sizes in by_type.items():
        print(f"  {t:13} {len(sizes):4} docs  chars: min {min(sizes):5}  avg {sum(sizes) // len(sizes):5}  max {max(sizes):6}")


def write_sources_csv(docs: list[dict]):
    """One row per source URL that contributed at least one document."""
    manifest = load_manifest()
    rows = {}
    for d in docs:
        m = d["metadata"]
        url = m["source_url"]
        if url not in rows:
            source_type = {"scheme_fact": "scheme_page", "scheme_faq": "scheme_page",
                           "blog_section": "blog", "blog_faq": "blog"}.get(m["doc_type"], m["doc_type"])
            title = m.get("scheme_name") or m.get("page_title") or m["title"]
            rows[url] = {"url": url, "title": title, "source_type": source_type,
                         "crawl_job": manifest[url]["job"], "scraped_at": m["scraped_at"], "documents": 0}
        rows[url]["documents"] += 1
    with config.SOURCES_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "source_type", "crawl_job", "scraped_at", "documents"])
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: (r["source_type"], r["url"])))
    print(f"✓ {len(rows)} sources → {config.SOURCES_CSV_PATH.name}")


if __name__ == "__main__":
    main()
