"""Offline: turn crawled scheme pages into clean facts + FAQs, with provenance.

Replaces project_files/2-data_extraction.ipynb, whose regexes broke when Groww
redesigned its pages. Two sources per scheme:

- Facts come from the JSON that Groww embeds in every page for its own frontend
  (<script id="__NEXT_DATA__">). Typed values, no regex on visible text.
- FAQs and the returns table come from the cleaned Markdown, split on headings
  (any level). Returns are read from the visible table because the embedded
  JSON's figures differ for some periods, and citations point users at the page.

Performance data (returns, rankings, risk ratios, Groww's pros/cons analysis)
is kept as facts. Whether and how answers may quote it is decided at answer
time, not here.

Every run is validated; a missing required field fails loudly instead of
silently writing None.

Run:  python -m mf_assistant.ingest.extract_scheme
"""

import json
import re
from datetime import datetime, timedelta, timezone

from mf_assistant import config
from mf_assistant.ingest.manifest import load_manifest

IST = timezone(timedelta(hours=5, minutes=30))

# A scheme record is unusable for the assistant without these
REQUIRED_FIELDS = [
    "scheme_name", "category", "sub_category", "riskometer", "benchmark",
    "expense_ratio", "exit_load", "min_sip", "min_lumpsum", "nav", "nav_date",
]

# Max gap (percentage points) between page-table and embedded-JSON fund returns
RETURNS_TOLERANCE = 0.1


# ---------- embedded JSON (facts) ----------

def load_page_data(html: str) -> dict:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("__NEXT_DATA__ script not found (page layout changed?)")
    return json.loads(m.group(1))["props"]["pageProps"]["mfServerSideData"]


def _date(value: str | None) -> str | None:
    """'2019-07-03T18:30:00.000Z' -> '2019-07-04' (Groww stores midnight IST as UTC)."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(IST).date().isoformat()


def _lock_in(lock: dict | None) -> str | None:
    lock = lock or {}
    parts = [f"{lock[k]} {k}" for k in ("years", "months", "days") if lock.get(k)]
    return " ".join(parts) or None


def _strip_strings(o):
    """Recursively trim whitespace in every string value (Groww data has stray spaces)."""
    if isinstance(o, str):
        return o.strip()
    if isinstance(o, list):
        return [_strip_strings(v) for v in o]
    if isinstance(o, dict):
        return {k: _strip_strings(v) for k, v in o.items()}
    return o


def extract_facts(d: dict) -> dict:
    d = _strip_strings(d)
    risk_stats = (d.get("return_stats") or [{}])[0]
    risk = risk_stats.get("risk")  # current riskometer; `nfo_risk` is the stale launch-time value
    return {
        "scheme_name": d.get("scheme_name"),
        "fund_house": d.get("fund_house"),
        "category": d.get("category"),
        "sub_category": d.get("sub_category"),
        "plan_type": d.get("plan_type"),
        "riskometer": f"{risk} Risk" if risk else None,
        "benchmark": d.get("benchmark_name"),
        "launch_date": d.get("launch_date"),
        "expense_ratio": f"{d['expense_ratio']}%" if d.get("expense_ratio") else None,
        "exit_load": d.get("exit_load"),
        "stamp_duty": d.get("stamp_duty"),
        "lock_in": _lock_in(d.get("lock_in")),  # None = no lock-in
        "min_sip": d.get("min_sip_investment"),
        "min_lumpsum": d.get("min_investment_amount"),
        "min_additional": d.get("mini_additional_investment"),
        "nav": d.get("nav"),
        "nav_date": d.get("nav_date"),
        "aum_cr": round(d["aum"], 2) if d.get("aum") else None,
        "tax_implication": (d.get("category_info") or {}).get("tax_impact"),
        "objective": d.get("description"),
        "registrar": (d.get("rta_details") or {}).get("rta_name"),
        "registrar_website": (d.get("rta_details") or {}).get("website"),
        "fund_managers": [
            {
                "name": m.get("person_name"),
                "since": _date(m.get("date_from")),
                "education": m.get("education"),
                "experience": m.get("experience"),
            }
            for m in d.get("fund_manager_details") or []
        ],
        "holdings": [
            {
                "name": h.get("company_name"),
                "sector": h.get("sector_name"),
                "instrument": h.get("instrument_name"),
                "weight_pct": round(h["corpus_per"], 2) if h.get("corpus_per") is not None else None,
            }
            for h in d.get("holdings") or []
        ],
        "holdings_as_of": _date((d.get("holdings") or [{}])[0].get("portfolio_date")),
        # Embedded-only: these ratios are not in the page's visible text
        "risk_ratios": {
            k: risk_stats.get(k)
            for k in ("sharpe_ratio", "sortino_ratio", "alpha", "beta", "standard_deviation", "information_ratio")
        },
        "groww_analysis": [
            {"kind": a.get("analysis_type"), "text": a.get("analysis_desc")}
            for a in d.get("analysis") or []
        ],
        # Only used to cross-check the page's returns table; not written out
        "_json_fund_returns": next(
            ({"1Y": s.get("stat_1y"), "3Y": s.get("stat_3y"), "5Y": s.get("stat_5y")}
             for s in d.get("stats") or [] if s.get("type") == "FUND_RETURN"),
            {},
        ),
    }


# ---------- Markdown (FAQs) ----------

def split_sections(md: str) -> list[tuple[int, str, str]]:
    """Split Markdown into (level, heading, body) at every heading, regardless of level."""
    sections, current = [], None
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s*(.*?)\s*$", line)
        if m:
            if current:
                sections.append(current)
            current = (len(m.group(1)), m.group(2), [])
        elif current:
            current[2].append(line)
    if current:
        sections.append(current)
    return [(lvl, head, "\n".join(body).strip()) for lvl, head, body in sections]


def extract_faqs(md: str) -> list[dict]:
    """Questions are the headings after the 'FAQs' heading; answers are their bodies."""
    sections = split_sections(md)
    start = next((i for i, (_, h, _) in enumerate(sections) if h.lower() == "faqs"), None)
    if start is None:
        return []
    faqs = []
    for _, question, answer in sections[start + 1:]:
        if not question.endswith("?"):
            break  # first non-question heading ends the FAQ block
        answer = answer.split("\n!Looking to invest")[0]  # trailing promo banner
        answer = re.sub(r"\s*\n\s*", "\n", answer).strip()
        if answer:
            faqs.append({"question": question, "answer": answer})
    return faqs


def extract_returns(md: str) -> dict | None:
    """Parse the 'Returns and rankings' table (annualised view, the page default).

    Returns {"basis": "annualised", "periods": [...], "rows": {label: {period: value}}}.
    """
    section = next((body for _, h, body in split_sections(md) if "returns and rankings" in h.lower()), None)
    if not section:
        return None
    table = [line for line in section.splitlines() if "|" in line]
    if len(table) < 3:
        return None
    periods = [c.strip() for c in table[0].split("|")][1:]
    rows = {}
    for line in table[2:]:  # skip the header and the ---|--- separator
        cells = [c.strip() for c in line.split("|")]
        rows[cells[0]] = {p: (v if v not in ("--", "") else None) for p, v in zip(periods, cells[1:]) if p}
    return {"basis": "annualised", "periods": [p for p in periods if p], "rows": rows}


def _pct(value: str | None) -> float | None:
    return float(value.replace("%", "").replace("+", "")) if value else None


# ---------- pipeline ----------

def validate(record: dict) -> list[str]:
    problems = [f"missing {f}" for f in REQUIRED_FIELDS if record["facts"].get(f) in (None, "")]
    if not record["facts"]["fund_managers"]:
        problems.append("no fund managers")
    if not record["faqs"]:
        problems.append("no FAQs")
    returns = record["facts"].get("returns")
    if not returns or "Fund returns" not in returns["rows"]:
        problems.append("no returns table")
    else:
        # Cross-check: the visible table and the embedded JSON must agree for 1Y/3Y/5Y
        page = returns["rows"]["Fund returns"]
        for period, json_value in record["facts"]["_json_fund_returns"].items():
            page_value = _pct(page.get(period))
            if None not in (page_value, json_value) and abs(page_value - json_value) > RETURNS_TOLERANCE:
                problems.append(f"{period} fund return mismatch: page {page_value} vs JSON {json_value}")
    return problems


def main():
    schemes = json.loads(config.SCHEMES_CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest()

    records, all_problems = [], {}
    for scheme in schemes:
        scheme_id, url = scheme["scheme_id"], scheme["source_url"]
        html = (config.RAW_HTML_DIR / f"{scheme_id}.html").read_text(encoding="utf-8")
        md = (config.CLEAN_MD_DIR / f"{scheme_id}.md").read_text(encoding="utf-8")

        facts = extract_facts(load_page_data(html))
        facts["returns"] = extract_returns(md)
        record = {
            "scheme_id": scheme_id,
            "source_url": url,
            "scraped_at": (manifest.get(url) or {}).get("scraped_at"),
            "facts": facts,
            "faqs": extract_faqs(md),
        }
        problems = validate(record)
        if not record["scraped_at"]:
            problems.append("URL not in crawl manifest")
        if problems:
            all_problems[scheme_id] = problems
        records.append(record)
        print(f"{'✗' if problems else '✓'} {record['facts']['scheme_name']}: "
              f"{len(record['facts']['holdings'])} holdings, "
              f"{len(record['facts']['fund_managers'])} managers, "
              f"{len(record['faqs'])} FAQs, returns table {'✓' if facts['returns'] else '✗'}")

    if all_problems:
        # Fail before writing anything: a broken run must not replace good data
        raise SystemExit(f"Validation failed, nothing written:\n{json.dumps(all_problems, indent=2)}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    facts = [
        {k: r[k] for k in ("scheme_id", "source_url", "scraped_at")}
        | {k: v for k, v in r["facts"].items() if not k.startswith("_")}
        for r in records
    ]
    faqs = [
        {"scheme_id": r["scheme_id"], "scheme_name": r["facts"]["scheme_name"],
         "source_url": r["source_url"], "scraped_at": r["scraped_at"], **f}
        for r in records for f in r["faqs"]
    ]
    config.SCHEME_FACTS_PATH.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    config.SCHEME_FAQS_PATH.write_text(json.dumps(faqs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(facts)} schemes → {config.SCHEME_FACTS_PATH.name}, "
          f"{len(faqs)} FAQs → {config.SCHEME_FAQS_PATH.name}")


if __name__ == "__main__":
    main()
