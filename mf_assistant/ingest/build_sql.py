"""Offline: load extracted scheme facts into SQLite for exact lookups.

Replaces archive/legacy_pipeline/3-build_structured_sql_db.py. Performance data
(returns, rankings, risk ratios, Groww's analysis) is stored as facts too.

Run:  python -m mf_assistant.ingest.build_sql
"""

import json
import sqlite3

from mf_assistant import config

SCHEMA = """
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS risk_ratios;
DROP TABLE IF EXISTS groww_analysis;
DROP TABLE IF EXISTS holdings;
DROP TABLE IF EXISTS fund_managers;
DROP TABLE IF EXISTS schemes;

CREATE TABLE schemes (
    scheme_id         TEXT PRIMARY KEY,
    scheme_name       TEXT NOT NULL,
    fund_house        TEXT,
    category          TEXT,
    sub_category      TEXT,
    plan_type         TEXT,
    riskometer        TEXT,
    benchmark         TEXT,
    launch_date       TEXT,
    expense_ratio     TEXT,
    exit_load         TEXT,
    stamp_duty        TEXT,
    lock_in           TEXT,      -- NULL = no lock-in
    min_sip           INTEGER,
    min_lumpsum       INTEGER,
    min_additional    INTEGER,
    nav               REAL,
    nav_date          TEXT,
    aum_cr            REAL,
    tax_implication   TEXT,
    objective         TEXT,
    registrar         TEXT,
    registrar_website TEXT,
    holdings_as_of    TEXT,
    source_url        TEXT NOT NULL,  -- citation for every fact in this row
    scraped_at        TEXT NOT NULL   -- "Last updated from sources" date
);

CREATE TABLE fund_managers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id  TEXT NOT NULL REFERENCES schemes (scheme_id),
    name       TEXT,
    since      TEXT,
    education  TEXT,
    experience TEXT
);

CREATE TABLE holdings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id  TEXT NOT NULL REFERENCES schemes (scheme_id),
    name       TEXT,
    sector     TEXT,
    instrument TEXT,
    weight_pct REAL
);

-- Long format: one row per (scheme, metric, period), e.g. ('…elss…', 'Fund returns', '3Y', '+7.7%')
CREATE TABLE returns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id TEXT NOT NULL REFERENCES schemes (scheme_id),
    basis     TEXT,   -- 'annualised'
    metric    TEXT,   -- 'Fund returns' | 'Category average (…)' | 'Rank (…)'
    period    TEXT,   -- '1Y' | '3Y' | '5Y' | 'All'
    value     TEXT    -- as displayed on the page; NULL when the page shows '--'
);

-- From the page's embedded data; not shown in its visible text
CREATE TABLE risk_ratios (
    scheme_id          TEXT PRIMARY KEY REFERENCES schemes (scheme_id),
    sharpe_ratio       REAL,
    sortino_ratio      REAL,
    alpha              REAL,
    beta               REAL,
    standard_deviation REAL,
    information_ratio  REAL
);

CREATE TABLE groww_analysis (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id TEXT NOT NULL REFERENCES schemes (scheme_id),
    kind      TEXT,   -- 'PROS' | 'CONS'
    text      TEXT
);
"""

RATIO_COLUMNS = ["sharpe_ratio", "sortino_ratio", "alpha", "beta", "standard_deviation", "information_ratio"]

SCHEME_COLUMNS = [
    "scheme_id", "scheme_name", "fund_house", "category", "sub_category", "plan_type",
    "riskometer", "benchmark", "launch_date", "expense_ratio", "exit_load", "stamp_duty",
    "lock_in", "min_sip", "min_lumpsum", "min_additional", "nav", "nav_date", "aum_cr",
    "tax_implication", "objective", "registrar", "registrar_website", "holdings_as_of",
    "source_url", "scraped_at",
]


def main():
    facts = json.loads(config.SCHEME_FACTS_PATH.read_text(encoding="utf-8"))
    config.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(config.SQLITE_PATH)
    try:
        conn.executescript(SCHEMA)
        placeholders = ", ".join("?" for _ in SCHEME_COLUMNS)
        for f in facts:
            conn.execute(
                f"INSERT INTO schemes ({', '.join(SCHEME_COLUMNS)}) VALUES ({placeholders})",
                [f.get(col) for col in SCHEME_COLUMNS],
            )
            conn.executemany(
                "INSERT INTO fund_managers (scheme_id, name, since, education, experience) VALUES (?, ?, ?, ?, ?)",
                [(f["scheme_id"], m["name"], m["since"], m["education"], m["experience"]) for m in f["fund_managers"]],
            )
            conn.executemany(
                "INSERT INTO holdings (scheme_id, name, sector, instrument, weight_pct) VALUES (?, ?, ?, ?, ?)",
                [(f["scheme_id"], h["name"], h["sector"], h["instrument"], h["weight_pct"]) for h in f["holdings"]],
            )
            r = f["returns"]
            conn.executemany(
                "INSERT INTO returns (scheme_id, basis, metric, period, value) VALUES (?, ?, ?, ?, ?)",
                [(f["scheme_id"], r["basis"], metric, period, value)
                 for metric, values in r["rows"].items() for period, value in values.items()],
            )
            conn.execute(
                f"INSERT INTO risk_ratios (scheme_id, {', '.join(RATIO_COLUMNS)}) VALUES (?, {', '.join('?' for _ in RATIO_COLUMNS)})",
                [f["scheme_id"]] + [f["risk_ratios"].get(c) for c in RATIO_COLUMNS],
            )
            conn.executemany(
                "INSERT INTO groww_analysis (scheme_id, kind, text) VALUES (?, ?, ?)",
                [(f["scheme_id"], a["kind"], a["text"]) for a in f["groww_analysis"]],
            )
        conn.commit()
        for table in ("schemes", "fund_managers", "holdings", "returns", "risk_ratios", "groww_analysis"):
            print(f"✓ {table}: {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]} rows")
    finally:
        conn.close()
    print(f"SQLite DB → {config.SQLITE_PATH}")


if __name__ == "__main__":
    main()
