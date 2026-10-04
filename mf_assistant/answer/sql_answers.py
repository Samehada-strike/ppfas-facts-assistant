"""Online: exact scheme facts from SQLite, each with its citation.

Every fact field maps to a fixed, parameterized query plus a formatter that
turns the row into one factual sentence. There is deliberately no
text-to-SQL: an LLM never writes SQL here, so a question can't produce a wrong
join, an unexpected query or an injected statement. The router only picks
*which* field to look up; this module does the lookup.

Each FactAnswer carries source_url and scraped_at from the same row, so the
citation always belongs to the value it supports.
"""

import sqlite3
from dataclasses import dataclass

from mf_assistant import config

TOP_HOLDINGS = 5


@dataclass
class FactAnswer:
    scheme_id: str
    scheme_name: str
    field: str
    sentence: str  # one factual sentence, no advice
    source_url: str
    scraped_at: str


def _rupees(n) -> str:
    return f"₹{n:,.0f}" if isinstance(n, (int, float)) else str(n)


def _scheme_row(con, scheme_id: str) -> sqlite3.Row:
    return con.execute("SELECT * FROM schemes WHERE scheme_id = ?", (scheme_id,)).fetchone()


# field -> function(con, scheme row) -> sentence
def _expense_ratio(con, s):
    return f"The expense ratio of {s['scheme_name']} is {s['expense_ratio']}."


def _exit_load(con, s):
    return f"The exit load of {s['scheme_name']} is: {s['exit_load'].rstrip('.')}."


def _min_investment(con, s):
    return (f"For {s['scheme_name']}, the minimum SIP is {_rupees(s['min_sip'])}, the minimum first (lumpsum) "
            f"investment is {_rupees(s['min_lumpsum'])}, and additional investments start at {_rupees(s['min_additional'])}.")


def _lock_in(con, s):
    if s["lock_in"]:
        return f"{s['scheme_name']} has a lock-in period of {s['lock_in']}."
    return f"{s['scheme_name']} has no lock-in period."


def _riskometer(con, s):
    return f"The riskometer rating of {s['scheme_name']} is {s['riskometer']}."


def _benchmark(con, s):
    return f"The benchmark of {s['scheme_name']} is the {s['benchmark']}."


def _nav(con, s):
    return f"The NAV of {s['scheme_name']} was ₹{s['nav']:,.2f} as of {s['nav_date']}."


def _aum(con, s):
    return f"The fund size (AUM) of {s['scheme_name']} is ₹{s['aum_cr']:,.2f} Cr."


def _category(con, s):
    article = "an" if s["category"][0].lower() in "aeiou" else "a"
    return f"{s['scheme_name']} is {article} {s['category']} fund in the {s['sub_category']} category ({s['plan_type']} plan)."


def _launch_date(con, s):
    return f"{s['scheme_name']} was launched on {s['launch_date']}."


def _objective(con, s):
    return f"Investment objective of {s['scheme_name']}: {s['objective']}"


def _tax(con, s):
    return f"Tax implication for {s['scheme_name']}: {s['tax_implication']}"


def _stamp_duty(con, s):
    return f"Stamp duty on investments in {s['scheme_name']} is {s['stamp_duty']}."


def _registrar(con, s):
    return (f"The registrar and transfer agent (RTA) for {s['scheme_name']} is {s['registrar']} "
            f"({s['registrar_website']}), which issues account statements.")


def _fund_managers(con, s):
    rows = con.execute("SELECT name, since FROM fund_managers WHERE scheme_id = ? ORDER BY since",
                       (s["scheme_id"],)).fetchall()
    names = ", ".join(f"{r['name']} (since {r['since']})" for r in rows)
    return f"{s['scheme_name']} is managed by {names}."


def _top_holdings(con, s):
    rows = con.execute("SELECT name, weight_pct FROM holdings WHERE scheme_id = ? ORDER BY weight_pct DESC LIMIT ?",
                       (s["scheme_id"], TOP_HOLDINGS)).fetchall()
    items = ", ".join(f"{r['name']} ({r['weight_pct']}%)" for r in rows)
    return f"The top {len(rows)} holdings of {s['scheme_name']} (as of {s['holdings_as_of']}) are {items}."


def _returns(con, s):
    rows = con.execute("SELECT metric, period, value FROM returns WHERE scheme_id = ? AND value IS NOT NULL",
                       (s["scheme_id"],)).fetchall()
    fund = ", ".join(f"{r['period']} {r['value']}" for r in rows if r["metric"] == "Fund returns")
    return f"Annualised returns of {s['scheme_name']} as published on Groww: {fund}. Past performance does not indicate future returns."


def _risk_ratios(con, s):
    r = con.execute("SELECT * FROM risk_ratios WHERE scheme_id = ?", (s["scheme_id"],)).fetchone()
    parts = [f"{k.replace('_', ' ')} {r[k]}" for k in
             ("sharpe_ratio", "sortino_ratio", "alpha", "beta", "standard_deviation", "information_ratio") if r[k] is not None]
    return f"Risk ratios of {s['scheme_name']} published by Groww: {', '.join(parts)}."


FACT_FIELDS = {
    "expense_ratio": _expense_ratio,
    "exit_load": _exit_load,
    "min_investment": _min_investment,
    "lock_in": _lock_in,
    "riskometer": _riskometer,
    "benchmark": _benchmark,
    "nav": _nav,
    "aum": _aum,
    "category": _category,
    "launch_date": _launch_date,
    "objective": _objective,
    "tax": _tax,
    "stamp_duty": _stamp_duty,
    "registrar": _registrar,
    "fund_managers": _fund_managers,
    "top_holdings": _top_holdings,
    "returns": _returns,
    "risk_ratios": _risk_ratios,
}


def lookup(scheme_ids: list[str], fields: list[str]) -> list[FactAnswer]:
    """One FactAnswer per (scheme, field). Unknown fields are ignored."""
    con = sqlite3.connect(config.SQLITE_PATH)
    con.row_factory = sqlite3.Row
    try:
        answers = []
        for scheme_id in scheme_ids:
            s = _scheme_row(con, scheme_id)
            if s is None:
                continue
            for field in fields:
                if field in FACT_FIELDS:
                    answers.append(FactAnswer(scheme_id, s["scheme_name"], field, FACT_FIELDS[field](con, s),
                                              s["source_url"], s["scraped_at"][:10]))
        return answers
    finally:
        con.close()


def all_scheme_ids() -> list[str]:
    con = sqlite3.connect(config.SQLITE_PATH)
    try:
        return [r[0] for r in con.execute("SELECT scheme_id FROM schemes ORDER BY scheme_name")]
    finally:
        con.close()
