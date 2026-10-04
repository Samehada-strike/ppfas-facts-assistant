"""Online: detect which PPFAS scheme(s) a question is about.

Matching is deterministic: each scheme has a list of aliases matched as whole
words (longest alias first), so "liquid" matches but "liquidity" doesn't, and
"long term capital gains" does not match the Flexi Cap fund's old name
("long term value"). The old notebook's alias map had exactly that bug.

The detected scheme_ids are used two ways downstream:
- as a filter: search only that scheme's chunks plus general knowledge;
- to strip the scheme name from the keyword query, so BM25 scores the intent
  words ("take money out early") instead of the fund name every chunk shares.
"""

import re
import sqlite3
from dataclasses import dataclass

from mf_assistant import config

# Keyed by a distinctive word in the scheme_id; matched as whole words, case-insensitive
SCHEME_ALIASES = {
    "long-term-value": ["flexi cap", "flexicap", "flexi-cap", "long term value", "ppfcf"],
    "liquid": ["liquid fund", "liquid"],
    "elss": ["elss", "tax saver", "tax saving", "tax-saver", "taxsaver"],
    "conservative-hybrid": ["conservative hybrid", "conservative"],
    "arbitrage": ["arbitrage"],
    "dynamic-asset-allocation": ["dynamic asset allocation", "dynamic asset", "daaf"],
}
# Mentions of the fund house, not of a specific scheme (stripped from keyword queries too)
AMC_ALIASES = ["parag parikh", "ppfas mutual fund", "ppfas mf", "ppfas", "pp"]
GENERIC_WORDS = ["direct growth", "direct plan", "mutual fund", "fund", "scheme", "plan"]


@dataclass
class SchemeMatch:
    scheme_ids: list[str]  # empty = no specific scheme mentioned
    stripped_query: str     # query without scheme/AMC mentions, for keyword search


def _pattern(aliases: list[str]) -> re.Pattern:
    alts = sorted(aliases, key=len, reverse=True)  # longest first: "liquid fund" before "liquid"
    return re.compile(r"\b(" + "|".join(re.escape(a) for a in alts) + r")\b", re.IGNORECASE)


class SchemeResolver:
    def __init__(self):
        con = sqlite3.connect(config.SQLITE_PATH)
        rows = con.execute("SELECT scheme_id, scheme_name FROM schemes").fetchall()
        con.close()
        self.names = dict(rows)
        self.patterns = {}
        for key, aliases in SCHEME_ALIASES.items():
            scheme_id = next(sid for sid in self.names if key in sid)
            self.patterns[scheme_id] = _pattern(aliases)
        self.amc_pattern = _pattern(AMC_ALIASES)
        self.generic_pattern = _pattern(GENERIC_WORDS)

    def resolve(self, query: str) -> SchemeMatch:
        found = [sid for sid, pat in self.patterns.items() if pat.search(query)]
        stripped = query
        for sid in found:
            stripped = self.patterns[sid].sub(" ", stripped)
        if found or self.amc_pattern.search(query):
            stripped = self.generic_pattern.sub(" ", self.amc_pattern.sub(" ", stripped))
        return SchemeMatch(found, re.sub(r"\s+", " ", stripped).strip() or query)
