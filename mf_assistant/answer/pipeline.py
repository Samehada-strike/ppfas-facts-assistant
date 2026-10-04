"""Online: question → route → handler → AnswerDraft.

Phase 4 wires the paths together:
- sql_fact: exact sentences from SQLite, cited to the scheme's own page.
- retrieval: hybrid search results (the grounded LLM answer is Phase 6).
- every other handler: a fixed message (refusal wording and educational links are refined in Phase 5).

One citation per answer (a brief requirement): a single-scheme fact cites that scheme's page; facts
spanning several schemes cite the AMC's Groww page, which lists them all; retrieval cites the top chunk.

Run:  python -m mf_assistant.answer.pipeline "your question"
"""

import sys
from dataclasses import dataclass, field

from mf_assistant import config
from mf_assistant.answer.router import RouteDecision, Router
from mf_assistant.answer.sql_answers import FactAnswer, lookup
from mf_assistant.retrieval.hybrid import Hit, HybridRetriever

EDUCATION_URL = "https://groww.in/blog/mutual-funds-things-you-should-know-as-a-beginner"

MESSAGES = {
    "refuse_advice": "I can only share facts about PPFAS mutual fund schemes, not recommendations on what to buy, sell or hold.",
    "refuse_comparison": "I can share each fund's published figures, but I don't compare fund performance.",
    "off_topic": "I can only answer factual questions about PPFAS mutual fund schemes and Groww's mutual fund features.",
    "small_talk": "Hi! I answer factual questions about PPFAS mutual fund schemes on Groww, like expense ratios, exit loads, lock-ins or how to download statements.",
    "ask_scheme": "Which PPFAS scheme do you mean? I cover Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, Conservative Hybrid and Dynamic Asset Allocation.",
    "not_found": "I couldn't find this in my sources.",
}


@dataclass
class AnswerDraft:
    question: str
    route: RouteDecision
    text: str | None                  # None = needs generation (retrieval path, Phase 6)
    citation_url: str | None
    last_updated: str | None          # from the cited source's scraped_at
    facts: list[FactAnswer] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


class Pipeline:
    def __init__(self):
        self.router = Router()
        self.retriever = HybridRetriever()

    def answer(self, question: str) -> AnswerDraft:
        route = self.router.route(question)

        if route.handler == "sql_fact":
            facts = lookup(route.scheme_ids, route.fact_fields)
            if not facts:
                return AnswerDraft(question, route, MESSAGES["not_found"], None, None)
            single_scheme = len({f.scheme_id for f in facts}) == 1
            return AnswerDraft(
                question, route, " ".join(f.sentence for f in facts),
                citation_url=facts[0].source_url if single_scheme else config.AMC_SOURCE_URL,
                last_updated=max(f.scraped_at for f in facts), facts=facts,
            )

        if route.handler == "retrieval":
            result = self.retriever.retrieve(question)
            if not result.confident or not result.hits:
                return AnswerDraft(question, route, MESSAGES["not_found"], None, None)
            top = result.hits[0].metadata
            return AnswerDraft(question, route, None, top["source_url"], top["scraped_at"], hits=result.hits)

        url = EDUCATION_URL if route.handler in ("refuse_advice", "refuse_comparison") else None
        return AnswerDraft(question, route, MESSAGES[route.handler], url, None)


def _show(d: AnswerDraft):
    print(f"Q: {d.question}")
    print(f"   route: {d.route.intent} → {d.route.handler}  schemes={d.route.scheme_ids}  fields={d.route.fact_fields}")
    if d.text:
        print(f"   draft: {d.text}")
    for h in d.hits[:3]:
        print(f"   hit:   {h.score:.4f}  {h.metadata['title'][:80]}")
    print(f"   cite:  {d.citation_url}  (last updated {d.last_updated})\n")


if __name__ == "__main__":
    p = Pipeline()
    for q in sys.argv[1:] or ["What is the exit load of PPFAS ELSS?"]:
        _show(p.answer(q))
