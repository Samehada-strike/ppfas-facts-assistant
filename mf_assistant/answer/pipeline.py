"""Online: the guarded question → reply pipeline.

Every message passes the same layers, in this order:

  1. PII guard (input)      raw message; on a hit: refuse, don't send it anywhere, don't store it
  2. Condense (history)     rewrite a follow-up into a standalone question (only if USE_CHAT_HISTORY)
  3. Advice net (input)     deterministic patterns on raw + standalone question → refuse early
  4. Decompose + route      split a mixed question (only if it might be one); LLM classifies each part,
                            code picks funds and the handler; fact parts about the same funds re-merge
  5. Handler                SQL fact | retrieval + grounded generation | fixed message
  6. Output guard           strip extra links, ≤3 sentences, block advice language, one link + date footer
  7. History                store (standalone question, reply) for the next follow-up (only if USE_CHAT_HISTORY)

Run:  python -m mf_assistant.answer.pipeline "question" ["another question" ...]
"""

import sys
from dataclasses import dataclass, field

from mf_assistant import config
from mf_assistant.answer.decompose import Decomposer
from mf_assistant.answer.generate import Generation, Generator
from mf_assistant.answer.history import ChatSession, Condenser
from mf_assistant.answer.router import RouteDecision, Router
from mf_assistant.answer.sql_answers import FactAnswer, lookup
from mf_assistant.guardrails.output import FinalAnswer, finalize
from mf_assistant.guardrails.pii import detect_pii
from mf_assistant.guardrails.policy import refusal_kind
from mf_assistant.retrieval.hybrid import Hit, HybridRetriever

_SCHEMES = "Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, Conservative Hybrid and Dynamic Asset Allocation"

# Fixed replies: (text, link). Each is at most 3 sentences and carries one link.
MESSAGES = {
    "pii": ("For your privacy, please don't share personal details such as PAN, Aadhaar, phone numbers, email, "
            "account or folio numbers, or OTPs. I haven't processed or stored your message. "
            "Please ask your question again without them.", config.HELP_CENTRE_URL),
    "refuse_advice": ("I can't give investment advice or recommendations. I can share facts such as a PPFAS "
                      "scheme's expense ratio, exit load, lock-in or published returns. "
                      "This beginner's guide explains how to evaluate mutual funds yourself.", config.EDUCATION_URL),
    "refuse_comparison": ("I don't compare fund performance. I can share each scheme's published returns "
                          "separately, for example: \"What are the returns of the ELSS fund?\"", config.EDUCATION_URL),
    "off_topic": ("I can only help with factual questions about PPFAS mutual fund schemes and Groww's mutual "
                  "fund features.", config.HELP_CENTRE_URL),
    "small_talk": ("Happy to help! I answer factual questions about PPFAS mutual fund schemes on Groww, such as "
                   "expense ratios, exit loads, lock-in periods or how to download statements.", config.AMC_SOURCE_URL),
    # With history, a one-word reply ("ELSS") is resolved against the question; without it, ask for the full question
    "ask_scheme": (f"Which PPFAS scheme do you mean? I cover {_SCHEMES}."
                   if config.USE_CHAT_HISTORY else
                   f"Please include the scheme name in your question, for example \"What is the lock-in of the "
                   f"ELSS fund?\". I cover {_SCHEMES}.", config.AMC_SOURCE_URL),
    "not_found": ("I couldn't find this in my sources. Groww's help centre may have more on it.", config.HELP_CENTRE_URL),
}


@dataclass
class Reply:
    message: str                       # what the user typed
    standalone: str | None             # after condensing (None if refused before that step)
    handler: str                       # pii | refuse_advice | ... | sql_fact | retrieval | not_found
    answer: FinalAnswer
    route: RouteDecision | None = None
    facts: list[FactAnswer] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    generation: Generation | None = None
    parts: list["Reply"] = field(default_factory=list)  # set when a mixed question was split

    @property
    def text(self) -> str:
        return self.answer.text


def _fixed(key: str) -> FinalAnswer:
    text, url = MESSAGES[key]
    return finalize(text, url, None)


def _merge_fact_routes(routes: list[RouteDecision]) -> list[RouteDecision]:
    """Re-join SQL-fact parts that the decomposer split unnecessarily.

    Two fact parts merge when they cover the same funds (→ union of fields) or the same
    fields (→ union of funds), so "expense ratio and benchmark of Flexi Cap" becomes one
    lookup with one citation, whatever the LLM did. Unrelated facts stay separate.
    """
    merged: list[RouteDecision] = []
    for r in routes:
        target = next((m for m in merged if m.handler == r.handler == "sql_fact"
                       and (set(m.scheme_ids) == set(r.scheme_ids) or set(m.fact_fields) == set(r.fact_fields))),
                      None)
        if target is None:
            merged.append(r)
            continue
        target.scheme_ids += [s for s in r.scheme_ids if s not in target.scheme_ids]
        target.fact_fields += [f for f in r.fact_fields if f not in target.fact_fields]
        target.question = f"{target.question} {r.question}"
    return merged


def _combine(message: str, standalone: str, replies: list[Reply]) -> Reply:
    """One reply from several parts: each part keeps its own answer, source and date."""
    seen, unique = set(), []
    for r in replies:  # two parts that both end in the same fixed message: show it once
        if r.answer.text not in seen:
            seen.add(r.answer.text)
            unique.append(r)
    dates = [r.answer.last_updated for r in unique if r.answer.last_updated]
    combined = FinalAnswer(
        text="\n\n".join(r.answer.text for r in unique),
        body="\n\n".join(r.answer.body for r in unique),
        citation_url=unique[0].answer.citation_url,
        last_updated=max(dates) if dates else None,
    )
    return Reply(message, standalone, "multi", combined, parts=unique)


class Assistant:
    def __init__(self):
        self.condenser = Condenser() if config.USE_CHAT_HISTORY else None
        self.decomposer = Decomposer()
        self.router = Router()
        self.retriever = HybridRetriever()
        self.generator = Generator()

    def ask(self, message: str, session: ChatSession | None = None) -> Reply:
        if not config.USE_CHAT_HISTORY:
            session = None  # each question is answered on its own

        # 1. PII guard: before any LLM call, before history
        if detect_pii(message).found:
            return Reply(message, None, "pii", _fixed("pii"))

        # 2. Resolve follow-ups against history
        standalone = self.condenser.standalone(message, session) if self.condenser else message

        # 3. Deterministic advice / comparison net (either version of the question)
        refusal = refusal_kind(message) or refusal_kind(standalone)
        if refusal:
            return self._remember(session, Reply(message, standalone, refusal, _fixed(refusal)))

        # 4. Decompose a mixed question, route each part
        parts = self.decomposer.split(standalone)
        routes = [self.router.route(p) for p in parts]

        # Any part asking for advice or a comparison → refuse the whole message (the safe side wins)
        for refusal in ("refuse_advice", "refuse_comparison"):
            if any(r.handler == refusal for r in routes):
                return self._remember(session, Reply(message, standalone, refusal, _fixed(refusal), routes[0]))

        routes = _merge_fact_routes(routes)
        if len(routes) > 1 and all(r.handler == "retrieval" for r in routes):
            # Splitting only helps when parts need different handlers (fact + explanation). Related how-to
            # parts lose context when split ("when will the money reach my bank?" without "after redeeming"),
            # so explanation-only questions are retrieved and answered as one.
            routes = [self.router.route(standalone)]
        if len(routes) > 1:  # pleasantries next to a real question need no reply of their own
            routes = [r for r in routes if r.handler != "small_talk"] or routes[:1]

        # 5. Handle each part
        replies = [self._handle(message, r.question, r) for r in routes]
        reply = replies[0] if len(replies) == 1 else _combine(message, standalone, replies)
        return self._remember(session, reply)

    def _handle(self, message: str, standalone: str, route: RouteDecision) -> Reply:
        if route.handler == "sql_fact":
            facts = lookup(route.scheme_ids, route.fact_fields)
            if not facts:
                return Reply(message, standalone, "not_found", _fixed("not_found"), route)
            if len({f.scheme_id for f in facts}) == 1:
                body, url = " ".join(f.sentence for f in facts), facts[0].source_url
            else:  # several schemes: one bullet per fact; cite the AMC page that lists them all
                body, url = "\n".join(f"- {f.sentence}" for f in facts), config.AMC_SOURCE_URL
            answer = finalize(body, url, max(f.scraped_at for f in facts))
            return Reply(message, standalone, "sql_fact", answer, route, facts=facts)

        if route.handler == "retrieval":
            result = self.retriever.retrieve(standalone)
            if not result.confident or not result.hits:
                return Reply(message, standalone, "not_found", _fixed("not_found"), route, hits=result.hits)
            gen = self.generator.generate(standalone, result.hits)
            if not gen.found:
                return Reply(message, standalone, "not_found", _fixed("not_found"), route,
                             hits=result.hits, generation=gen)
            # Cite the passage the answer was written from, not simply the top-ranked one
            meta = gen.source.metadata
            answer = finalize(gen.answer, meta["source_url"], meta["scraped_at"])
            return Reply(message, standalone, "retrieval", answer, route, hits=result.hits, generation=gen)

        return Reply(message, standalone, route.handler, _fixed(route.handler), route)

    @staticmethod
    def _remember(session: ChatSession | None, reply: Reply) -> Reply:
        if session is not None:
            session.add(reply.standalone or reply.message, reply.answer.body)
        return reply


if __name__ == "__main__":
    assistant, session = Assistant(), ChatSession()
    for msg in sys.argv[1:] or ["How do I download my capital gains statement?"]:
        r = assistant.ask(msg, session)
        print(f"USER: {msg}")
        if r.standalone and r.standalone != msg:
            print(f"   (standalone: {r.standalone})")
        print(f"   [{r.handler}]")
        print("   " + r.text.replace("\n", "\n   ") + "\n")
