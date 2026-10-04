"""Online: the guarded question → reply pipeline.

Every message passes the same layers, in this order:

  1. PII guard (input)      raw message; on a hit: refuse, don't send it anywhere, don't store it
  2. Condense (history)     rewrite a follow-up into a standalone question (LLM; only if there is history)
  3. Advice net (input)     deterministic patterns on raw + standalone question → refuse early
  4. Router                 LLM classifies; code picks funds and the handler
  5. Handler                SQL fact | retrieval | fixed message
  6. Output guard           strip extra links, ≤3 sentences, block advice language, one link + date footer
  7. History                store (standalone question, reply) for the next follow-up

Retrieval answers are interim here (the best source's own first sentences);
Phase 6 replaces them with a grounded LLM answer.

Run:  python -m mf_assistant.answer.pipeline "question" ["follow-up" ...]   (one shared session)
"""

import re
import sys
from dataclasses import dataclass, field

from mf_assistant import config
from mf_assistant.answer.history import ChatSession, Condenser
from mf_assistant.answer.router import RouteDecision, Router
from mf_assistant.answer.sql_answers import FactAnswer, lookup
from mf_assistant.guardrails.output import FinalAnswer, finalize
from mf_assistant.guardrails.pii import detect_pii
from mf_assistant.guardrails.policy import asks_for_advice
from mf_assistant.retrieval.hybrid import Hit, HybridRetriever

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
    "ask_scheme": ("Which PPFAS scheme do you mean? I cover Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, "
                   "Conservative Hybrid and Dynamic Asset Allocation.", config.AMC_SOURCE_URL),
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

    @property
    def text(self) -> str:
        return self.answer.text


def _fixed(key: str) -> FinalAnswer:
    text, url = MESSAGES[key]
    return finalize(text, url, None)


def _interim_retrieval_body(hit: Hit) -> str:
    """Until Phase 6: the best chunk's own text, minus its header and 'Question:' line."""
    body = hit.text.split("\n", 1)[-1]
    body = re.sub(r"^Question:.*\n", "", body)
    return re.sub(r"^Answer:\s*", "", body).strip()


class Assistant:
    def __init__(self):
        self.condenser = Condenser()
        self.router = Router()
        self.retriever = HybridRetriever()

    def ask(self, message: str, session: ChatSession | None = None) -> Reply:
        # 1. PII guard: before any LLM call, before history
        if detect_pii(message).found:
            return Reply(message, None, "pii", _fixed("pii"))

        # 2. Resolve follow-ups against history
        standalone = self.condenser.standalone(message, session)

        # 3. Deterministic advice net (either version of the question)
        if asks_for_advice(message) or asks_for_advice(standalone):
            return self._remember(session, Reply(message, standalone, "refuse_advice", _fixed("refuse_advice")))

        # 4. Route, 5. handle
        route = self.router.route(standalone)
        reply = self._handle(message, standalone, route)
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
                return Reply(message, standalone, "not_found", _fixed("not_found"), route)
            top = result.hits[0]
            answer = finalize(_interim_retrieval_body(top), top.metadata["source_url"], top.metadata["scraped_at"])
            return Reply(message, standalone, "retrieval", answer, route, hits=result.hits)

        return Reply(message, standalone, route.handler, _fixed(route.handler), route)

    @staticmethod
    def _remember(session: ChatSession | None, reply: Reply) -> Reply:
        if session is not None:
            session.add(reply.standalone or reply.message, reply.answer.body)
        return reply


if __name__ == "__main__":
    assistant, session = Assistant(), ChatSession()
    for msg in sys.argv[1:] or ["What is the exit load of PPFAS ELSS?", "and its lock-in?"]:
        r = assistant.ask(msg, session)
        print(f"USER: {msg}")
        if r.standalone and r.standalone != msg:
            print(f"   (standalone: {r.standalone})")
        print(f"   [{r.handler}]")
        print("   " + r.text.replace("\n", "\n   ") + "\n")
