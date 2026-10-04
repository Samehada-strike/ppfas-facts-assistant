"""Online: decide what kind of question this is and which path handles it.

Two parts, each doing what it is reliable at:
- An LLM classifies the question into a typed Route (structured output):
  intent, which fact fields, and safety flags. It never answers the question.
- Deterministic code (SchemeResolver) decides which fund(s) are meant. An LLM
  could hallucinate a scheme; whole-word alias matching can't.

Then plain Python maps (intent, schemes) to a handler:
  FACT        → SQL lookup (exact value + citation)
  EXPLAIN     → hybrid retrieval → (Phase 6) grounded LLM answer
  ADVICE      → polite refusal + educational link
  PERFORMANCE_COMPARISON → refusal: published figures can be quoted per fund, never compared
  OFF_TOPIC   → polite scope message
  SMALL_TALK  → welcome message
  FACT without a recognisable fund → ask which fund
"""

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from mf_assistant import config
from mf_assistant.answer.sql_answers import FACT_FIELDS, all_scheme_ids
from mf_assistant.retrieval.schemes import SchemeResolver

FactField = Literal[tuple(FACT_FIELDS)]  # the router may only pick fields SQL can answer


class Route(BaseModel):
    """How to handle a user question about PPFAS (Parag Parikh) mutual funds on Groww."""

    intent: Literal["FACT", "EXPLAIN", "ADVICE", "PERFORMANCE_COMPARISON", "OFF_TOPIC", "SMALL_TALK"] = Field(
        description=(
            "FACT: asks for a specific attribute of a specific PPFAS scheme (its expense ratio, exit load, NAV, "
            "returns, managers, holdings, lock-in, benchmark, riskometer...). "
            "EXPLAIN: asks what a concept means, how to do something on Groww / with mutual funds in general, or "
            "why something happened with the user's mutual fund orders, SIPs, payments, redemptions or statements "
            "on Groww (troubleshooting) (e.g. 'what is exit load', 'how do I download my CAS', 'how to cancel a "
            "SIP', 'why did my SIP fail', 'why is my order pending'). "
            "ADVICE: asks for a recommendation or opinion: should I buy/sell/hold/switch, is it good, which is "
            "better/best, how much to invest, suitability for the user. "
            "PERFORMANCE_COMPARISON: asks to compare returns/performance between funds or against a benchmark/"
            "category, or which performed better. "
            "OFF_TOPIC: not about mutual funds, Groww's mutual-fund features, or investing concepts. "
            "SMALL_TALK: greetings, thanks, or questions about the assistant itself."
        )
    )
    fact_fields: list[FactField] = Field(
        default_factory=list,
        description="For FACT only: every scheme attribute the question asks for. "
                    "'minimum SIP/lumpsum/investment' → min_investment; 'returns/CAGR/performance of one fund' → returns; "
                    "'sharpe/beta/alpha/standard deviation' → risk_ratios; 'who manages' → fund_managers; "
                    "'how risky/risk level' → riskometer; 'statement/registrar/RTA' → registrar; 'tax' → tax.",
    )
    all_schemes: bool = Field(
        default=False, description="True if the question asks about all / every PPFAS scheme at once.")


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You classify questions for a facts-only mutual fund assistant. It covers six PPFAS (Parag Parikh) schemes "
     "listed on Groww (Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, Conservative Hybrid, Dynamic Asset "
     "Allocation) plus Groww's help-centre FAQs about mutual funds. Identify exactly what is being asked. Do not "
     "answer the question. When a question both asks a fact and seeks a recommendation, choose ADVICE."),
    ("human", "{question}"),
])


@dataclass
class RouteDecision:
    question: str
    intent: str
    handler: str  # "sql_fact" | "retrieval" | "refuse_advice" | "refuse_comparison" | "off_topic" | "small_talk" | "ask_scheme"
    scheme_ids: list[str] = field(default_factory=list)
    fact_fields: list[str] = field(default_factory=list)


class Router:
    def __init__(self):
        llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0)
        self.classifier = ROUTER_PROMPT | llm.with_structured_output(Route)
        self.resolver = SchemeResolver()

    def classify(self, question: str) -> Route:
        return self.classifier.invoke({"question": question})

    def route(self, question: str) -> RouteDecision:
        r = self.classify(question)
        scheme_ids = self.resolver.resolve(question).scheme_ids
        decision = RouteDecision(question, r.intent, handler="", scheme_ids=scheme_ids, fact_fields=r.fact_fields)

        if r.intent == "FACT":
            if r.all_schemes and not scheme_ids:
                decision.scheme_ids = all_scheme_ids()
            if not decision.fact_fields:
                decision.handler = "retrieval"   # a fact SQL doesn't hold: let retrieval find it
            elif not decision.scheme_ids:
                decision.handler = "ask_scheme"  # e.g. "what's the exit load?" with no fund named
            else:
                decision.handler = "sql_fact"
        else:
            decision.handler = {
                "EXPLAIN": "retrieval",
                "ADVICE": "refuse_advice",
                "PERFORMANCE_COMPARISON": "refuse_comparison",
                "OFF_TOPIC": "off_topic",
                "SMALL_TALK": "small_talk",
            }[r.intent]
        return decision
