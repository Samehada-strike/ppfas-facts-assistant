"""Online: split a mixed question into standalone sub-questions.

The router gives each message one intent, so a mixed question loses a half:
    "What is the exit load of the ELSS fund and how do I download my CAS?"
    → a SQL fact (exit load)  +  a help-centre explanation (CAS download)

Decomposition rewrites such a message into 1-MAX_SUB_QUESTIONS standalone sub-questions;
each then runs through the normal route → handler → output-guard path.

Cost control: a deterministic gate (`might_be_multi_part`) decides whether the LLM is asked
at all. Most messages have no "and", "also", ";" or second "?", so they skip the extra call.
A gate false positive is harmless: the decomposer returns the question as a single part.
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from mf_assistant import config

_GATE = re.compile(r"\?.*\S.*\?|\b(and|also|plus|as well as)\b|;", re.I)


def might_be_multi_part(question: str) -> bool:
    return bool(_GATE.search(question))


class SubQuestions(BaseModel):
    """The user's message as standalone sub-questions."""

    questions: list[str] = Field(
        description=f"1 to {config.MAX_SUB_QUESTIONS} standalone questions that together cover the whole message.")


SYSTEM = (
    "Split the user's message into standalone sub-questions only when it asks for things that need different "
    "kinds of answers or are about unrelated topics, for example a fact about a specific fund plus a how-to or "
    "concept question. Keep several attributes or several funds of the same request together as ONE question "
    "('expense ratio and benchmark of the Flexi Cap fund', 'exit load of ELSS and Flexi Cap'). Each "
    "sub-question must make sense on its own: repeat the fund name where needed. Keep the user's wording and "
    "intent; never add requests, advice or facts. Drop pure greetings or thanks when there is a real question. "
    "If the message is one request, return it unchanged as the only item."
)

EXAMPLES = [
    ("What is the expense ratio of the Arbitrage fund and how do I set up AutoPay?",
     ["What is the expense ratio of the Arbitrage fund?", "How do I set up AutoPay?"]),
    ("benchmark and riskometer of the dynamic asset allocation fund",
     ["benchmark and riskometer of the dynamic asset allocation fund"]),
    ("Who manages the conservative hybrid fund? Also, what is a folio?",
     ["Who manages the Conservative Hybrid fund?", "What is a folio?"]),
    ("How do I redeem and when will the money reach my bank?",
     ["How do I redeem and when will the money reach my bank?"]),
]

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM)]
    + [m for q, parts in EXAMPLES
       for m in (("human", q), ("ai", "\n".join(parts)))]
    + [("human", "{question}")]
)


class Decomposer:
    def __init__(self):
        llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0)
        self.chain = PROMPT | llm.with_structured_output(SubQuestions)

    def split(self, question: str) -> list[str]:
        if not might_be_multi_part(question):
            return [question]
        parts = [q.strip() for q in self.chain.invoke({"question": question}).questions if q.strip()]
        return parts[:config.MAX_SUB_QUESTIONS] or [question]
