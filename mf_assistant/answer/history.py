"""Online: short-term chat memory, used to rewrite follow-up questions.

Follow-ups only make sense with the previous turn in mind:
    User: What is the exit load of the ELSS fund?
    User: and its lock-in?                     → "What is the lock-in period of the ELSS fund?"
    Bot:  Which PPFAS scheme do you mean?
    User: the liquid one                       → "What is the minimum SIP of the Liquid fund?"

Rather than giving the whole history to every component, a "condense" step
rewrites the newest message into a standalone question once. Routing,
retrieval and SQL then work exactly as they do for a first question.

Privacy:
- History lives in memory only, per session, capped at HISTORY_TURNS turns. Nothing
  is written to disk or logged.
- Only turns that passed the PII check are stored, so personal data never
  enters history (and is never resent to the LLM in later rewrites).
"""

from collections import deque
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from mf_assistant import config


@dataclass
class Turn:
    question: str     # standalone version of what the user asked
    answer: str       # the assistant's reply body (no footer)


class ChatSession:
    """In-memory history for one user's conversation."""

    def __init__(self, max_turns: int = config.HISTORY_TURNS):
        self.turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str):
        self.turns.append(Turn(question, answer))

    def transcript(self) -> str:
        return "\n".join(f"User: {t.question}\nAssistant: {t.answer[:300]}" for t in self.turns)

    def clear(self):
        self.turns.clear()


CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as a standalone question. Follow-ups often leave out either the fund "
    "or the attribute from the previous user question: carry over whichever is missing. A message that names "
    "a new fund but no attribute ('what about X?', 'and X?', 'the X one') reuses the previous attribute; a "
    "message that names an attribute but no fund ('and its lock-in?') reuses the previous fund. "
    "A message that already names its own fund(s), or says all/every fund, keeps exactly those funds. "
    "Keep the user's intent exactly: never add requests, opinions, facts or advice. If the message is already "
    "a complete question, or is a greeting or thanks, return it unchanged. Return only the question."
)

# Few-shot examples use different funds than typical test questions, so the model learns the pattern
# rather than copying answers.
CONDENSE_EXAMPLES = [
    ("User: What is the expense ratio of the Arbitrage fund?\nAssistant: The expense ratio of Parag Parikh Arbitrage Fund Direct Growth is 0.99%.",
     "and its benchmark?", "What is the benchmark of the Arbitrage fund?"),
    ("User: What is the expense ratio of the Arbitrage fund?\nAssistant: The expense ratio of Parag Parikh Arbitrage Fund Direct Growth is 0.99%.",
     "what about the conservative hybrid fund?", "What is the expense ratio of the Conservative Hybrid fund?"),
    ("User: Who manages this fund?\nAssistant: Which PPFAS scheme do you mean? I cover Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, Conservative Hybrid and Dynamic Asset Allocation.",
     "dynamic asset allocation", "Who manages the Dynamic Asset Allocation fund?"),
    ("User: What is the exit load of the Arbitrage fund?\nAssistant: The exit load of Parag Parikh Arbitrage Fund Direct Growth is: Exit load of 0.25%, if redeemed within 30 days.",
     "How do I cancel a SIP on Groww?", "How do I cancel a SIP on Groww?"),
    ("User: What is the exit load of the Arbitrage fund?\nAssistant: The exit load of Parag Parikh Arbitrage Fund Direct Growth is: Exit load of 0.25%, if redeemed within 30 days.",
     "thanks!", "thanks!"),
    ("User: What is the exit load of the Arbitrage fund?\nAssistant: The exit load of Parag Parikh Arbitrage Fund Direct Growth is: Exit load of 0.25%, if redeemed within 30 days.",
     "benchmark of every PPFAS fund", "What is the benchmark of every PPFAS fund?"),
]

_TEMPLATE = "Conversation:\n{transcript}\n\nLatest message: {question}"
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONDENSE_SYSTEM)]
    + [m for t, q, a in CONDENSE_EXAMPLES
       for m in (("human", _TEMPLATE.format(transcript=t, question=q).replace("{", "{{").replace("}", "}}")),
                 ("ai", a))]
    + [("human", _TEMPLATE)]
)


class Condenser:
    def __init__(self):
        self.chain = CONDENSE_PROMPT | ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0)

    def standalone(self, question: str, session: ChatSession | None) -> str:
        if not session or not session.turns:
            return question  # first message: nothing to resolve, no extra LLM call
        out = self.chain.invoke({"transcript": session.transcript(), "question": question})
        return out.content.strip().strip('"') or question
