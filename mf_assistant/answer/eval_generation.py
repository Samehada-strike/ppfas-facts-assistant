"""Evaluate grounded generation (retrieval → generator), including an LLM-as-a-judge.

Answerable questions (the corpus covers them):
  answered       generator returned found=True and passed the number check
  cited right    the cited passage matches the expected source pattern
  faithful       a judge model finds no claim unsupported by the passages
  relevant       the judge says the answer addresses the question
  ≤3 sentences   after the output guard

Unanswerable questions (the corpus doesn't cover them):
  declined       generator said found=False (instead of inventing an answer)

The judge is a different, stronger model than the generator (JUDGE_MODEL), because a model
grading its own output tends to be lenient. Judges are still imperfect: read the flagged claims.

Run:  python -m mf_assistant.answer.eval_generation
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from mf_assistant.answer.generate import Generator, format_context
from mf_assistant.guardrails.output import finalize, split_sentences
from mf_assistant.guardrails.policy import gives_advice
from mf_assistant.retrieval.hybrid import HybridRetriever

JUDGE_MODEL = "gpt-4o"

ANSWERABLE = [
    ("How can I stop my SIP?", r"cancel"),
    ("what is an exit load?", r"exit load"),
    ("which day's NAV applies when I redeem?", r"which day's NAV"),
    ("How do I generate my CAS from CAMS?", r"Consolidated Account Statement"),
    ("How do I download my capital gains statement?", r"CAS|ELSS report|Consolidated"),
    ("what is a folio?", r"folio"),
    ("what is a step-up SIP?", r"Step-Up"),
    ("how long does a redemption take to reach my bank?", r"redeem|redemption|credited"),
    ("Can I pay for mutual funds with a credit card?", r"credit/debit card"),
    ("What are the AutoPay charges?", r"AutoPay charges"),
    ("how do I switch from a regular plan to a direct plan on Groww?", r"switch"),
    ("why did my SIP installment fail?", r"fail"),
    ("what is an NFO?", r"NFO"),
    ("can I skip an SIP installment?", r"skip"),
    ("what does AUM mean?", r"Jargon|AUM"),
    ("are mutual funds safe for beginners?", r"safe for beginners|Risks"),
]

UNANSWERABLE = [
    "how do I add a nominee to my mutual funds on Groww?",
    "what is the penalty for missing a PPF deposit?",
    "how do I apply for an IPO on Groww?",
    "what is the GST rate on brokerage for stocks?",
]


class Judgement(BaseModel):
    unsupported_claims: list[str] = Field(
        description="Each claim in the answer that the passages do not state or directly imply. Empty if none.")
    faithful: bool = Field(description="True if every claim in the answer is supported by the passages.")
    answers_question: bool = Field(description="True if the answer addresses what the question asked.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict evaluator of a retrieval-augmented assistant. Check the answer only against "
               "the passages: a claim counts as supported only if the passages state it or directly imply it. "
               "General knowledge does not count as support."),
    ("human", "<passages>\n{context}\n</passages>\n\nQuestion: {question}\n\nAnswer: {answer}"),
])


def main():
    retriever, generator = HybridRetriever(), Generator()
    judge = JUDGE_PROMPT | ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(Judgement)

    stats = dict(answered=0, cited=0, faithful=0, relevant=0, short=0, no_advice=0)
    print("ANSWERABLE")
    for question, source_pattern in ANSWERABLE:
        hits = retriever.retrieve(question).hits
        gen = generator.generate(question, hits)
        if not gen.found:
            print(f"  ✗ not answered: {question!r}" + (f" (unsupported numbers {gen.unsupported_numbers})"
                                                      if gen.unsupported_numbers else ""))
            continue
        stats["answered"] += 1
        cited = bool(re.search(source_pattern, gen.source.metadata["title"] + " " + gen.source.text[:300], re.I))
        final = finalize(gen.answer, gen.source.metadata["source_url"], gen.source.metadata["scraped_at"])
        verdict: Judgement = judge.invoke({"context": format_context(hits), "question": question,
                                           "answer": gen.answer})
        stats["cited"] += cited
        stats["faithful"] += verdict.faithful
        stats["relevant"] += verdict.answers_question
        stats["short"] += len(split_sentences(final.body)) <= 3
        stats["no_advice"] += not gives_advice(gen.answer)
        flags = [] if cited else ["citation"]
        flags += [] if verdict.faithful else ["UNFAITHFUL"]
        flags += [] if verdict.answers_question else ["off-target"]
        print(f"  {'✓' if not flags else '!'} {question!r}" + (f"  [{', '.join(flags)}]" if flags else ""))
        if not verdict.faithful:
            print(f"      answer: {gen.answer}")
            for claim in verdict.unsupported_claims:
                print(f"      unsupported: {claim}")

    declined = 0
    print("\nUNANSWERABLE")
    for question in UNANSWERABLE:
        gen = generator.generate(question, retriever.retrieve(question).hits)
        declined += not gen.found
        print(f"  {'✓ declined' if not gen.found else '✗ answered'}: {question!r}" + ("" if not gen.found else f" → {gen.answer}"))

    n = len(ANSWERABLE)
    answered = stats["answered"] or 1
    print(f"\nAnswerable ({n}): answered {stats['answered']}/{n}; of those answered: cited right "
          f"{stats['cited']}/{answered}, faithful {stats['faithful']}/{answered}, relevant "
          f"{stats['relevant']}/{answered}, ≤3 sentences {stats['short']}/{answered}, no advice {stats['no_advice']}/{answered}")
    print(f"Unanswerable ({len(UNANSWERABLE)}): declined {declined}/{len(UNANSWERABLE)}")


if __name__ == "__main__":
    main()
