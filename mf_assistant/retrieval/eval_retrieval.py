"""Measure retrieval quality on a small labelled question set.

For each question we know which chunk(s) answer it (a regex over the chunk's
title + start of its text). Reported per configuration:
- hit@1 / hit@3 / hit@5: share of questions with a correct chunk in the top 1/3/5
- MRR: mean of 1/rank of the first correct chunk (1.0 = always first)

A second set of off-topic questions calibrates MIN_DENSE_COSINE: in-scope
questions should score above the threshold, off-topic ones below it.

Run:  python -m mf_assistant.retrieval.eval_retrieval
"""

import re

from mf_assistant import config
from mf_assistant.retrieval.hybrid import HybridRetriever

# (question, regex matching an acceptable chunk's "title || text-start")
LABELLED = [
    # scheme facts (paraphrased; these will go to SQL in Phase 4, but retrieval should still find them)
    ("Can I take my money out of the tax saver fund early?", r"ELSS.*key facts|Redeem.*ELSS"),
    ("lock in period of ppfas elss", r"ELSS.*key facts"),
    ("exit load for the flexi cap fund", r"Flexi Cap.*key facts"),
    ("expense ratio of parag parikh liquid fund", r"Liquid.*(key facts|expense ratio)"),
    ("who manages the arbitrage fund", r"Arbitrage.*fund managers"),
    ("benchmark of the conservative hybrid fund", r"Conservative Hybrid.*key facts"),
    ("riskometer of dynamic asset allocation fund", r"Dynamic Asset.*key facts"),
    ("top holdings of PPFAS flexi cap", r"Flexi Cap.*top holdings"),
    ("minimum SIP amount for ELSS", r"ELSS.*(key facts|SIP and Lump Sum)"),
    ("3 year return of the liquid fund", r"Liquid.*(returns and rankings|kind of returns)"),
    ("Sharpe ratio of the ELSS fund", r"ELSS.*returns and rankings"),
    ("investment objective of the arbitrage fund", r"Arbitrage.*investment objective"),
    # statements and tax documents
    ("How do I download my capital gains statement?", r"Consolidated Account Statement|What is CAS|ELSS report"),
    ("How to get CAS from CAMS", r"generate my Consolidated Account Statement"),
    ("tax statement for my ELSS investments", r"ELSS report"),
    # concepts
    ("What is an exit load?", r"What is exit load|Exit Load"),
    ("what does NAV mean", r"NAVs priced|\|\| NAV|Net Asset Value"),
    ("what is a folio number", r"What is a folio"),
    ("what is a step up SIP", r"Step-Up SIP"),
    ("are mutual funds safe for beginners", r"safe for beginners"),
    ("what are the types of mutual funds", r"Types of Mutual Funds|categories are as follows|Different Types"),
    ("what does AUM mean", r"\|\| AUM|Assets Under Management"),
    ("what is an NFO", r"What is NFO"),
    # Groww app how-tos
    ("How can I stop my SIP?", r"cancel an ongoing SIP|cancel my SIP"),
    ("my sip money got deducted but the order is not showing", r"deducted but (the )?order is not reflected"),
    ("how to change my SIP date", r"change my SIP date"),
    ("skip one sip installment", r"skip an SIP installment|skip my SIP"),
    ("switch from regular plan to direct plan", r"regular fund to direct fund|switch to direct|switch from regular"),
    ("when will redemption money reach my bank", r"redeem amount to reflect|credited to my bank|redemption has been completed"),
    ("which day's NAV applies when I redeem", r"which day's NAV"),
    ("autopay charges", r"AutoPay charges"),
    ("can I pay for mutual funds with a credit card", r"credit/debit card"),
]

OFF_TOPIC = [
    "best pizza place in Mumbai", "who won the cricket world cup", "how to bake a chocolate cake",
    "what is the capital of France", "write a poem about the sea", "how do I renew my passport",
    "bitcoin price today", "weather in Delhi tomorrow", "how to file GST returns",
    "how to open a fixed deposit in SBI",
]

CONFIGS = [
    ("dense only (no scheme filter)", dict(mode="dense", use_schemes=False)),
    ("bm25 only (no scheme filter)", dict(mode="bm25", use_schemes=False)),
    ("hybrid RRF (no scheme filter)", dict(mode="hybrid", use_schemes=False)),
    ("dense + scheme filter", dict(mode="dense")),
    ("bm25 + filter + name strip", dict(mode="bm25", strip_names=True)),
    ("hybrid + filter + name strip", dict(mode="hybrid", strip_names=True)),
    ("bm25 + scheme filter", dict(mode="bm25")),
    ("hybrid + scheme filter", dict(mode="hybrid")),
]


def first_correct_rank(result, pattern) -> int | None:
    for rank, hit in enumerate(result.hits, 1):
        if re.search(pattern, f"{hit.metadata['title']} || {hit.text.split(chr(10), 1)[-1][:300]}", re.I):
            return rank
    return None


def main():
    retriever = HybridRetriever()
    print(f"{len(LABELLED)} labelled questions\n")
    print(f"{'configuration':32} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}")
    misses = {}
    for name, kwargs in CONFIGS:
        ranks = [first_correct_rank(retriever.retrieve(q, k=5, **kwargs), pat) for q, pat in LABELLED]
        n = len(ranks)
        hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{name:32} {hit(1):6.0%} {hit(3):6.0%} {hit(5):6.0%} {mrr:6.2f}")
        misses[name] = [q for (q, _), r in zip(LABELLED, ranks) if not r]
    final = CONFIGS[-1][0]
    print(f"\nMissed in top 5 by '{final}': {misses[final] or 'none'}")

    print(f"\nThreshold calibration (best in-scope dense cosine, MIN_DENSE_COSINE = {config.MIN_DENSE_COSINE}):")
    in_scores = sorted(retriever.retrieve(q).best_cosine for q, _ in LABELLED)
    off = sorted(((retriever.retrieve(q).best_cosine, q) for q in OFF_TOPIC), reverse=True)
    print(f"  in-scope questions: min {in_scores[0]:.3f}, 10th pct {in_scores[len(in_scores) // 10]:.3f}, median {in_scores[len(in_scores) // 2]:.3f}")
    print(f"  off-topic questions: max {off[0][0]:.3f}  ({off[0][1]!r})")
    for score, q in off[:4]:
        print(f"      {score:.3f}  {q}")
    wrongly_refused = sum(s < config.MIN_DENSE_COSINE for s in in_scores)
    wrongly_accepted = sum(s >= config.MIN_DENSE_COSINE for s, _ in off)
    print(f"  at {config.MIN_DENSE_COSINE}: {wrongly_refused}/{len(in_scores)} in-scope refused, "
          f"{wrongly_accepted}/{len(off)} off-topic accepted")


if __name__ == "__main__":
    main()
