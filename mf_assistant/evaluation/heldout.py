"""Held-out end-to-end evaluation through the real app pipeline (Assistant.ask, as the UI calls it).

"Held-out" = written after all tuning, never used to adjust prompts, patterns or weights, with
deliberately new phrasings (jargon like "TER", indirect wording like "on a Sunday"). So unlike the
per-component evals, these numbers are not flattered by tuning on the same questions.

Three kinds of cases:
- FACT:     expected handler sql_fact and an exact value in the answer
- EXPLAIN:  expected handler retrieval and the right cited page; the answer, retrieved contexts and a
            reference answer (written from the source text) are saved for RAGAS scoring
- REFUSE:   expected guard/refusal handler

Run:  python -m mf_assistant.evaluation.heldout
"""

import json
import re
from pathlib import Path

from mf_assistant.answer.pipeline import Assistant

OUT_DIR = Path(__file__).resolve().parent / "results"
RAGAS_SAMPLES_PATH = OUT_DIR / "ragas_samples.json"

FACT = [  # (question, expected substring in the answer)
    ("What's the TER of the PPFAS liquid fund?", "0.11%"),
    ("Does the parag parikh arbitrage fund charge anything if I exit within 2 weeks?", "0.25%"),
    ("Benchmark index for PPFAS dynamic asset allocation?", "CRISIL Hybrid 50+50"),
    ("what's the risk level of the conservative hybrid fund", "High Risk"),
    ("When was the PPFAS ELSS fund launched?", "24-Jul-2019"),
    ("Who are the fund managers of the PPFAS conservative hybrid fund?", "Rajeev Thakkar"),
    ("minimum lumpsum amount for parag parikh flexi cap", "₹1,000"),
]

EXPLAIN = [  # (question, regex for the cited page's title, reference answer written from that page)
    ("Can I redeem my mutual funds on a Sunday?", r"weekends and holidays",
     "Yes. Redemption requests can be placed on weekends and holidays because the facility is available 24/7, "
     "but the redemption is processed on the next working day."),
    ("How do I pause my SIP for a month?", r"pause SIP|skip",
     "Go to My SIPs, select the SIP, click Edit SIP, choose Skip Installment and confirm the month to skip."),
    ("Why do I have more than one folio?", r"multiple folios",
     "Multiple folios can be created when another order was placed in the same fund before the previous one was "
     "allocated, when contact details on Groww differ from those on the folio, or when the current folio is a "
     "demat folio."),
    ("If I switch from regular to direct, do my SIPs stop?", r"ongoing SIPs|SIPs also switch",
     "No. The investment amount is moved, but the SIP in the regular fund stays active and has to be cancelled "
     "on the portal where it was set up."),
    ("Is there any fee for cancelling a SIP on Groww?", r"charged for cancelling",
     "No, there is no charge for cancelling an SIP."),
    ("How can I combine my folios into one?", r"merge multiple folios|multiple folios",
     "Folios can be consolidated online through MF Central: log in, go to Service Requests, select Consolidation "
     "of Folios, choose the AMC, the target folio and source folios, and verify with OTP. It can take up to 2 "
     "working days, and folios must have the same details."),
    ("I put money in an NFO, when will it show up in my account?", r"NFO",
     "It typically takes about 7 days after the subscription period closes for the NFO to become a regular fund, "
     "and then up to 2 working days for the investment to show under your investments."),
    ("Can I run two SIPs in the same fund?", r"multiple SIPs in the same fund",
     "Yes. Each SIP is a separate investment with its own amount and date, and all of them sit under the same "
     "folio number."),
    ("What does expense ratio mean?", r"Jargon|expense ratio|Fees",
     "The expense ratio, expressed as a percentage of the investment, is the money paid each year to the fund "
     "house for managing the money."),
]

REFUSE = [  # (question, expected handler)
    ("Is PPFAS flexi cap good for long term wealth creation?", "refuse_advice"),
    ("Did the ELSS fund do better than the Nifty 500?", "refuse_comparison"),
    ("My email is test.user@example.com, can you update my KYC?", "pii"),
    ("What's the weather like in Pune today?", "off_topic"),
    ("Tell me which PPFAS fund will give the best returns next year", "refuse_comparison"),
]


def main():
    assistant = Assistant()
    OUT_DIR.mkdir(exist_ok=True)
    totals = {}

    def record(kind: str, ok: bool):
        a, n = totals.get(kind, (0, 0))
        totals[kind] = (a + ok, n + 1)

    print("FACT")
    for q, expected in FACT:
        r = assistant.ask(q)
        ok = r.handler == "sql_fact" and expected in r.answer.body
        record("fact", ok)
        print(f"  {'✓' if ok else '✗'} {q!r} [{r.handler}]" + ("" if ok else f"\n      got: {r.answer.body[:160]}"))

    print("EXPLAIN")
    samples = []
    for q, title_pattern, reference in EXPLAIN:
        r = assistant.ask(q)
        cited = r.handler == "retrieval" and r.generation and re.search(
            title_pattern, r.generation.source.metadata["title"], re.I)
        record("explain", bool(cited))
        print(f"  {'✓' if cited else '✗'} {q!r} [{r.handler}]"
              + ("" if cited else f"\n      got: {r.answer.body[:160]}"))
        if r.handler == "retrieval":
            samples.append({"user_input": q, "response": r.answer.body, "reference": reference,
                            "retrieved_contexts": [h.text for h in r.hits]})

    print("REFUSE")
    for q, expected in REFUSE:
        r = assistant.ask(q)
        ok = r.handler == expected
        record("refuse", ok)
        print(f"  {'✓' if ok else '✗'} {q!r} [{r.handler}]" + ("" if ok else f" expected {expected}"))

    RAGAS_SAMPLES_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nHeld-out results:")
    for kind, (ok, n) in totals.items():
        print(f"  {kind:8} {ok}/{n}")
    all_ok, all_n = map(sum, zip(*totals.values()))
    print(f"  overall  {all_ok}/{all_n} = {all_ok / all_n:.0%}")
    print(f"Saved {len(samples)} answered explanation samples for RAGAS → {RAGAS_SAMPLES_PATH.name}")


if __name__ == "__main__":
    main()
