"""Measure routing accuracy on labelled questions, including deliberately tricky ones.

Each case: (question, expected handler, expected fact fields or None to skip the field check).
Reports overall handler accuracy, accuracy per expected handler, field accuracy for FACT
questions, and every mistake.

Run:  python -m mf_assistant.answer.eval_router
"""

from collections import defaultdict

from mf_assistant.answer.router import Router

CASES = [
    # plain facts
    ("What is the expense ratio of PPFAS ELSS?", "sql_fact", {"expense_ratio"}),
    ("exit load of parag parikh flexi cap", "sql_fact", {"exit_load"}),
    ("minimum SIP for the liquid fund", "sql_fact", {"min_investment"}),
    ("Is there a lock-in for the tax saver fund?", "sql_fact", {"lock_in"}),
    ("who manages the arbitrage fund?", "sql_fact", {"fund_managers"}),
    ("benchmark and riskometer of the conservative hybrid fund", "sql_fact", {"benchmark", "riskometer"}),
    ("NAV of the dynamic asset allocation fund", "sql_fact", {"nav"}),
    ("What are the top holdings of PPFAS flexi cap?", "sql_fact", {"top_holdings"}),
    ("3 year return of the ELSS fund", "sql_fact", {"returns"}),
    ("sharpe ratio of parag parikh liquid fund", "sql_fact", {"risk_ratios"}),
    ("how risky is the arbitrage fund", "sql_fact", {"riskometer"}),
    ("expense ratio of all PPFAS funds", "sql_fact", {"expense_ratio"}),
    ("expense ratio of ELSS and flexi cap", "sql_fact", {"expense_ratio"}),  # two facts listed, not a performance comparison
    # fact asked, but no fund named → ask which
    ("what is the minimum SIP amount for this fund?", "ask_scheme", None),
    # concepts and how-tos → retrieval
    ("what is an exit load?", "retrieval", None),
    ("How do I download my capital gains statement?", "retrieval", None),
    ("how to cancel my SIP on Groww", "retrieval", None),
    ("what is a folio number", "retrieval", None),
    ("why did my SIP installment fail?", "retrieval", None),
    ("what does riskometer mean?", "retrieval", None),
    ("how can I switch from regular to direct plan", "retrieval", None),
    ("what is ELSS?", "retrieval", None),
    # advice (including disguised)
    ("Should I invest in the flexi cap fund?", "refuse_advice", None),
    ("Which is better, ELSS or flexi cap?", "refuse_advice", None),
    ("Is the liquid fund a good place for my emergency money?", "refuse_advice", None),
    ("I am 25, how much should I put in the ELSS fund every month?", "refuse_advice", None),
    ("Should I redeem my arbitrage fund now that markets are down?", "refuse_advice", None),
    ("What is the expense ratio of ELSS and is it worth buying?", "refuse_advice", None),
    ("best PPFAS fund for tax saving", "refuse_advice", None),
    # performance comparisons
    ("Compare the 5 year returns of ELSS and flexi cap", "refuse_comparison", None),
    ("Has the flexi cap fund beaten its benchmark?", "refuse_comparison", None),
    ("which PPFAS fund gave the highest returns last year?", "refuse_comparison", None),
    # off-topic (incl. finance-adjacent)
    ("best pizza place in Mumbai", "off_topic", None),
    ("how to open a fixed deposit in SBI", "off_topic", None),
    ("what is the price of bitcoin today?", "off_topic", None),
    ("write me a poem about the stock market", "off_topic", None),
    # small talk
    ("hi there!", "small_talk", None),
    ("thanks, that helped", "small_talk", None),
    ("what can you do?", "small_talk", None),
]


def main():
    router = Router()
    correct, field_ok, field_total = 0, 0, 0
    per_handler = defaultdict(lambda: [0, 0])
    mistakes = []
    for question, expected, fields in CASES:
        d = router.route(question)
        ok = d.handler == expected
        correct += ok
        per_handler[expected][0] += ok
        per_handler[expected][1] += 1
        if fields is not None and d.handler == "sql_fact":
            field_total += 1
            field_ok += set(d.fact_fields) == fields
            if set(d.fact_fields) != fields:
                mistakes.append(f"FIELDS  {question!r}: got {d.fact_fields}, expected {sorted(fields)}")
        if not ok:
            mistakes.append(f"ROUTE   {question!r}: got {d.handler} ({d.intent}), expected {expected}")

    print(f"Handler accuracy: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")
    for handler, (ok, n) in sorted(per_handler.items()):
        print(f"  {handler:18} {ok}/{n}")
    print(f"Fact-field accuracy (routed FACT questions): {field_ok}/{field_total}")
    print("\nMistakes:" if mistakes else "\nNo mistakes.")
    for m in mistakes:
        print("  " + m)


if __name__ == "__main__":
    main()
