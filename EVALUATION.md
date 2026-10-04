# Evaluation report

Generated 2026-10-04 18:00 by `python -m mf_assistant.evaluation.run_all`. Models: gpt-4o-mini (app), text-embedding-3-small (embeddings), gpt-4o (judge/RAGAS evaluator).

## Unit tests

*PII detection, advice/comparison patterns, output formatting (no API calls)* (10 s, ok)

```
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 5.67s
```

## Retrieval

*hit@k and MRR on 32 labelled questions: dense vs BM25 vs hybrid, with/without scheme filter; 'not found' threshold calibration* (97 s, ok)

```
32 labelled questions
configuration                     hit@1  hit@3  hit@5    MRR
dense only (no scheme filter)       78%    97%   100%   0.87
bm25 only (no scheme filter)        69%    78%    91%   0.76
hybrid RRF (no scheme filter)       81%    97%   100%   0.89
dense + scheme filter               78%    97%   100%   0.87
bm25 + filter + name strip          56%    69%    81%   0.64
hybrid + filter + name strip        84%    94%    94%   0.89
bm25 + scheme filter                69%    78%    91%   0.76
hybrid + scheme filter              81%    94%   100%   0.89
Missed in top 5 by 'hybrid + scheme filter': none
Threshold calibration (best in-scope dense cosine, MIN_DENSE_COSINE = 0.3):
  in-scope questions: min 0.449, 10th pct 0.531, median 0.595
  off-topic questions: max 0.422  ('how to open a fixed deposit in SBI')
      0.422  how to open a fixed deposit in SBI
      0.387  how to file GST returns
      0.251  how do I renew my passport
      0.219  weather in Delhi tomorrow
  at 0.3: 0/32 in-scope refused, 2/10 off-topic accepted
```

## Routing

*39 labelled questions: handler + fact-field accuracy (incl. disguised advice, comparisons, off-topic)* (44 s, ok)

```
Handler accuracy: 39/39 = 100%
  ask_scheme         1/1
  off_topic          4/4
  refuse_advice      7/7
  refuse_comparison  3/3
  retrieval          8/8
  small_talk         3/3
  sql_fact           13/13
Fact-field accuracy (routed FACT questions): 13/13
No mistakes.
```

## Decomposition

*12 messages: mixed questions answered in full, single requests kept whole, unsafe parts refused* (51 s, ok)

```
✓ 'What is the exit load of the ELSS fund and how do I download my CAS?'
    got ['sql_fact', 'retrieval']
    The exit load of Parag Parikh ELSS Tax Saver Fund Direct Growth is: Nil.
    Source: https://groww.in/mutual-funds/parag-parikh-elss-tax-saver-fund-direct-growth
    Last updated from sources: 2026-10-04
    To download your Consolidated Account Statement (CAS), visit the official CAMS website, click on 'MF Investor', select 'Statement', and choose 'CAS - CAMS+KFintech'. Enter the required details, including your email ID and PAN number, then submit the information to receive your CAS via email in PDF format.
    Source: https://groww.in/help/mutual-funds/mf-track-and-switch/how-to-generate-my-consolidated-account-statement--cas--from-the-cams-website--60
    Last updated from sources: 2026-10-04
✓ 'What is the lock-in of the tax saver fund and what is a folio?'
    got ['sql_fact', 'retrieval']
    Parag Parikh ELSS Tax Saver Fund Direct Growth has a lock-in period of 3 years.
    Source: https://groww.in/mutual-funds/parag-parikh-elss-tax-saver-fund-direct-growth
    Last updated from sources: 2026-10-04
    A folio is a unique number issued by the mutual fund company, also known as the Asset Management Company (AMC), when you make your first mutual fund investment. It serves as your account number with that specific AMC and remains constant for all future investments within the same scheme.
    Source: https://groww.in/help/mutual-funds/discoverable/what-is-a-folio--48
    Last updated from sources: 2026-10-04
✓ 'What is NAV? Also, who manages the liquid fund?'
    got ['retrieval', 'sql_fact']
    NAV stands for Net Asset Value, which is the value per share of a mutual fund or an exchange-traded fund (ETF) on a specific date or time. It represents the true value of a mutual fund scheme unit after subtracting any liabilities, calculated using the formula: NAV per unit = (Total Assets - Total Liabilities) / Total Number of Outstanding Units.
    Source: https://groww.in/blog/what-are-mutual-funds
    Last updated from sources: 2026-10-04
    Parag Parikh Liquid Fund Direct Growth is managed by Mansi Kariya (since 2023-12-22), Aishwarya Dhar (since 2025-09-01), Tejas Soman (since 2025-09-01).
    Source: https://groww.in/mutual-funds/parag-parikh-liquid-fund-direct-growth
    Last updated from sources: 2026-10-04
✓ 'minimum SIP for the arbitrage fund and how do I start a SIP on Groww?'
    got ['sql_fact', 'retrieval']
    For Parag Parikh Arbitrage Fund Direct Growth, the minimum SIP is ₹1,000, the minimum first (lumpsum) investment is ₹1,000, and additional investments start at ₹1,000.
    Source: https://groww.in/mutual-funds/parag-parikh-arbitrage-fund-direct-growth
    Last updated from sources: 2026-10-04
    To start a SIP on Groww, go to the 'Mutual Funds' section, select a fund, and click 'Start SIP.' Enter the amount, select the deduction date, and complete the payment via UPI or Netbanking, ensuring to set up AutoPay for future installments.
    Source: https://groww.in/help/mutual-funds/mf-sip/how-to-start-a-sip-on-groww
    Last updated from sources: 2026-10-04
✓ 'What is an NFO and what is a step-up SIP?'
    got ['retrieval']
✓ 'expense ratio and benchmark of the flexi cap fund'
    got ['sql_fact']
✓ 'exit load of ELSS and flexi cap'
    got ['sql_fact']
✓ 'How do I redeem and when will the money reach my bank?'
    got ['retrieval']
✓ 'thanks! what is the NAV of the liquid fund?'
    got ['sql_fact']
✓ 'What is the expense ratio of ELSS and should I buy it?'
    got ['refuse_advice']
✓ 'Who manages the flexi cap fund and which fund gave the highest returns?'
    got ['refuse_comparison']
✓ 'How do I cancel my SIP?'
    got ['retrieval']
12/12 messages produced the expected parts; decomposer LLM called for 10/12 (gate)
```

## Grounded generation (LLM judge)

*16 answerable + 4 unanswerable questions; gpt-4o judges faithfulness* (53 s, ok)

```
ANSWERABLE
  ✓ 'How can I stop my SIP?'
  ✓ 'what is an exit load?'
  ✓ "which day's NAV applies when I redeem?"
  ✓ 'How do I generate my CAS from CAMS?'
  ✗ not answered: 'How do I download my capital gains statement?'
  ✓ 'what is a folio?'
  ✓ 'what is a step-up SIP?'
  ✓ 'how long does a redemption take to reach my bank?'
  ✓ 'Can I pay for mutual funds with a credit card?'
  ✓ 'What are the AutoPay charges?'
  ✓ 'how do I switch from a regular plan to a direct plan on Groww?'
  ✓ 'why did my SIP installment fail?'
  ✓ 'what is an NFO?'
  ✓ 'can I skip an SIP installment?'
  ✓ 'what does AUM mean?'
  ✓ 'are mutual funds safe for beginners?'
UNANSWERABLE
  ✓ declined: 'how do I add a nominee to my mutual funds on Groww?'
  ✓ declined: 'what is the penalty for missing a PPF deposit?'
  ✓ declined: 'how do I apply for an IPO on Groww?'
  ✓ declined: 'what is the GST rate on brokerage for stocks?'
Answerable (16): answered 15/16; of those answered: cited right 15/15, faithful 15/15, relevant 15/15, ≤3 sentences 15/15, no advice 15/15
Unanswerable (4): declined 4/4
```

## Held-out end-to-end

*21 new questions through the full app pipeline (facts, explanations, refusals)* (42 s, ok)

```
FACT
  ✓ "What's the TER of the PPFAS liquid fund?" [sql_fact]
  ✓ 'Does the parag parikh arbitrage fund charge anything if I exit within 2 weeks?' [sql_fact]
  ✓ 'Benchmark index for PPFAS dynamic asset allocation?' [sql_fact]
  ✓ "what's the risk level of the conservative hybrid fund" [sql_fact]
  ✓ 'When was the PPFAS ELSS fund launched?' [sql_fact]
  ✓ 'Who are the fund managers of the PPFAS conservative hybrid fund?' [sql_fact]
  ✓ 'minimum lumpsum amount for parag parikh flexi cap' [sql_fact]
EXPLAIN
  ✓ 'Can I redeem my mutual funds on a Sunday?' [retrieval]
  ✓ 'How do I pause my SIP for a month?' [retrieval]
  ✓ 'Why do I have more than one folio?' [retrieval]
  ✓ 'If I switch from regular to direct, do my SIPs stop?' [retrieval]
  ✓ 'Is there any fee for cancelling a SIP on Groww?' [retrieval]
  ✓ 'How can I combine my folios into one?' [retrieval]
  ✓ 'I put money in an NFO, when will it show up in my account?' [retrieval]
  ✓ 'Can I run two SIPs in the same fund?' [retrieval]
  ✓ 'What does expense ratio mean?' [retrieval]
REFUSE
  ✓ 'Is PPFAS flexi cap good for long term wealth creation?' [refuse_advice]
  ✓ 'Did the ELSS fund do better than the Nifty 500?' [refuse_comparison]
  ✓ 'My email is test.user@example.com, can you update my KYC?' [pii]
  ✓ "What's the weather like in Pune today?" [off_topic]
  ✓ 'Tell me which PPFAS fund will give the best returns next year' [refuse_comparison]
Held-out results:
  fact     7/7
  explain  9/9
  refuse   5/5
  overall  21/21 = 100%
Saved 9 answered explanation samples for RAGAS → ragas_samples.json
```

## RAGAS

*faithfulness, answer relevancy, context precision, context recall on the held-out explanations* (257 s, ok)

```
RAGAS on 9 held-out explanation answers (evaluator: gpt-4o)
question                                                   faithfulness answer_relev llm_context_ context_reca
Can I redeem my mutual funds on a Sunday?                          1.00         0.74         0.48         1.00
How do I pause my SIP for a month?                                 1.00         0.70         1.00         1.00
Why do I have more than one folio?                                 1.00         0.90         1.00         1.00
If I switch from regular to direct, do my SIPs stop?               1.00         0.72         1.00         1.00
Is there any fee for cancelling a SIP on Groww?                    1.00         0.92         1.00         1.00
How can I combine my folios into one?                              1.00         0.79         1.00         1.00
I put money in an NFO, when will it show up in my account?         1.00         0.71         1.00         1.00
Can I run two SIPs in the same fund?                               1.00         0.92         1.00         1.00
What does expense ratio mean?                                      0.50         0.73         1.00         1.00
MEAN                                                               0.94         0.79         0.94         1.00
Per-sample scores → E:\ML Projects\RAG_Chatbot_PP_MF\mf_assistant\evaluation\results\ragas_scores.csv
```
