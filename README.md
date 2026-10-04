# PPFAS Mutual Fund Facts: a facts-only RAG assistant

A retrieval-augmented chatbot that answers **factual** questions about six **PPFAS (Parag Parikh)** mutual fund
schemes and Groww's mutual fund features, using public **Groww** pages. Every answer carries **one source link**
and a **"Last updated from sources"** date. It refuses investment advice and performance comparisons, and it
never accepts personal data.

> **Facts-only. No investment advice.** Answers come from public Groww pages about six PPFAS mutual fund schemes
> and Groww's help centre, as of the date shown with each answer. Nothing here is a recommendation to buy, sell
> or hold any fund. Past performance does not indicate future returns. Independent prototype, not affiliated
> with Groww or PPFAS Mutual Fund.

Built for Milestone 1 ("Mutual Fund FAQs, facts-only Q&A") of the [RAG assignment](https://nextleap.app/course/generative-ai-course). Product: **Groww**.

**Live app: https://ppfas-facts-assistant-25ej4ewnf9th28lhfj7pvr.streamlit.app/** (the first question after a
period of inactivity takes ~30 s while the app wakes up and loads its knowledge base).

**Coursebook: https://claude.ai/artifact/DD9c8cwE8c381xDfKrbkjB**: a teaching-style walkthrough of the whole
project (context, RAG concepts, phase-by-phase build log, code walkthrough, toolbox, limitations and next steps).

## Scope

| | |
|---|---|
| **AMC** | PPFAS Mutual Fund |
| **Schemes (6)** | Parag Parikh Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, Conservative Hybrid, Dynamic Asset Allocation (Direct Growth) |
| **Facts covered** | expense ratio, exit load, lock-in, minimum SIP/lumpsum, riskometer, benchmark, NAV, AUM, fund managers, top holdings, tax, stamp duty, registrar, published returns and risk ratios |
| **How-to / concepts** | 166 Groww help-centre mutual fund articles (statements/CAS, SIPs, redemptions, switching to direct, AutoPay, NFOs...) and 2 Groww blog explainers |
| **Sources** | [`sources.csv`](sources.csv): 174 Groww URLs with crawl date (core subset below) |

## Architecture

```
OFFLINE (mf_assistant/ingest, mf_assistant/index)                ONLINE (app.py → mf_assistant/answer)
crawl4ai crawl ─► clean ─► extract + validate ─┬─► SQLite (exact scheme facts)
  (manifest records URL + date)                └─► documents.jsonl ─► chunks ─► OpenAI embeddings ─► FAISS

message ─► PII guard (local) ─► advice/comparison net ─► decompose mixed questions ─► LLM router
        ├─ fact        ─► fixed SQL query            ─┐
        ├─ explanation ─► hybrid search (FAISS + BM25, RRF, scheme filter) ─► grounded LLM answer + number check
        └─ advice / comparison / off-topic / small talk ─► polite fixed reply
                                                      ─► output guard: ≤3 sentences, one link, date, no advice
```

- **Exact facts come from SQLite, not from the LLM**: a typed router picks one of 18 fact fields; fixed,
  parameterized queries return the value with its page URL and crawl date.
- **Explanations are grounded**: the LLM may only use retrieved passages, must say when they don't cover the
  question, names the passage it used (that becomes the citation), and every number it writes must appear in
  the passages.
- **Guardrails are layered**: local PII detection before any API call, deterministic advice/comparison patterns
  plus the LLM router (either one refusing is enough), and an output guard on every reply.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd RAG_Chatbot_PP_MF
uv sync                                   # app runtime + tests
cp .env.example .env                      # then put your OPENAI_API_KEY in .env
uv run streamlit run app.py               # opens http://localhost:8501
```

The repo includes the processed data and the search index, so the app runs without re-crawling.

**Rebuild the data** (optional; crawls Groww politely: robots.txt checked, 1-2 s between requests):

```bash
uv sync --group ingest && uv run crawl4ai-setup
uv run python -m mf_assistant.ingest.crawl_schemes        # 6 scheme pages
uv run python -m mf_assistant.ingest.clean
uv run python -m mf_assistant.ingest.extract_scheme       # facts + FAQs, validated
uv run python -m mf_assistant.ingest.build_sql
uv run python -m mf_assistant.ingest.crawl_knowledge      # help articles + blogs (frozen URL list)
uv run python -m mf_assistant.ingest.extract_knowledge
uv run python -m mf_assistant.ingest.build_documents      # documents.jsonl + sources.csv
uv run python -m mf_assistant.index.chunk
uv run python -m mf_assistant.index.build_index           # skipped automatically if nothing changed
```

`discover_knowledge` re-reads Groww's support sitemap; run it only when you want to change the URL list.

## Evaluation

Full report: [`EVALUATION.md`](EVALUATION.md) (`uv sync --group eval && uv run python -m mf_assistant.evaluation.run_all`).

| What | Result |
|---|---|
| Unit tests (PII, advice patterns, output format) | 76 pass |
| Retrieval, 32 labelled questions (hybrid + scheme filter) | hit@1 81%, hit@5 100%, MRR 0.89; wrong-fund passages in top 5: 3/65 → 0 with the filter |
| Routing, 39 questions incl. disguised advice | 39/39 handlers, 13/13 fact-field sets |
| Grounded generation, gpt-4o as judge | 15/16 answered, all faithful and correctly cited; 4/4 out-of-scope questions declined |
| **Held-out end-to-end** (21 new questions, never tuned on) | **19/21 (90%) on first run**; the 2 misses were fixed afterwards (21/21), so a fresh held-out set is needed to re-measure |
| RAGAS on held-out explanations (gpt-4o evaluator, 2 runs) | faithfulness 0.94-1.00, answer relevancy 0.79-0.81, context precision 0.94, context recall 1.00 (LLM-judged scores vary between runs) |

Component scores were measured on the same questions used while building, so treat them as optimistic; the
held-out figure is the more honest estimate.

## Known limitations

- **Sources are Groww pages only.** The brief asks for AMC/SEBI/AMFI pages; this prototype deliberately scopes
  to "Groww facts for a Groww chatbot". Switching to official sources would only change `mf_assistant/ingest`.
- **More sources than the brief's 15-25.** All 174 are listed in `sources.csv`; the core set is below.
- **Broken Groww pages.** 17 help articles listed in Groww's sitemap consistently return Groww's error screen
  (for example the capital-gains-report and CAS-password articles), so those questions get partial answers or
  "not found".
- **Freshness.** Data was crawled on the date shown in each answer. NAV, AUM and returns change daily; refresh by
  re-running the pipeline.
- **Performance figures are quoted, never compared.** Published returns and risk ratios are answered per fund
  with "past performance" wording; comparisons and recommendations are refused. Risk ratios come from data
  embedded in Groww's page and aren't visible in its text.
- **No chat memory** (`USE_CHAT_HISTORY = False`): each question is answered on its own, so follow-ups must name
  the fund. Memory code exists and can be switched on.
- **One link per answer**: an explanation covering two topics cites only one page.
- **PII detection** covers PAN, Aadhaar, phone, email, UPI, OTP/PIN/password values, account/folio/card numbers;
  names and addresses are not detected.
- LLM outputs can vary slightly between runs even at temperature 0.

## Core sources (subset of `sources.csv`)

| Page | URL |
|---|---|
| Parag Parikh Flexi Cap Fund | https://groww.in/mutual-funds/parag-parikh-long-term-value-fund-direct-growth |
| Parag Parikh ELSS Tax Saver Fund | https://groww.in/mutual-funds/parag-parikh-elss-tax-saver-fund-direct-growth |
| Parag Parikh Liquid Fund | https://groww.in/mutual-funds/parag-parikh-liquid-fund-direct-growth |
| Parag Parikh Arbitrage Fund | https://groww.in/mutual-funds/parag-parikh-arbitrage-fund-direct-growth |
| Parag Parikh Conservative Hybrid Fund | https://groww.in/mutual-funds/parag-parikh-conservative-hybrid-fund-direct-growth |
| Parag Parikh Dynamic Asset Allocation Fund | https://groww.in/mutual-funds/parag-parikh-dynamic-asset-allocation-fund-direct-growth |
| How do I generate my CAS from CAMS? | https://groww.in/help/mutual-funds/mf-track-and-switch/how-to-generate-my-consolidated-account-statement--cas--from-the-cams-website--60 |
| What is CAS and why do I need it? | https://groww.in/help/mutual-funds/mf-track-and-switch/what-is-cas--why-is-it-needed--99 |
| What is a mutual funds ELSS report? | https://groww.in/help/mutual-funds/order/how-to-download-tax-statement--for-elss--77 |
| What is exit load? | https://groww.in/help/mutual-funds/order/what-is-exit-load--71 |
| How do I withdraw or redeem from a mutual fund? | https://groww.in/help/mutual-funds/order/how-to-withdraw-redeem-4 |
| Which day's NAV applies when redeeming? | https://groww.in/help/mutual-funds/order/when-withdrawing-redeeming-which-days-nav-will-be-applicable-1 |
| How long until the redeemed amount reaches my bank? | https://groww.in/help/mutual-funds/order/how-long-will-it-take-for-my-redeem-amount-to-reflect-in-my-bank-account-groww-balance--16 |
| How can I start a SIP on Groww? | https://groww.in/help/mutual-funds/mf-sip/how-to-start-a-sip-on-groww |
| How can I cancel an ongoing SIP? | https://groww.in/help/mutual-funds/mf-sip/how-to-cancel-my-sip |
| How can I switch from regular to direct? | https://groww.in/help/mutual-funds/switch-to-direct/how-can-i-switch-from-regular-fund-to-direct-fund-on-groww--30 |
| What is a folio? | https://groww.in/help/mutual-funds/discoverable/what-is-a-folio--48 |
| Can I invest using a credit/debit card? | https://groww.in/help/mutual-funds/order/can-i-invest-using-a-credit-debit-card-4 |
| What are Mutual Funds? (blog) | https://groww.in/blog/what-are-mutual-funds |
| Things to know before investing in mutual funds (blog) | https://groww.in/blog/mutual-funds-things-you-should-know-as-a-beginner |

## Deliverables

| Brief deliverable | Where |
|---|---|
| Working prototype | [Live app on Streamlit Community Cloud](https://ppfas-facts-assistant-25ej4ewnf9th28lhfj7pvr.streamlit.app/) (`app.py`) + demo video |
| Source list | [`sources.csv`](sources.csv) |
| README | this file |
| Sample Q&A (5-10 queries) | [`SAMPLE_QA.md`](SAMPLE_QA.md), generated by the app pipeline |
| Disclaimer snippet | the quote at the top; shown in the app sidebar (`config.DISCLAIMER_LONG`) |

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (`.env` is git-ignored; the index in `data/index/` is committed).
2. On [share.streamlit.io](https://share.streamlit.io): **Create app** → pick the repo and branch → main file
   `app.py` → **Advanced settings**: Python 3.13, and under **Secrets** add `OPENAI_API_KEY = "..."`.
3. Deploy. Community Cloud installs from `uv.lock`; runtime dependencies are kept lean (the crawler, RAGAS and
   course tools are in optional dependency groups).

## Project structure

```
app.py                      Streamlit UI (entry point)
mf_assistant/
  config.py                 paths, models, thresholds, messages
  ingest/                   crawl → clean → extract (validated) → SQLite + documents.jsonl + sources.csv
  index/                    chunking, embeddings, FAISS index, search demo
  retrieval/                scheme resolver, hybrid search (FAISS + BM25 + RRF), retrieval eval
  answer/                   router, SQL facts, decomposition, grounded generation, pipeline, evals
  guardrails/               PII, advice/comparison policy, output format
  evaluation/               held-out set, RAGAS, sample Q&A, run_all report
tests/                      unit tests (pytest)
data/
  json_files/               scheme list, crawl manifest (URL + date per page), knowledge URL list
  raw/                      crawled HTML/Markdown (help-centre crawl is git-ignored)
  processed/                scheme facts, FAQs, knowledge pages, documents.jsonl
  structured_data/          structured.sqlite (exact facts)
  index/                    chunks.jsonl, FAISS index, index_info.json
docs/                       assignment brief
archive/                    earlier notebooks, notebook-era data, course material (not used by the app)
sources.csv  SAMPLE_QA.md  EVALUATION.md      deliverables
.streamlit/config.toml      theme (Groww colour tokens)
```
