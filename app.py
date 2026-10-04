"""Streamlit UI for the Facts-Only MF Assistant.

Run:  streamlit run app.py

Streamlit re-runs this whole script on every interaction (a click, a message). So:
- the heavy Assistant (FAISS index, BM25, LLM clients) is built once per server process
  with @st.cache_resource, not on every rerun;
- the chat transcript lives in st.session_state, which is per browser tab, in memory only.

Privacy: a message refused for containing personal data is never shown back or kept in the
transcript; only a placeholder is. Nothing is written to disk.
"""

import csv
from urllib.parse import urlparse

import streamlit as st

from mf_assistant import config
from mf_assistant.guardrails.pii import detect_pii

st.set_page_config(page_title="PPFAS Facts Assistant", page_icon="📊", layout="centered")

# Small additions to the theme in .streamlit/config.toml, using Groww's design tokens
st.markdown("""
<style>
  .facts-note { display:inline-block; background:#E9FAF3; color:#04B488; font-weight:600;
                padding:4px 12px; border-radius:999px; font-size:0.85rem; margin:4px 0 12px; }
  .welcome    { color:#7C7E8C; margin-bottom:4px; }
  .src        { display:inline-block; background:#EEF0FF; color:#5367FF !important; padding:2px 10px;
                border-radius:999px; font-size:0.8rem; text-decoration:none; margin-top:4px; }
  .updated    { color:#A1A3AD; font-size:0.75rem; margin-top:2px; }
  .refusal    { border-left:3px solid #ED5533; padding-left:10px; }
</style>
""", unsafe_allow_html=True)

REFUSALS = {"pii", "refuse_advice", "refuse_comparison", "off_topic"}
PII_PLACEHOLDER = "🔒 Message hidden: it contained personal information."


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_assistant():
    # Imported here, not at the top: LangChain/FAISS/NLTK take ~30 s to import cold, and a
    # top-level import would leave the page blank until done. This way the page draws at once.
    from mf_assistant.answer.pipeline import Assistant
    return Assistant()


@st.cache_data
def source_stats() -> dict:
    with config.SOURCES_CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_type = {}
    for r in rows:
        by_type[r["source_type"]] = by_type.get(r["source_type"], 0) + 1
    return {"total": len(rows), "by_type": by_type, "updated": max(r["scraped_at"] for r in rows)}


def to_parts(reply) -> list[dict]:
    """Render-ready pieces of a reply (one per sub-question); plain data for session_state."""
    return [
        {"body": p.answer.body, "url": p.answer.citation_url,
         "updated": p.answer.last_updated, "refusal": p.handler in REFUSALS}
        for p in (reply.parts or [reply])
    ]


@st.cache_data
def source_titles() -> dict[str, str]:
    with config.SOURCES_CSV_PATH.open(encoding="utf-8") as f:
        titles = {r["url"]: r["title"] for r in csv.DictReader(f)}
    # Links used by fixed replies (refusals, clarifications) aren't crawled sources
    titles.setdefault(config.HELP_CENTRE_URL, "Mutual funds help centre")
    titles.setdefault(config.AMC_SOURCE_URL, "PPFAS Mutual Fund schemes")
    return titles


def short_link(url: str) -> str:
    """Chip label: the page's real title (URL slugs can be stale, e.g. Flexi Cap's old 'long term value' name)."""
    title = source_titles().get(url)
    if title:
        return f"Groww · {title[:60]}"
    u = urlparse(url)
    path = u.path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")[:48]
    return f"{u.netloc} › {path}" if path else u.netloc


def render_parts(parts: list[dict]):
    for i, p in enumerate(parts):
        if i:
            st.divider()
        if p["refusal"]:
            st.markdown(f'<div class="refusal">{p["body"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(p["body"].replace("$", r"\$"))  # "$" would otherwise start LaTeX
        st.markdown(f'<a class="src" href="{p["url"]}" target="_blank">Source: {short_link(p["url"])}</a>',
                    unsafe_allow_html=True)
        if p["updated"]:
            st.markdown(f'<div class="updated">Last updated from sources: {p["updated"]}</div>',
                        unsafe_allow_html=True)


# ---------- sidebar ----------
with st.sidebar:
    st.subheader("About")
    stats = source_stats()
    st.markdown(
        f"Answers questions about **6 PPFAS schemes**: Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, "
        f"Conservative Hybrid and Dynamic Asset Allocation.\n\n"
        f"**Sources:** {stats['total']} public Groww pages ({stats['by_type'].get('scheme_page', 0)} scheme pages, "
        f"{stats['by_type'].get('help_faq', 0)} help-centre articles, {stats['by_type'].get('blog', 0)} blog posts), "
        f"last updated {stats['updated']}."
    )
    st.caption(config.DISCLAIMER_LONG)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- header ----------
st.title("PPFAS Mutual Fund Facts")
st.markdown('<div class="welcome">Hi! Ask me factual questions about PPFAS mutual fund schemes: expense ratios, '
            'exit loads, lock-in periods, minimum SIPs, benchmarks, riskometers, or how to download your '
            'statements. Every answer links to its source.</div>', unsafe_allow_html=True)
st.markdown(f'<span class="facts-note">{config.DISCLAIMER}</span>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = None
if not st.session_state.messages:
    st.caption("Try an example:")
    for i, q in enumerate(config.EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example-{i}", use_container_width=True):
            prompt = q

# ---------- transcript ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_parts(m["parts"])

typed = st.chat_input("Ask a factual question about a PPFAS fund…")
prompt = typed or prompt

# ---------- new message ----------
if prompt:
    # Local check, instant: decides whether the question may be echoed back at all
    shown_question = PII_PLACEHOLDER if detect_pii(prompt).found else prompt
    with st.chat_message("user"):
        st.markdown(shown_question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking it up…"):
                reply = get_assistant().ask(prompt)
            parts = to_parts(reply)
        except Exception:
            parts = [{"body": "Sorry, something went wrong while answering. Please try again in a moment.",
                      "url": config.HELP_CENTRE_URL, "updated": None, "refusal": True}]
        render_parts(parts)
    st.session_state.messages += [{"role": "user", "content": shown_question},
                                  {"role": "assistant", "parts": parts}]
    st.rerun()  # redraw so the question appears above its answer and the examples disappear
