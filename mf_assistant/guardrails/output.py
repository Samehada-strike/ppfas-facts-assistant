"""Output guard: every reply leaves in the same, brief-compliant shape.

    <answer, at most MAX_ANSWER_SENTENCES sentences, no URLs of its own>
    Source: <exactly one link>
    Last updated from sources: <crawl date>      (only for answers built from sources)

- URLs inside the text are removed, so the one citation is the only link.
- The sentence limit is enforced in code, not left to the prompt. It applies to
  prose; a bulleted list (one fact per scheme) has its own cap, MAX_LIST_ITEMS,
  so "expense ratio of all funds" isn't cut to three funds.
- Advice or comparison language in the text replaces the answer with a safe
  refusal: the last line of defence for generated answers.
"""

import re
from dataclasses import dataclass

from mf_assistant import config
from mf_assistant.guardrails.policy import gives_advice

SAFE_FALLBACK = ("I can only share factual information about PPFAS mutual fund schemes, "
                 "not recommendations or comparisons.")

# Split after . ! ? followed by space/newline — but not inside numbers (0.65%), after "Rs." / "No.",
# or after a step number ("download it: 1. Select ...")
_SENTENCE_END = re.compile(r"(?<!\bRs\.)(?<!\bNo\.)(?<![\s:]\d\.)(?<![\s:]\d\d\.)(?<=[.!?])\s+(?=[A-Z0-9₹\"'(])")


@dataclass
class FinalAnswer:
    text: str           # what the user sees, footer included
    body: str           # the answer without footer
    citation_url: str
    last_updated: str | None
    blocked: bool = False  # True if the output guard replaced the answer


def split_sentences(text: str) -> list[str]:
    pieces = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[-*•]\s", line):
            pieces.append(line)          # a bullet counts as one sentence
        else:
            pieces += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return pieces


def _is_bullet(s: str) -> bool:
    return s.startswith(("- ", "* ", "• "))


def limit_sentences(text: str, n: int = config.MAX_ANSWER_SENTENCES) -> str:
    sentences = split_sentences(text)
    prose = [s for s in sentences if not _is_bullet(s)][:n]
    bullets = [s for s in sentences if _is_bullet(s)][:config.MAX_LIST_ITEMS]
    return "\n".join(filter(None, [" ".join(prose), "\n".join(bullets)]))


def finalize(body: str, citation_url: str, last_updated: str | None) -> FinalAnswer:
    blocked = gives_advice(body)
    if blocked:
        body, last_updated = SAFE_FALLBACK, None
        citation_url = config.EDUCATION_URL
    # The citation is the only link: drop URLs (not the punctuation after them) and emptied brackets
    body = re.sub(r"https?://[^\s)\]]+[^\s)\].,;:!?]|www\.[^\s)\]]+[^\s)\].,;:!?]", "", body)
    body = re.sub(r"\(\s*(?:see|at|via|link)?\s*\)", "", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+([.,;:!?])", r"\1", body).strip()
    body = limit_sentences(body)
    footer = f"Source: {citation_url}"
    if last_updated:
        footer += f"\nLast updated from sources: {last_updated}"
    return FinalAnswer(f"{body}\n\n{footer}", body, citation_url, last_updated, blocked)
