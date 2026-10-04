"""Advice / comparison guards: deterministic safety nets around the LLM.

- Input: `asks_for_advice()` runs alongside the router. If either the router or
  these patterns say "advice", the question is refused (OR = the safe side wins).
  It catches the clear phrasings even if the LLM ever misclassifies one.
- Output: `gives_advice()` scans any generated answer for recommendation or
  comparison language before it is shown (Phase 6 generation, plus anything else).

Patterns are deliberately narrow, so factual questions ("how much is the minimum
SIP?", "can I redeem on weekends?") are not caught. False positives are tested
against the router's labelled questions.
"""

import re

_ADVICE_IN = [
    r"\bshould (i|we)\b",
    r"\bworth (it|buying|investing|holding)\b",
    r"\b(recommend|suggest)\w*\b",
    r"\bhow much (should|do|must) (i|we)\b",
    r"\b(good|best|better|right|ideal|suitable|safe) (fund|scheme|option|choice|investment|time|place) (for|to)\b",
    r"\bbest (ppfas |parag parikh )?(fund|scheme|option)\b",
    r"\bwhich (one|fund|scheme)\b.*\b(better|best|choose|pick|go for)\b",
    r"\bwhich is (better|best)\b",
    r"\b(is|are) (it|this|they|the \w+( \w+)? fund)( a)? (good|bad|safe|worth)\b",
    r"\bbetter than\b",
]

_COMPARISON_IN = [
    r"\bcompar\w*\b.*\b(returns?|performance|performed)\b",
    r"\b(highest|lowest|top|best|worst)[- ](returns?|performance|performing)\b",
    r"\b(outperform\w*|underperform\w*)\b",
    r"\bbeat(s|en)?\b.*\b(benchmark|index|category|market)\b",
    r"\b(do|does|did|done|perform\w*|return\w*) (better|worse) than\b",
]

_ADVICE_OUT = [
    r"\b(you|i|we) (should|must|could|might want to) (buy|sell|invest|redeem|hold|switch|choose|pick|go for|consider)\b",
    r"\b(i|we) (recommend|suggest|advise)\b",
    r"\b(good|great|better|best|ideal|suitable|safe|smart) (investment|choice|option|fund|bet)\b",
    r"\bbetter than\b",
    r"\boutperform\w*\b",
]

_IN = [re.compile(p, re.I) for p in _ADVICE_IN]
_CMP = [re.compile(p, re.I) for p in _COMPARISON_IN]
_OUT = [re.compile(p, re.I) for p in _ADVICE_OUT]


def refusal_kind(question: str) -> str | None:
    """"refuse_comparison" | "refuse_advice" | None. Comparison is checked first: it's the more specific reply."""
    if any(p.search(question) for p in _CMP):
        return "refuse_comparison"
    if any(p.search(question) for p in _IN):
        return "refuse_advice"
    return None


def asks_for_advice(question: str) -> bool:
    """True for advice or performance-comparison requests."""
    return refusal_kind(question) is not None


def gives_advice(answer: str) -> bool:
    return any(p.search(answer) for p in _OUT)
