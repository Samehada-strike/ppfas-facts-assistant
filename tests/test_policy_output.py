import pytest

from mf_assistant.answer.eval_router import CASES
from mf_assistant.guardrails.output import finalize, split_sentences
from mf_assistant.guardrails.policy import asks_for_advice, gives_advice

ADVICE_QUESTIONS = [q for q, handler, _ in CASES if handler in ("refuse_advice", "refuse_comparison")]
NON_ADVICE_QUESTIONS = [q for q, handler, _ in CASES if handler not in ("refuse_advice", "refuse_comparison")]


@pytest.mark.parametrize("question", NON_ADVICE_QUESTIONS)
def test_advice_net_has_no_false_positives(question):
    # The net may miss subtle advice (the router catches those) but must never block a factual question
    assert not asks_for_advice(question), question


def test_advice_net_catches_most_clear_advice():
    caught = [q for q in ADVICE_QUESTIONS if asks_for_advice(q)]
    assert len(caught) >= len(ADVICE_QUESTIONS) * 0.6, f"caught only {caught}"


@pytest.mark.parametrize("text", [
    "You should invest in the ELSS fund for tax saving.",
    "I recommend the Flexi Cap fund.",
    "The ELSS fund is a good investment for beginners.",
    "Flexi Cap has done better than ELSS.",
    "It has outperformed its benchmark.",
])
def test_output_guard_flags_advice(text):
    assert gives_advice(text)


@pytest.mark.parametrize("text", [
    "The expense ratio of Parag Parikh ELSS Tax Saver Fund Direct Growth is 0.65%.",
    "You can redeem from the Mutual Funds section of the Groww app.",
    "Annualised returns as published on Groww: 1Y -10.4%, 3Y +7.7%.",
])
def test_output_guard_allows_facts(text):
    assert not gives_advice(text)


def test_finalize_shape():
    body = ("First fact is 0.65%. Second fact (see https://example.com). Third fact. Fourth fact.")
    out = finalize(body, "https://groww.in/x", "2026-10-04")
    assert out.body.count(".") >= 3 and "Fourth" not in out.body      # max 3 sentences
    assert "example.com" not in out.text                               # only one link
    assert out.text.count("https://") == 1
    assert out.text.endswith("Last updated from sources: 2026-10-04")


def test_finalize_blocks_advice():
    out = finalize("You should buy this fund now.", "https://groww.in/x", "2026-10-04")
    assert out.blocked and "recommendations" in out.body


def test_sentence_split_keeps_decimals_and_bullets():
    assert len(split_sentences("NAV is ₹30.43. Exit load is Nil.")) == 2
    assert len(split_sentences("- Flexi Cap: 0.69%\n- ELSS: 0.65%")) == 2


def test_sentence_split_keeps_step_numbers_and_abbreviations():
    steps = "Here's how: 1. Select Reports. 2. Choose the year. Done."
    assert split_sentences(steps) == ["Here's how: 1. Select Reports.", "2. Choose the year.", "Done."]
    assert len(split_sentences("Limit is Rs. 1 lakh. No. Of units varies.")) == 2


def test_multi_scheme_bullets_not_cut_to_three():
    body = "\n".join(f"- The expense ratio of fund {i} is 0.{i}%." for i in range(1, 7))
    assert finalize(body, "https://groww.in/x", "2026-10-04").body.count("\n- ") == 5  # all 6 bullets kept
