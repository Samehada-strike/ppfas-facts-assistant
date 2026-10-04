"""Measure follow-up rewriting (condense) on labelled conversations, over several runs.

Each case: a short history, the new message, and regexes the standalone question must match.
Runs RUNS times because an LLM at temperature 0 can still vary between calls.

Run:  python -m mf_assistant.answer.eval_history
"""

import re

from mf_assistant.answer.history import ChatSession, Condenser

RUNS = 3

EXIT_ELSS = ("What is the exit load of PPFAS ELSS?",
             "The exit load of Parag Parikh ELSS Tax Saver Fund Direct Growth is: Nil.")
ASK_SCHEME = ("What is the minimum SIP for this fund?",
              "Which PPFAS scheme do you mean? I cover Flexi Cap, ELSS Tax Saver, Liquid, Arbitrage, "
              "Conservative Hybrid and Dynamic Asset Allocation.")

CASES = [
    ([EXIT_ELSS], "and its lock-in?", [r"lock", r"elss|tax saver"]),
    ([EXIT_ELSS], "what about the flexi cap fund?", [r"exit load", r"flexi"]),
    ([EXIT_ELSS], "same for liquid", [r"exit load", r"liquid"]),
    ([ASK_SCHEME], "the liquid one", [r"(minimum|min)\.? sip", r"liquid"]),
    ([ASK_SCHEME], "ELSS", [r"(minimum|min)\.? sip", r"elss|tax saver"]),
    ([EXIT_ELSS], "who manages it?", [r"manag", r"elss|tax saver"]),
    ([EXIT_ELSS], "How do I download my CAS?", [r"^How do I download my CAS\?$"]),  # already standalone
    ([EXIT_ELSS], "thanks", [r"^thanks$"]),                                           # small talk unchanged
    ([EXIT_ELSS], "is it worth buying?", [r"worth"]),                                 # advice intent kept → refused later
    ([EXIT_ELSS], "expense ratio of all PPFAS funds", [r"expense ratio", r"\ball\b"]),  # own scope kept, not ELSS
    ([EXIT_ELSS], "what is the NAV of the liquid fund?", [r"nav", r"liquid"]),        # complete question unchanged
]


def main():
    condenser = Condenser()
    for run in range(1, RUNS + 1):
        ok = 0
        for turns, message, patterns in CASES:
            session = ChatSession()
            for q, a in turns:
                session.add(q, a)
            out = condenser.standalone(message, session)
            good = all(re.search(p, out, re.I) for p in patterns)
            ok += good
            if not good:
                print(f"  run {run} FAIL: {message!r} -> {out!r}")
        print(f"run {run}: {ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
