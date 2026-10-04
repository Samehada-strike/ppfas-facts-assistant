"""Run every evaluation and write one report: EVALUATION.md at the repo root.

Each evaluation runs as its own process (python -m ...), so one failing doesn't stop the rest,
and its printed output goes into the report under its own heading. Library warnings are dropped.
Costs a few cents of OpenAI usage and takes several minutes.

Run:  python -m mf_assistant.evaluation.run_all
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime

from mf_assistant import config

REPORT_PATH = config.REPO_ROOT / "EVALUATION.md"

# (title, what it measures, command)
EVALS = [
    ("Unit tests", "PII detection, advice/comparison patterns, output formatting (no API calls)",
     [sys.executable, "-m", "pytest", "tests", "-q"]),
    ("Retrieval", "hit@k and MRR on 32 labelled questions: dense vs BM25 vs hybrid, with/without scheme filter; "
                  "'not found' threshold calibration", [sys.executable, "-m", "mf_assistant.retrieval.eval_retrieval"]),
    ("Routing", "39 labelled questions: handler + fact-field accuracy (incl. disguised advice, comparisons, off-topic)",
     [sys.executable, "-m", "mf_assistant.answer.eval_router"]),
    ("Decomposition", "12 messages: mixed questions answered in full, single requests kept whole, unsafe parts refused",
     [sys.executable, "-m", "mf_assistant.answer.eval_decompose"]),
    ("Grounded generation (LLM judge)", "16 answerable + 4 unanswerable questions; gpt-4o judges faithfulness",
     [sys.executable, "-m", "mf_assistant.answer.eval_generation"]),
    ("Held-out end-to-end", "21 new questions through the full app pipeline (facts, explanations, refusals)",
     [sys.executable, "-m", "mf_assistant.evaluation.heldout"]),
    ("RAGAS", "faithfulness, answer relevancy, context precision, context recall on the held-out explanations",
     [sys.executable, "-m", "mf_assistant.evaluation.ragas_eval"]),
]

_NOISE = re.compile(r"warn|Warning|DeprecationWarning|from ragas|LangChainDeprecation|^\s*$|ScriptRunContext")


def main():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    sections = []
    for title, measures, cmd in EVALS:
        print(f"Running: {title} ...", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=config.REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        output = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines() if not _NOISE.search(line))
        status = "ok" if proc.returncode == 0 else f"exit code {proc.returncode}"
        sections.append(f"## {title}\n\n*{measures}* ({time.time() - started:.0f} s, {status})\n\n"
                        f"```\n{output.strip()}\n```\n")

    REPORT_PATH.write_text(
        f"# Evaluation report\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by "
        f"`python -m mf_assistant.evaluation.run_all`. Models: {config.LLM_MODEL_NAME} (app), "
        f"{config.EMBEDDING_MODEL_NAME} (embeddings), gpt-4o (judge/RAGAS evaluator).\n\n"
        + "\n".join(sections), encoding="utf-8")
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
