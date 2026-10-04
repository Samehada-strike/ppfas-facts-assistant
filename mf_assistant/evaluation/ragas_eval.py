"""Score the held-out explanation answers with RAGAS.

Reads results/ragas_samples.json (written by heldout.py: question, the app's answer, the retrieved
passages, and a reference answer written from the source page) and computes four standard RAG metrics:

  faithfulness        share of the answer's claims supported by the retrieved passages   (generation)
  answer_relevancy    how directly the answer addresses the question                       (generation)
  context_precision   are the passages that support the reference ranked near the top?    (retrieval)
  context_recall      share of the reference answer's claims found in the passages         (retrieval)

All four are scored 0-1 by an evaluator LLM (gpt-4o, a stronger model than the app's gpt-4o-mini);
answer_relevancy also uses embeddings. Scores are estimates from an LLM judge, not ground truth.

Run:  python -m mf_assistant.evaluation.heldout   (first, to produce the samples)
      python -m mf_assistant.evaluation.ragas_eval
"""

import json
import warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from mf_assistant import config
from mf_assistant.evaluation.heldout import OUT_DIR, RAGAS_SAMPLES_PATH

EVALUATOR_MODEL = "gpt-4o"
RESULTS_CSV = OUT_DIR / "ragas_scores.csv"
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    samples = json.loads(RAGAS_SAMPLES_PATH.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.from_list(samples)

    result = evaluate(
        dataset=dataset,
        # strictness=1: the LangChain wrapper returns one generation per call (RAGAS's default asks for 3)
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)),
        run_config=RunConfig(max_workers=4, timeout=300),  # fewer parallel calls: the default timed out
        show_progress=False,
    )
    df = result.to_pandas()
    df.to_csv(RESULTS_CSV, index=False)

    cols = [c for c in METRIC_COLUMNS if c in df.columns]
    print(f"RAGAS on {len(df)} held-out explanation answers (evaluator: {EVALUATOR_MODEL})\n")
    print(f"{'question':58} " + " ".join(f"{c[:12]:>12}" for c in cols))
    for _, row in df.iterrows():
        print(f"{row['user_input'][:58]:58} " + " ".join(f"{row[c]:12.2f}" for c in cols))
    print(f"{'MEAN':58} " + " ".join(f"{df[c].mean():12.2f}" for c in cols))
    print(f"\nPer-sample scores → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
