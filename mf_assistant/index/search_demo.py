"""Learning tool: run one query through different search types and compare.

Modes:
  dense    embedding similarity (FAISS), shown as cosine similarity
  bm25     keyword scoring (BM25) over the same chunks
  mmr      Maximal Marginal Relevance: relevant *and* diverse results
  filter   dense search restricted by metadata, e.g. scheme_id=...elss...

Run:  python -m mf_assistant.index.search_demo "exit load of ELSS fund" [--mode dense|bm25|mmr|filter|all] [-k 5]
"""

import argparse
import json
import re

from rank_bm25 import BM25Okapi

from mf_assistant import config
from mf_assistant.index.build_index import load_index


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def l2_to_cosine(squared_l2: float) -> float:
    """OpenAI vectors have length 1, so ||a-b||² = 2 - 2·cos(a,b)  ⇒  cos = 1 - d/2."""
    return 1 - squared_l2 / 2


def _label(meta: dict) -> str:
    return f"{meta['doc_type']:12} {meta['title'][:70]}"


def dense(store, query, k):
    print(f"\n== DENSE (cosine similarity, higher = closer)")
    for doc, dist in store.similarity_search_with_score(query, k=k):
        print(f"  {l2_to_cosine(dist):.3f}  {_label(doc.metadata)}")


def bm25(chunks, query, k):
    print(f"\n== BM25 (keyword score, unbounded)")
    index = BM25Okapi([tokenize(c["text"]) for c in chunks])
    scores = index.get_scores(tokenize(query))
    for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]:
        print(f"  {scores[i]:6.2f}  {_label(chunks[i]['metadata'])}")


def mmr(store, query, k):
    print(f"\n== MMR (fetch 20 by similarity, then pick {k} that are relevant but not redundant)")
    for doc in store.max_marginal_relevance_search(query, k=k, fetch_k=20, lambda_mult=0.5):
        print(f"         {_label(doc.metadata)}")


def filtered(store, query, k, scheme_hint="elss"):
    scheme_ids = {c["metadata"]["scheme_id"] for c in _chunks() if c["metadata"]["scheme_id"]}
    scheme_id = next(s for s in scheme_ids if scheme_hint in s)
    print(f"\n== FILTERED dense search (scheme_id = {scheme_id})")
    for doc, dist in store.similarity_search_with_score(query, k=k, filter={"scheme_id": scheme_id}, fetch_k=100):
        print(f"  {l2_to_cosine(dist):.3f}  {_label(doc.metadata)}")


def _chunks():
    return [json.loads(line) for line in config.CHUNKS_PATH.open(encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="all", choices=["dense", "bm25", "mmr", "filter", "all"])
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    store, chunks = load_index(), _chunks()
    print(f'Query: "{args.query}"')
    if args.mode in ("dense", "all"):
        dense(store, args.query, args.k)
    if args.mode in ("bm25", "all"):
        bm25(chunks, args.query, args.k)
    if args.mode in ("mmr", "all"):
        mmr(store, args.query, args.k)
    if args.mode == "filter":
        filtered(store, args.query, args.k)


if __name__ == "__main__":
    main()
