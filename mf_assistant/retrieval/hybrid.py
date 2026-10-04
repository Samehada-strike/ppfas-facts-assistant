"""Online: hybrid retrieval = dense (FAISS) + keyword (BM25), fused with RRF.

For one question:
1. Scope: detect the scheme(s) mentioned. If any, search only those schemes'
   chunks plus general-knowledge chunks (help FAQs, blogs); other funds' chunks
   are excluded.
2. Dense: embed the question once and rank every in-scope chunk by cosine
   similarity (exact, via FAISS's flat index).
3. Keyword: rank in-scope chunks by BM25, using the question with scheme/AMC
   names stripped, so the shared fund name doesn't drown out the intent words.
4. Fuse: Reciprocal Rank Fusion over the top RRF_CANDIDATES of each list:
   score(chunk) = Σ 1 / (RRF_K + rank). It uses ranks only, so the two
   retrievers' incompatible score scales never need normalizing.
5. Confidence: if even the best dense similarity in scope is below
   MIN_DENSE_COSINE, the question is probably not covered by the corpus.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from mf_assistant import config
from mf_assistant.index.build_index import load_index
from mf_assistant.retrieval.schemes import SchemeResolver


_stemmer = SnowballStemmer("english")

# Filler words that carry no topic; BM25 would otherwise reward chunks for matching them
STOPWORDS = set("""
a an the and or but if of to in on at by for with from into about as is are was were be been being
do does did done have has had i me my we our you your it its this that these those there here
what which who whom whose when where why how can could should would will shall may might must
please tell know want need get got any some all much many more most very also just than then so
""".split())


def tokenize(text: str) -> list[str]:
    """Keyword tokens for BM25: lowercase words, minus stopwords, stemmed ("managers" → "manag")."""
    return [_stemmer.stem(w) for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float                 # fused (RRF) score, or the single retriever's score
    cosine: float                # dense similarity to the question
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class RetrievalResult:
    query: str
    scheme_ids: list[str]
    hits: list[Hit] = field(default_factory=list)
    best_cosine: float = 0.0
    confident: bool = False


class HybridRetriever:
    def __init__(self):
        self.store = load_index()
        chunks = {c["chunk_id"]: c for c in map(json.loads, config.CHUNKS_PATH.open(encoding="utf-8"))}
        # Align everything to FAISS row order, so row i in FAISS = self.chunks[i]
        self.chunks = [chunks[self.store.index_to_docstore_id[i]] for i in range(self.store.index.ntotal)]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.resolver = SchemeResolver()

    def _scope(self, scheme_ids: list[str]) -> np.ndarray:
        """Boolean mask over chunks: True = searchable for this question."""
        if not scheme_ids:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c["metadata"]["scheme_id"] in (None, *scheme_ids) for c in self.chunks])

    def _dense(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to every chunk (row order)."""
        q = np.array([self.store.embedding_function.embed_query(query)], dtype="float32")
        dist, idx = self.store.index.search(q, self.store.index.ntotal)  # flat index: exact, all rows
        cos = np.empty(self.store.index.ntotal, dtype="float32")
        cos[idx[0]] = 1 - dist[0] / 2  # unit vectors: squared L2 = 2 - 2·cos
        return cos

    def retrieve(self, query: str, k: int = config.RETRIEVAL_K, mode: str = "hybrid",
                 use_schemes: bool = True, strip_names: bool = False) -> RetrievalResult:
        """mode: "hybrid" | "dense" | "bm25". use_schemes=False disables the scheme filter
        and name stripping (for measuring what they contribute)."""
        match = self.resolver.resolve(query)
        if not use_schemes:
            match.scheme_ids, match.stripped_query = [], query
        scope = self._scope(match.scheme_ids)
        in_scope = np.flatnonzero(scope)

        cos = self._dense(query)
        dense_order = in_scope[np.argsort(-cos[in_scope])]
        bm25_scores = self.bm25.get_scores(tokenize(match.stripped_query if strip_names else query))
        bm25_order = [i for i in in_scope[np.argsort(-bm25_scores[in_scope])] if bm25_scores[i] > 0]

        dense_rank = {int(i): r for r, i in enumerate(dense_order[:config.RRF_CANDIDATES], 1)}
        bm25_rank = {int(i): r for r, i in enumerate(bm25_order[:config.RRF_CANDIDATES], 1)}

        if mode == "dense":
            ranked = [(int(i), float(cos[i])) for i in dense_order[:k]]
        elif mode == "bm25":
            ranked = [(int(i), float(bm25_scores[i])) for i in bm25_order[:k]]
        else:
            fused = {}
            for weight, ranks in ((config.RRF_WEIGHTS["dense"], dense_rank), (config.RRF_WEIGHTS["bm25"], bm25_rank)):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + weight / (config.RRF_K + r)
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]

        best = float(cos[in_scope].max()) if len(in_scope) else 0.0
        hits = [
            Hit(chunk_id=self.chunks[i]["chunk_id"], text=self.chunks[i]["text"],
                metadata=self.chunks[i]["metadata"], score=s, cosine=float(cos[i]),
                dense_rank=dense_rank.get(i), bm25_rank=bm25_rank.get(i))
            for i, s in ranked
        ]
        return RetrievalResult(query, match.scheme_ids, hits, best, best >= config.MIN_DENSE_COSINE)
