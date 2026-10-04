# -------------------------------------------------------------------------
# COPY THIS INTO YOUR NOTEBOOK TO FIX THE RETRIEVAL ISSUE
# -------------------------------------------------------------------------

# The issue was that 'PPFAS' was not matching 'Parag Parikh' in the documents.
# We need to apply the same 'canonicalize' function from Section 2 to the Unstructured Query.

def hybrid_retrieve(query: str, k: int = RETRIEVAL_K) -> List[Document]:
    if not _BM25_INDEX or not _VECTOR_STORE:
        return []

    # ---------------------------------------------------------
    # FIX: Normalize the query (e.g., PPFAS -> Parag Parikh)
    # ---------------------------------------------------------
    # Ensure 'canonicalize' is defined (it should be from Section 2)
    try:
        query_norm = canonicalize(query)
    except NameError:
        # Fallback if canonicalize isn't in scope (e.g. distinct sections)
        print("⚠️ 'canonicalize' function not found. Using raw query.")
        query_norm = query

    # 1. BM25 Search (Use normalized query)
    bm25_scores = _BM25_INDEX.get_scores(split_text_tokenized(query_norm))
    
    # Pair (index, score) and sort
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k*2]
    
    # Normalize BM25 (simple min-max of top results)
    bm25_results = {}
    if bm25_ranked:
        max_score = bm25_ranked[0][1] or 1.0
        for idx, score in bm25_ranked:
            doc = _DOCS[idx]
            bm25_results[id(doc)] = {
                "doc": doc,
                "score": (score / max_score) * WEIGHTS["bm25"]
            }

    # 2. Vector Search (Use normalized query)
    vector_results_raw = _VECTOR_STORE.similarity_search_with_score(query_norm, k=k*2)
    
    combined_results = bm25_results.copy()
    
    if vector_results_raw:
        # FAISS returns L2 distance (lower is better).
        # We perform a robust normalization based on the max distance in the batch.
        max_dist = max(score for _, score in vector_results_raw) or 1.0
        
        for doc, dist in vector_results_raw:
            # Invert distance: (1 - dist/max) * weight
            # Adding epsilon to max_dist to avoid division by zero
            sim_score = (1 - (dist / (max_dist + 0.01))) * WEIGHTS["vector"]
            
            doc_id = id(doc)
            if doc_id in combined_results:
                combined_results[doc_id]["score"] += sim_score
            else:
                combined_results[doc_id] = {
                    "doc": doc,
                    "score": sim_score
                }

    # Sort final results by combined score
    final_ranked = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in final_ranked[:k]]
