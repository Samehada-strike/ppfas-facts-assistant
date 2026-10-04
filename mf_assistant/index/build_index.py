"""Offline: embed chunks.jsonl with OpenAI and save a FAISS index to disk.

- Every chunk's text (context header included) becomes one vector from
  EMBEDDING_MODEL_NAME, requested in batches.
- FAISS stores the vectors in a flat (exact, brute-force) index; LangChain's
  docstore keeps each chunk's text and metadata alongside it, keyed by chunk_id.
- index_info.json records the model and a hash of chunks.jsonl. The build is
  skipped when neither has changed (use --force to rebuild anyway), and the app
  can refuse an index built with a different model than the one it queries with.

Run:  python -m mf_assistant.index.build_index [--force]
"""

import hashlib
import json
import sys
import time
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from mf_assistant import config


def get_embeddings() -> OpenAIEmbeddings:
    """The one embedding model for both indexing and querying."""
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME, chunk_size=config.EMBED_BATCH_SIZE)


def chunks_hash() -> str:
    return hashlib.sha256(config.CHUNKS_PATH.read_bytes()).hexdigest()


def is_up_to_date() -> bool:
    if not (config.INDEX_INFO_PATH.exists() and (config.FAISS_INDEX_DIR / "index.faiss").exists()):
        return False
    info = json.loads(config.INDEX_INFO_PATH.read_text(encoding="utf-8"))
    return info.get("chunks_sha256") == chunks_hash() and info.get("embedding_model") == config.EMBEDDING_MODEL_NAME


def main(force: bool = False):
    if not force and is_up_to_date():
        print("Index is up to date (same chunks, same model); nothing to do. Use --force to rebuild.")
        return

    chunks = [json.loads(line) for line in config.CHUNKS_PATH.open(encoding="utf-8")]
    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL_NAME} "
          f"(batches of {config.EMBED_BATCH_SIZE}) ...")
    started = time.time()
    store = FAISS.from_texts(
        texts=[c["text"] for c in chunks],
        embedding=get_embeddings(),
        metadatas=[{**c["metadata"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"]} for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
    )
    store.save_local(str(config.FAISS_INDEX_DIR))

    info = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "dimensions": store.index.d,
        "n_chunks": store.index.ntotal,
        "faiss_index_type": type(store.index).__name__,
        "chunks_sha256": chunks_hash(),
        "total_tokens": sum(c["metadata"]["tokens"] for c in chunks),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    config.INDEX_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"✓ {info['n_chunks']} vectors × {info['dimensions']} dims ({info['faiss_index_type']}) "
          f"in {time.time() - started:.1f}s → {config.FAISS_INDEX_DIR}")
    print(f"  ~{info['total_tokens']:,} tokens embedded ≈ ${info['total_tokens'] / 1e6 * 0.02:.4f}")


def load_index() -> FAISS:
    """Load the saved index for querying; refuse it if it was built with another model."""
    info = json.loads(config.INDEX_INFO_PATH.read_text(encoding="utf-8"))
    if info["embedding_model"] != config.EMBEDDING_MODEL_NAME:
        raise RuntimeError(f"Index built with {info['embedding_model']}, but the query model is "
                           f"{config.EMBEDDING_MODEL_NAME}; rebuild the index.")
    # allow_dangerous_deserialization: the docstore is a pickle; we only load files we built ourselves
    return FAISS.load_local(str(config.FAISS_INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
