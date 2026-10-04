"""Offline: split documents.jsonl into retrieval-sized chunks → chunks.jsonl.

Strategy (sizes in tokens):
- A document of at most CHUNK_MAX_TOKENS stays one chunk. Phase 1 already kept
  natural units (one FAQ, one heading section), so this covers most documents.
- A longer document is split by structure first:
  - Markdown tables are split between rows, repeating the header row in every
    piece. A long 2-column table (the glossary) gets one chunk per term.
  - Bulleted lists (e.g. fund managers) are split between items, repeating the
    intro line.
  - Remaining prose is split recursively (paragraph → line → sentence → word)
    with overlap.
- Every chunk's text starts with a context header built from its metadata,
  e.g. "[Parag Parikh ELSS Tax Saver Fund Direct Growth | FAQ]". The embedding
  sees only the text, so this is how a chunk "knows" what it is about.

Run:  python -m mf_assistant.index.chunk
"""

import json
import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mf_assistant import config

_ENC = tiktoken.get_encoding(config.TOKENIZER)

SECTION_LABELS = {
    "overview": "key facts", "objective": "investment objective", "performance": "returns and rankings",
    "fund_managers": "fund managers", "holdings": "top holdings", "faq": "FAQ",
}

_prose_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=config.TOKENIZER,
    chunk_size=config.CHUNK_TARGET_TOKENS,
    chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def context_header(meta: dict) -> str:
    t = meta["doc_type"]
    if t in ("scheme_fact", "scheme_faq"):
        return f"[{meta['scheme_name']} | {SECTION_LABELS.get(meta['section'], meta['section'])}]"
    if t == "help_faq":
        # Groww's topic slugs ("discoverable", "order") carry no meaning; the question line does
        return "[Groww Help Centre: mutual funds FAQ]"
    # Blog: page title > parent heading > own heading (skipping repeats)
    path = [meta["page_title"]]
    for heading in (meta.get("parent_heading"), None if t == "blog_faq" else meta["title"]):
        if heading and heading != path[-1]:
            path.append(heading)
    return f"[Groww blog: {' > '.join(path)}]"


# ---------- structural splitting ----------

def _blocks(text: str) -> list[tuple[str, list[str]]]:
    """Group lines into ("table" | "list" | "text", lines) blocks."""
    blocks = []
    for line in text.splitlines():
        kind = "table" if "|" in line else "list" if re.match(r"^\s*[-*] ", line) else "text"
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(line)
        else:
            blocks.append((kind, [line]))
    return blocks


def _pack(header_lines: list[str], items: list[str]) -> list[str]:
    """Greedily pack items into pieces of ~CHUNK_TARGET_TOKENS, repeating header_lines in each piece."""
    head = "\n".join(header_lines)
    budget = config.CHUNK_TARGET_TOKENS - n_tokens(head)
    pieces, current = [], []
    for item in items:
        if current and n_tokens("\n".join(current + [item])) > budget:
            pieces.append("\n".join(header_lines + current))
            current = []
        current.append(item)
    if current:
        pieces.append("\n".join(header_lines + current))
    return pieces


def _split_table(lines: list[str]) -> tuple[list[str], bool]:
    """Returns (pieces, atomic). Atomic pieces (glossary terms) must not be merged later."""
    has_header = len(lines) > 1 and re.match(r"^[\s|:-]+$", lines[1])
    header, rows = (lines[:2], lines[2:]) if has_header else ([], lines)
    n_cols = len(lines[0].split("|"))
    if n_cols == 2 and len(rows) >= config.GLOSSARY_MIN_ROWS:
        # Glossary: one self-contained "Term: definition" per chunk
        return [": ".join(c.strip() for c in row.split("|", 1)) for row in rows if row.strip()], True
    return _pack(header, rows), False


def split_body(body: str) -> list[str]:
    """Split one document body into pieces of at most ~CHUNK_MAX_TOKENS."""
    if n_tokens(body) <= config.CHUNK_MAX_TOKENS:
        return [body]

    pieces: list[tuple[str, bool]] = []  # (text, atomic)
    for kind, lines in _blocks(body):
        text = "\n".join(lines).strip()
        if not text:
            continue
        if kind == "table":
            table_pieces, atomic = _split_table(lines)
            pieces += [(p, atomic) for p in table_pieces]
        elif kind == "list" and n_tokens(text) > config.CHUNK_TARGET_TOKENS:
            pieces += [(p, False) for p in _pack([], lines)]
        elif n_tokens(text) > config.CHUNK_TARGET_TOKENS:
            pieces += [(p, False) for p in _prose_splitter.split_text(text)]
        else:
            pieces.append((text, False))

    # Merge neighbouring small non-atomic pieces (an intro line, a short paragraph)
    merged: list[tuple[str, bool]] = []
    for text, atomic in pieces:
        if (merged and not atomic and not merged[-1][1]
                and n_tokens(merged[-1][0] + "\n" + text) <= config.CHUNK_TARGET_TOKENS):
            merged[-1] = (merged[-1][0] + "\n" + text, False)
        else:
            merged.append((text, atomic))
    return [text for text, _ in merged]


def _manager_split(body: str) -> list[str]:
    """Fund-manager docs: keep the intro line ("Fund managers of X:") with every group of managers."""
    intro, *managers = body.splitlines()
    if n_tokens(body) <= config.CHUNK_MAX_TOKENS:
        return [body]
    return _pack([intro], managers)


# ---------- pipeline ----------

def chunk_document(doc: dict) -> list[dict]:
    meta = doc["metadata"]
    body = doc["page_content"]
    parts = _manager_split(body) if meta["section"] == "fund_managers" else split_body(body)
    header = context_header(meta)
    chunks = []
    for i, part in enumerate(parts):
        label = header if len(parts) == 1 else f"{header[:-1]} | part {i + 1} of {len(parts)}]"
        text = f"{label}\n{part.strip()}"
        chunks.append({
            "chunk_id": f"{doc['id']}-{i}",
            "doc_id": doc["id"],
            "text": text,
            "metadata": {**meta, "chunk_index": i, "n_chunks": len(parts), "tokens": n_tokens(text)},
        })
    return chunks


def main():
    docs = [json.loads(line) for line in config.DOCUMENTS_PATH.open(encoding="utf-8")]
    chunks = [c for d in docs for c in chunk_document(d)]

    too_big = [c for c in chunks if c["metadata"]["tokens"] > config.CHUNK_MAX_TOKENS + 30]
    if too_big:
        raise SystemExit(f"{len(too_big)} chunks exceed the size limit, e.g. {too_big[0]['chunk_id']}")

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(c["metadata"]["tokens"] for c in chunks)
    split_docs = sum(1 for d in docs if sum(c["doc_id"] == d["id"] for c in chunks) > 1)
    print(f"✓ {len(docs)} documents → {len(chunks)} chunks ({split_docs} documents were split) → {config.CHUNKS_PATH.name}")
    print(f"  tokens per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, total {sum(sizes):,}")


if __name__ == "__main__":
    main()
