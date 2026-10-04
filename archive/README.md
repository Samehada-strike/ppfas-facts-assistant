# Archive

Files kept for reference that the app and its pipeline no longer use. Nothing in `mf_assistant/` or
`app.py` reads from here.

| Folder / file | What it is |
|---|---|
| `legacy_pipeline/` | The first version of the project: crawl/clean scripts, the regex extraction notebook (`2-data_extraction.ipynb`, broken by Groww's page redesign), the SQLite builder, the DBML schema visualiser and the prototype RAG notebook (`5-RAG_Chatbot.ipynb`). Replaced by `mf_assistant/ingest` and `mf_assistant/answer`. Paths inside these files point at the old `project_files/` layout. |
| `legacy_pipeline/deprecated/` | Earlier notebook experiments (JSON parsing/chunking, a copy of the RAG notebook). One notebook with an API token in its outputs is git-ignored and exists only locally. |
| `legacy_data/` | Data produced by the notebook era: the old structured/chunkable JSON, pre-chunked JSONL files, and the old SQLite schema exports (`structured_data_v1/`). The current data is in `data/`. |
| `learning_material/` | Concept diagrams (LangChain document components, RAG architecture). |
| `requirements-course.txt` | The course's original package list (the project now uses `pyproject.toml` + `uv.lock`). |
| `main.py` | The empty uv project template. |
| `krish_naik_files/` | Course notebooks and PDFs. Local only (git-ignored: third-party material, and some files contain API keys). |
