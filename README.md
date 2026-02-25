# AI Study Buddy

CLI-first RAG pipeline for PDF study material using:
- Python 3.11
- LangChain
- Google Gemini API (embeddings + generation)
- ChromaDB (persistent vector storage)
- Streamlit (web interface)

## Project Layout

```text
data/             # Put input PDFs here
chroma_db/        # Chroma persistence + ingestion manifest
src/study_buddy/  # Core code
tests/            # Pytest suite
eval/             # Basic evaluation assets
```

## Quickstart (Windows PowerShell)

1. Create Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -e ".[dev]"
```

3. Configure environment:

```powershell
Copy-Item .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

`.env` and secret files are ignored by git via `.gitignore`.  
Use `.env.example` as the shareable template for config values.

4. Add your PDF files under `data/`.

5. Run health checks:

```powershell
study-buddy doctor
```

6. Ingest PDFs:

```powershell
study-buddy ingest --data-dir data --db-dir chroma_db --reset false
```

7. Ask questions:

```powershell
study-buddy ask --question "Summarize chapter 2" --k 6
```

8. Interactive mode:

```powershell
study-buddy repl --k 6
```

## Web App (Streamlit)

Run the dashboard:

```powershell
streamlit run src/study_buddy/app.py
```

The app provides:
- sidebar PDF upload and processing workflow
- chat-style interface with conversation memory
- persistent session-level vectorstore reuse

## Evaluation

Populate `eval/questions.jsonl` and run:

```powershell
python eval/run_eval.py --k 6
```

The evaluator performs:
- retrieval hit checks against expected source files
- answer generation smoke checks

## Troubleshooting

- `GOOGLE_API_KEY is required but not set`:
  - Set `GOOGLE_API_KEY` in `.env`.
- Gemini model errors:
  - Verify model names in `.env` (`GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL`).
- Empty retrieval results:
  - Confirm PDFs exist in `data/`.
  - Re-run `study-buddy ingest`.
- Chroma persistence issues:
  - Ensure `chroma_db/` is writable.
