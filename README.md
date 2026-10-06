# CareBridge RAG Assistant

A bilingual (Bangla / English) retrieval-augmented generation service. It answers questions about a
fictional healthcare-facilitation company using only retrieved documents, cites its sources, and refuses
to answer when retrieval is weak.

> All data in `data/docs` is fictional. This project grew out of a production chatbot I built for a
> healthcare client; the data and branding here are synthetic.

## Architecture
```
data/docs/*.md -> chunking (heading-aware) -> multilingual-e5 embeddings -> ChromaDB (cosine)
                                                                              |
user question -> embed query -> top-k search -> score gate -> LLM (Groq) -> answer + citations
                                                  | (score < MIN_SCORE)
                                                  v
                                           "I don't have that information"
```

## Key design decisions
- **Multilingual embeddings.** (TODO: paste the MiniLM vs e5 result from eval/results.md.)
- **Heading-aware chunking** with the heading path prepended to each chunk.
- **Score gate** instead of trusting the prompt alone: weak retrieval never reaches the LLM.
- **Incremental, hash-based ingestion**: changed files are re-embedded, removed files deleted.
- **Configurable model fallback** with a startup check against Groq's live model list.

## Evaluation
(TODO: paste the table from `eval/results.md`)

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY
python -m app.ingest            # build the index
python -m eval.evaluate         # measure retrieval, get a MIN_SCORE suggestion
uvicorn app.main:app --reload   # http://localhost:8000/docs
pytest                          # tests need no model download
```
Docker: `docker build -t carebridge-rag . && docker run -p 8000:8000 --env-file .env -v $(pwd)/chroma_db:/app/chroma_db carebridge-rag`

## API
- `POST /chat` `{"message": "...", "session_id": "abc"}` -> answer, sources, grounded flag
- `POST /search` `{"query": "...", "k": 5}` -> retrieval only (no LLM)
- `GET /health`

## Limitations / future work
- Voice (STT/TTS) is not included; Bangla speech recognition quality was the blocker.
- Pure dense retrieval; hybrid BM25 + reranking would help exact-match queries.
- Small corpus; scaling would need a managed vector DB.
