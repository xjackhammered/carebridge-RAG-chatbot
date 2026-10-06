"""FastAPI service.   Run:  uvicorn app.main:app --reload"""
import logging
import time
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.embedding import Embedder
from app.ingest import get_collection
from app.llm import LLM, LLMUnavailable
from app.rag import RagPipeline
from app.retriever import Retriever

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy objects are created ONCE at startup (not at import time), and stored on app.state.
    embedder = Embedder(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = get_collection(client, config.EMBEDDING_MODEL, config.CHUNK_MAX_CHARS)
    if collection.count() == 0:
        log.warning("Collection is empty. Run: python -m app.ingest")

    llm = None
    if config.GROQ_API_KEY:
        llm = LLM(config.GROQ_API_KEY, config.LLM_MODELS)
        try:  # warn early if a configured model has been retired
            live = llm.available_models()
            for m in config.LLM_MODELS:
                if m not in live:
                    log.warning("Configured model %s is NOT available on Groq", m)
        except Exception as e:
            log.warning("Could not verify Groq models: %s", e)
    else:
        log.warning("GROQ_API_KEY not set: /search works, /chat will return 503")

    retriever = Retriever(collection, embedder)
    app.state.retriever = retriever
    app.state.collection = collection
    app.state.pipeline = RagPipeline(retriever, llm, config.TOP_K, config.MIN_SCORE, config.LLM_MAX_TOKENS)
    app.state.sessions = {}  # session_id -> {"t": last_used, "history": [...]}
    yield


app = FastAPI(title="CareBridge RAG Assistant", version="2.0.0", lifespan=lifespan)


# ---------- schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(min_length=1, max_length=64)


class SourceOut(BaseModel):
    source: str
    heading: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    model: str | None
    top_score: float | None
    sources: list[SourceOut]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


def _sources(hits):
    return [SourceOut(source=h.source, heading=h.heading, score=round(h.score, 4), text=h.text) for h in hits]


def _get_session(sessions: dict, sid: str) -> dict:
    """Sessions expire after a TTL and the store is size-capped. The old version
    kept every conversation in memory forever."""
    now = time.time()
    for key in [k for k, v in sessions.items() if now - v["t"] > config.SESSION_TTL_SECONDS]:
        del sessions[key]
    if sid not in sessions and len(sessions) >= config.MAX_SESSIONS:
        del sessions[min(sessions, key=lambda k: sessions[k]["t"])]
    sess = sessions.setdefault(sid, {"t": now, "history": []})
    sess["t"] = now
    return sess


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "chunks": app.state.collection.count(),
            "embedding_model": config.EMBEDDING_MODEL, "llm_models": config.LLM_MODELS,
            "min_score": config.MIN_SCORE}


@app.post("/search", response_model=list[SourceOut])
def search(req: SearchRequest):
    """Retrieval only, no LLM. Great for demos and debugging: you can SEE what the retriever finds."""
    return _sources(app.state.retriever.search(req.query, req.k))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sess = _get_session(app.state.sessions, req.session_id)
    try:
        result = app.state.pipeline.answer(req.message, sess["history"])
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")
    sess["history"] += [{"role": "user", "content": req.message},
                        {"role": "assistant", "content": result.answer}]
    sess["history"] = sess["history"][-config.HISTORY_TURNS:]
    return ChatResponse(answer=result.answer, grounded=result.grounded, model=result.model,
                        top_score=None if result.top_score is None else round(result.top_score, 4),
                        sources=_sources(result.sources))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    app.state.sessions.pop(session_id, None)
    return {"status": "cleared"}
