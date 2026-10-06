from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

from app.ingest import get_collection, ingest
from app.llm import LLMUnavailable
from app.prompts import NO_ANSWER
from app.rag import RagPipeline
from app.retriever import Retriever

DOCS = str(Path(__file__).resolve().parents[1] / "data" / "docs")


def test_ingest_is_idempotent_and_detects_changes(tmp_path, fake_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## One\nAlpha text here.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\n## Two\nBeta text here.", encoding="utf-8")
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))

    first = ingest(client, fake_embedder, str(docs), 600, 100)
    assert sorted(first["added"]) == ["a.md", "b.md"]

    second = ingest(client, fake_embedder, str(docs), 600, 100)
    assert second["skipped"] == ["a.md", "b.md"] and not second["added"]

    (docs / "a.md").write_text("# A\n\n## One\nAlpha text CHANGED.", encoding="utf-8")
    (docs / "b.md").unlink()
    third = ingest(client, fake_embedder, str(docs), 600, 100)
    assert third["updated"] == ["a.md"] and third["removed"] == ["b.md"]
    col = get_collection(client, fake_embedder.model_name, 600)
    assert "CHANGED" in col.get()["documents"][0] and col.count() == 1


def _retriever(tmp_path, fake_embedder):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    ingest(client, fake_embedder, DOCS, 600, 100)
    return Retriever(get_collection(client, fake_embedder.model_name, 600), fake_embedder)


def test_retrieval_finds_right_file(tmp_path, fake_embedder):
    r = _retriever(tmp_path, fake_embedder)
    hits = r.search("Johor Bahru Straits Fertility Centre Malaysia", 3)
    assert hits[0].source == "04_hospitals_malaysia_turkey.md"
    assert hits[0].score > hits[-1].score  # sorted best-first


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens):
        self.calls += 1
        assert "CONTEXT:" in messages[0]["content"]
        return "stub answer [1]", "fake-llm"


def test_gate_refuses_without_calling_llm(tmp_path, fake_embedder):
    llm = FakeLLM()
    pipe = RagPipeline(_retriever(tmp_path, fake_embedder), llm, top_k=3, min_score=0.99, max_tokens=100)
    res = pipe.answer("What is the weather today?", [])
    assert not res.grounded and res.answer == NO_ANSWER["en"] and llm.calls == 0


def test_gate_replies_in_bangla_for_bangla_question(tmp_path, fake_embedder):
    pipe = RagPipeline(_retriever(tmp_path, fake_embedder), FakeLLM(), 3, 0.99, 100)
    assert pipe.answer("আজকের আবহাওয়া কেমন?", []).answer == NO_ANSWER["bn"]


def test_pipeline_calls_llm_when_context_is_relevant(tmp_path, fake_embedder):
    llm = FakeLLM()
    pipe = RagPipeline(_retriever(tmp_path, fake_embedder), llm, 3, 0.0, 100)
    res = pipe.answer("How long does an ambulance take in Dhaka?", [])
    assert res.grounded and llm.calls == 1 and res.sources


def test_api_search_and_chat_without_llm_key(tmp_path, fake_embedder, monkeypatch):
    from app import config, main

    client = chromadb.PersistentClient(path=str(tmp_path / "api_db"))
    ingest(client, fake_embedder, DOCS, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP)
    monkeypatch.setattr(main, "Embedder", lambda name: fake_embedder)
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "api_db"))
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", fake_embedder.model_name)

    with TestClient(main.app) as api:
        assert api.get("/health").json()["chunks"] > 10
        res = api.post("/search", json={"query": "emergency ambulance ICU", "k": 3})
        assert res.status_code == 200 and res.json()[0]["source"] == "07_emergency_ambulance.md"
        assert api.post("/chat", json={"message": "hello", "session_id": "s1"}).status_code == 503
        assert api.post("/chat", json={"message": "", "session_id": "s1"}).status_code == 422
