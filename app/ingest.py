"""Read documents -> chunk -> embed -> store in ChromaDB.

Run it with:  python -m app.ingest

It is idempotent and incremental: each file's SHA-256 hash is stored with its
chunks. Unchanged file -> skipped. Changed file -> its old chunks are deleted and
re-created. File removed from disk -> its chunks are deleted. That is your answer
to "how do you handle documents changing?".
"""
import hashlib
import re
from pathlib import Path

import chromadb

from app import config
from app.chunking import chunk_markdown


def collection_name(model_name: str, max_chars: int) -> str:
    """One collection per (embedding model, chunk size). Vectors from different models
    live in different spaces and must never be mixed, so the name encodes both."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")[-40:]
    return f"kb_{slug}_c{max_chars}"[:63]


def get_collection(client, model_name: str, max_chars: int):
    # cosine distance: the right metric for normalized sentence embeddings
    return client.get_or_create_collection(
        collection_name(model_name, max_chars), metadata={"hnsw:space": "cosine"}
    )


def ingest(client, embedder, docs_dir: str, max_chars: int, overlap: int) -> dict:
    col = get_collection(client, embedder.model_name, max_chars)
    files = {p.name: p for p in sorted(Path(docs_dir).glob("*.md"))}
    stats = {"added": [], "updated": [], "skipped": [], "removed": []}

    # 1. remove chunks of files that no longer exist
    existing_sources = {m["source"] for m in (col.get(include=["metadatas"])["metadatas"] or [])}
    for gone in existing_sources - set(files):
        col.delete(where={"source": gone})
        stats["removed"].append(gone)

    # 2. add / update / skip each file
    for name, path in files.items():
        text = path.read_text(encoding="utf-8")
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        old = col.get(where={"source": name}, include=["metadatas"], limit=1)["metadatas"]
        if old and old[0].get("doc_hash") == doc_hash:
            stats["skipped"].append(name)
            continue
        if old:
            col.delete(where={"source": name})
        chunks = chunk_markdown(name, text, max_chars, overlap)
        if not chunks:
            continue
        col.add(
            ids=[f"{name}::{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embedder.embed_passages([c.text for c in chunks]),
            metadatas=[{"source": name, "heading": c.heading, "lang": c.lang,
                        "chunk_index": c.index, "doc_hash": doc_hash} for c in chunks],
        )
        stats["updated" if old else "added"].append(name)
    stats["total_chunks"] = col.count()
    return stats


if __name__ == "__main__":
    from app.embedding import Embedder

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    print(f"Loading embedding model {config.EMBEDDING_MODEL} ...")
    result = ingest(client, Embedder(config.EMBEDDING_MODEL), config.DOCS_DIR,
                    config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP)
    for key, value in result.items():
        print(f"{key}: {value}")
