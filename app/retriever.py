"""Semantic search: embed the query, ask Chroma for the nearest chunks."""
from dataclasses import dataclass


@dataclass
class Hit:
    text: str
    source: str
    heading: str
    lang: str
    score: float  # cosine similarity: 1.0 = identical direction, lower = less related


class Retriever:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> list[Hit]:
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(text=doc, source=meta["source"], heading=meta["heading"],
                            lang=meta["lang"], score=1.0 - dist))  # cosine distance -> similarity
        return hits
