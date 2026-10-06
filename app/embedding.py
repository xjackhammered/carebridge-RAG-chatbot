"""Wrapper around sentence-transformers.

The e5 model family was trained with prefixes: "query: " for search queries and
"passage: " for documents. Skipping them measurably hurts retrieval quality. Other
models (like MiniLM) don't want prefixes. Keeping that logic here means the rest
of the code never has to think about it.
"""


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # imported lazily: it's slow

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_prefix = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True -> unit vectors, so cosine similarity = dot product
        return self.model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([("passage: " + t) if self.use_prefix else t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([("query: " + text) if self.use_prefix else text])[0]
