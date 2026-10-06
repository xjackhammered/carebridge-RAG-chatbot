import math
import re

import pytest


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. NOT semantic: it only matches shared
    words. It lets us test all the plumbing (chunking, storage, retrieval, API)
    without downloading a 1GB model. Real quality is measured by eval.evaluate."""
    DIM = 512

    def __init__(self, model_name="fake-model"):
        self.model_name = model_name

    def _vec(self, text):
        v = [0.0] * self.DIM
        for tok in re.findall(r"\w+", text.lower()):
            h = sum(ord(c) * 31 ** i for i, c in enumerate(tok)) % (2 * self.DIM)
            v[h % self.DIM] += 1.0 if h < self.DIM else -1.0
            v[(h * 7) % self.DIM] += 0.5
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
