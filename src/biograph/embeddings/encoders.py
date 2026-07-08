"""Bi-encoder and cross-encoder wrappers.

Section 1 of the notebook contrasts two retrieval architectures:

* **Bi-encoder** (dual-encoder): query and document are embedded *independently*
  and compared by cosine similarity. Cheap and pre-computable → good for the
  first-stage recall over millions of documents.
* **Cross-encoder** (re-ranker): query and document are concatenated and passed
  jointly through self-attention, giving a single relevance score. Accurate but
  quadratic → used only to re-rank a small candidate set.

The classic pipeline is *bi-encoder recall → cross-encoder re-rank*.
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np


class BiEncoder:
    """Independent sentence embeddings with cosine similarity search."""

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str] | str, normalize: bool = True) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=normalize)
        return np.asarray(vecs)

    def rank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        """Return ``(doc_index, cosine_score)`` sorted high→low."""
        q = self.encode(query)
        d = self.encode(list(documents))
        scores = d @ q  # already normalised → cosine
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]


class CrossEncoderReranker:
    """Joint query-document scoring for precise re-ranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]


@functools.lru_cache(maxsize=4)
def get_default_encoder(model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO") -> BiEncoder:
    """Cached bi-encoder so heavy models load once per process."""
    return BiEncoder(model_name)
