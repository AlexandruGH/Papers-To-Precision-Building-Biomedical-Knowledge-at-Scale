"""Semantic entity resolution (a.k.a. entity linking / coreference).

The single most important step for graph quality: mentions like ``EGFR``,
``EGFR receptor`` and ``epidermal growth factor receptor`` must collapse onto
one canonical node, otherwise multi-hop reasoning fractures across duplicates
(the "Semantic Bridge" argument in the notebook's closing section).

We resolve by embedding each surface form and merging mentions whose cosine
similarity to an existing canonical vector exceeds a threshold. An encoder is
optional: when ``sentence-transformers`` is unavailable we fall back to a hashed
character-n-gram vectoriser so the pipeline still runs offline (lower recall,
but deterministic and dependency-free).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _hashed_ngram_encoder(dim: int = 256) -> EncodeFn:
    """Cheap deterministic fallback embedding (character 3-grams → hashed bag)."""

    def encode(texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            s = f"  {text.lower()} "
            for j in range(len(s) - 2):
                gram = s[j : j + 3]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16) % dim
                mat[i, h] += 1.0
            norm = np.linalg.norm(mat[i])
            if norm:
                mat[i] /= norm
        return mat

    return encode


class EntityResolver:
    """Maintains a registry of canonical entities and links new mentions to them."""

    def __init__(self, encode_fn: EncodeFn | None = None, threshold: float = 0.92):
        self.threshold = threshold
        self._encode = encode_fn or _hashed_ngram_encoder()
        # canonical name -> embedding
        self._registry: dict[str, np.ndarray] = {}

    @classmethod
    def from_sentence_transformer(cls, model_name: str, threshold: float = 0.92) -> "EntityResolver":
        from biograph.embeddings.encoders import BiEncoder

        enc = BiEncoder(model_name)
        return cls(encode_fn=lambda xs: enc.encode(list(xs)), threshold=threshold)

    @property
    def registry(self) -> dict[str, np.ndarray]:
        return self._registry

    def resolve(self, name: str) -> tuple[str, np.ndarray]:
        """Return the canonical name (and its vector) for a single mention."""
        return self.resolve_batch([name])[name.strip()]

    def resolve_batch(self, names: Sequence[str]) -> dict[str, tuple[str, np.ndarray]]:
        """Resolve many mentions at once. Returns ``{input_name: (canonical, vec)}``.

        New canonical entities discovered within the batch are added to the
        registry immediately, so intra-batch duplicates also merge.
        """
        cleaned = [n.strip() for n in names]
        if not cleaned:
            return {}
        vecs = self._encode(cleaned)
        results: dict[str, tuple[str, np.ndarray]] = {}
        for name, vec in zip(cleaned, vecs):
            canonical, cvec = self._match_or_add(name, np.asarray(vec))
            results[name] = (canonical, cvec)
        return results

    def _match_or_add(self, name: str, vec: np.ndarray) -> tuple[str, np.ndarray]:
        if not self._registry:
            self._registry[name] = vec
            return name, vec
        canon_names = list(self._registry.keys())
        canon_vecs = np.array(list(self._registry.values()))
        sims = cosine_similarity([vec], canon_vecs)[0]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return canon_names[best], canon_vecs[best]
        self._registry[name] = vec
        return name, vec

    def resolve_triplets(self, triplets, *, verbose: bool = False):
        """Rewrite a list of Triplets so head/tail point at canonical entities.

        Returns ``(resolved_triplets, merge_report)`` where ``merge_report`` maps
        each merged surface form to the canonical name it collapsed into.
        """
        merges: dict[str, str] = {}
        # First pass: collect all surface names for a single batched encode.
        surface = []
        for t in triplets:
            surface.extend([t.head.name, t.tail.name])
        mapping = self.resolve_batch(surface)

        for t in triplets:
            for ent in (t.head, t.tail):
                canonical, vec = mapping[ent.name.strip()]
                if canonical != ent.name.strip():
                    merges[ent.name] = canonical
                ent.name = canonical
                ent.embedding = vec
        if verbose and merges:
            print(f"Entity resolution merged {len(merges)} mentions:")
            for surf, canon in merges.items():
                print(f"  {surf!r} -> {canon!r}")
        return triplets, merges
