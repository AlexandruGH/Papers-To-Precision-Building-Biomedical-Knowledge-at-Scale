"""Zero-shot relation extraction with GLiNER.

Consolidates notebook cells 25–27 and ``extract_biomedical_triplets``. GLiNER
performs *zero-shot* NER + relation extraction: we simply hand it the entity
types (Disease/Gene/Drug/Mutation) and relation types (treats/inhibits/…) we
care about — no task-specific fine-tuning required.

Output is normalised into :class:`~biograph.schema.Triplet` objects, and each
triplet is tagged with an evidence *status* (Experimental / Clinical / …)
inferred from cheap lexical cues in the source abstract.
"""

from __future__ import annotations

from typing import Sequence

from biograph.schema import Article, Entity, Triplet

_EXPERIMENTAL_CUES = (
    "preclinical", "pre-clinical", "in vitro", "in vivo", "mouse", "murine",
    "cell line", "laboratory", "xenograft",
)
_CLINICAL_CUES = ("approved", "standard of care", "phase iii", "phase 3", "clinical trial")
_SIMULATION_CUES = ("in silico", "computational", "simulation", "modeled", "modelled")


def classify_status(text: str) -> str:
    """Heuristically label the evidence strength behind a mention."""
    low = text.lower()
    if any(cue in low for cue in _EXPERIMENTAL_CUES):
        return "Experimental"
    if any(cue in low for cue in _CLINICAL_CUES):
        return "Clinical"
    if any(cue in low for cue in _SIMULATION_CUES):
        return "Simulation"
    return "Mentioned"


class GLiNERExtractor:
    """Wraps ``gliner.GLiNER.predict_relations`` and emits typed triplets."""

    def __init__(
        self,
        model_name: str = "knowledgator/gliner-relex-large-v1.0",
        entity_labels: Sequence[str] | None = None,
        relation_labels: Sequence[str] | None = None,
        threshold: float = 0.10,
    ):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.entity_labels = list(entity_labels or ["Disease", "Gene", "Drug", "Mutation"])
        self.relation_labels = list(relation_labels or ["treats", "inhibits", "causes", "targets"])
        self.threshold = threshold

    def _predict(self, text: str) -> list[dict]:
        res = self.model.predict_relations(
            text, self.entity_labels, self.relation_labels, threshold=self.threshold
        )
        # GLiNER returns either (entities, relations) or a bare list depending on version.
        if isinstance(res, tuple) and len(res) > 1:
            return res[1]
        if isinstance(res, list):
            return res
        return []

    def extract(self, article: Article) -> list[Triplet]:
        """Extract triplets from a single article, attaching provenance."""
        status = classify_status(article.text)
        triplets: list[Triplet] = []
        for raw in self._predict(article.text):
            head, tail = raw.get("head"), raw.get("tail")
            if not (head and tail):
                continue
            triplets.append(
                Triplet(
                    head=Entity(head["text"], head.get("type", "Entity")),
                    tail=Entity(tail["text"], tail.get("type", "Entity")),
                    relation=raw.get("relation", "related_to"),
                    score=float(raw.get("score", 0.0)),
                    status=status,
                    pmid=article.pmid,
                    context=article.text[:250] + "…",
                )
            )
        return triplets

    def extract_corpus(self, articles: Sequence[Article]) -> list[Triplet]:
        all_triplets: list[Triplet] = []
        for art in articles:
            found = self.extract(art)
            all_triplets.extend(found)
            print(f"PMID {art.pmid}: {len(found)} triplets")
        print(f"Total: {len(all_triplets)} triplets from {len(articles)} articles")
        return all_triplets
