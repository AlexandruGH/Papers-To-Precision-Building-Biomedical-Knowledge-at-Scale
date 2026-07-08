"""Typed data contracts shared across the pipeline.

The original notebook passed loosely-typed ``dict`` objects between cells. Here we
lift the recurring shapes into dataclasses so the ingestion → extraction →
graph → retrieval stages have a stable, documented interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Article:
    """A single PubMed record."""

    pmid: str
    title: str
    text: str  # abstract body
    year: Optional[int] = None
    journal: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"pmid": self.pmid, "title": self.title, "text": self.text,
                "year": self.year, "journal": self.journal}


@dataclass
class Entity:
    """A canonical node in the knowledge graph (after entity resolution)."""

    name: str
    label: str  # Disease | Gene | Drug | Mutation | Entity
    embedding: Optional[np.ndarray] = None

    def key(self) -> str:
        # Canonical identity is name + label; used for dedup / MERGE.
        return f"{self.label}:{self.name.lower()}"


@dataclass
class Triplet:
    """An extracted (head)-[relation]->(tail) fact with provenance."""

    head: Entity
    tail: Entity
    relation: str
    score: float = 0.0
    status: str = "Mentioned"  # Mentioned | Experimental | Simulation | Clinical
    pmid: Optional[str] = None
    context: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rel_type(self) -> str:
        """Cypher-safe uppercase relation label (e.g. ``TREATS``)."""
        return self.relation.upper().replace(" ", "_")

    def as_row(self) -> dict[str, Any]:
        return {
            "head": self.head.name,
            "head_label": self.head.label,
            "relation": self.relation,
            "tail": self.tail.name,
            "tail_label": self.tail.label,
            "score": round(self.score, 4),
            "status": self.status,
            "pmid": self.pmid,
        }
