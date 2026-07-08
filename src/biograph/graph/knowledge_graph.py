"""In-memory knowledge graph backed by ``networkx.MultiDiGraph``.

This is the backend-agnostic heart of the project. Triplets are ingested here
first; persistence stores (Kùzu/Neo4j) and the learning components (GraphSAGE,
G-Retriever) all read from this structure. Keeping one canonical in-memory graph
means the GNN and retrieval code never has to speak Cypher.

Node attributes:  ``label`` (Disease/Gene/Drug/Mutation), ``embedding`` (np.ndarray|None),
                  ``pmids`` (set of source articles).
Edge attributes:  ``relation``, ``status``, ``score``, ``pmid``, ``context``.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

import networkx as nx
import numpy as np

from biograph.schema import Triplet


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ------------------------------------------------------------------ build
    def add_triplet(self, t: Triplet) -> None:
        self._add_node(t.head.name, t.head.label, t.head.embedding, t.pmid)
        self._add_node(t.tail.name, t.tail.label, t.tail.embedding, t.pmid)
        self.g.add_edge(
            t.head.name,
            t.tail.name,
            key=t.rel_type,
            relation=t.relation,
            status=t.status,
            score=float(t.score),
            pmid=t.pmid,
            context=t.context,
        )

    def add_triplets(self, triplets: Iterable[Triplet]) -> "KnowledgeGraph":
        for t in triplets:
            self.add_triplet(t)
        return self

    def _add_node(self, name: str, label: str, embedding, pmid: Optional[str]) -> None:
        if name in self.g:
            data = self.g.nodes[name]
            if embedding is not None and data.get("embedding") is None:
                data["embedding"] = np.asarray(embedding)
            if pmid:
                data["pmids"].add(pmid)
        else:
            self.g.add_node(
                name,
                label=label,
                embedding=None if embedding is None else np.asarray(embedding),
                pmids={pmid} if pmid else set(),
            )

    # --------------------------------------------------------------- accessors
    @property
    def num_nodes(self) -> int:
        return self.g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.g.number_of_edges()

    def nodes(self) -> list[str]:
        return list(self.g.nodes())

    def node_label(self, name: str) -> str:
        return self.g.nodes[name].get("label", "Entity")

    def neighbors(self, name: str) -> Iterator[str]:
        yield from self.g.successors(name)
        yield from self.g.predecessors(name)

    def edge_triples(self) -> list[tuple[str, str, dict]]:
        """Return ``(head, tail, edge_attrs)`` for every edge."""
        return [(u, v, d) for u, v, d in self.g.edges(data=True)]

    def degree_ranking(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Highest-degree nodes = most central entities (notebook 'High-Impact Genes')."""
        deg = sorted(self.g.degree(), key=lambda kv: kv[1], reverse=True)
        return [(n, d) for n, d in deg[:top_k]]

    # -------------------------------------------------------------- embeddings
    def embedding_matrix(self, dim: int) -> tuple[np.ndarray, list[str]]:
        """Return ``(X, node_order)`` — a dense feature matrix for GNN input.

        Nodes lacking a text embedding get a zero row (GraphSAGE can still learn
        a structural representation for them from the neighbourhood).
        """
        order = self.nodes()
        X = np.zeros((len(order), dim), dtype=np.float32)
        for i, n in enumerate(order):
            emb = self.g.nodes[n].get("embedding")
            if emb is not None and len(emb) == dim:
                X[i] = emb
        return X, order

    def to_networkx(self) -> nx.MultiDiGraph:
        return self.g

    # ------------------------------------------------------------------- stats
    def summary(self) -> dict:
        labels: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            labels[d.get("label", "Entity")] = labels.get(d.get("label", "Entity"), 0) + 1
        rels: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            rels[d.get("relation", "?")] = rels.get(d.get("relation", "?"), 0) + 1
        return {"nodes": self.num_nodes, "edges": self.num_edges,
                "labels": labels, "relations": rels}

    @classmethod
    def from_triplets(cls, triplets: Sequence[Triplet]) -> "KnowledgeGraph":
        return cls().add_triplets(triplets)
