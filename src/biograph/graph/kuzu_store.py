"""Embedded Cypher persistence with Kùzu.

Port of the notebook's ``LocalCypherManager``. Kùzu is an embedded graph DB
(think "SQLite for graphs"): no server, a single on-disk directory, full Cypher.
Ideal for a self-contained pipeline that still supports structural queries and
multi-hop traversal (see :mod:`biograph.analytics`).
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Sequence

from biograph.schema import Triplet


class KuzuStore:
    def __init__(self, db_path: str = "./demo_db", embedding_dim: int = 768, reset: bool = True):
        import kuzu

        self.db_path = db_path
        self.embedding_dim = embedding_dim
        if reset and os.path.exists(db_path):
            shutil.rmtree(db_path) if os.path.isdir(db_path) else os.remove(db_path)

        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._setup_schema()
        print(f"Kùzu graph initialised at {db_path}")

    def _setup_schema(self) -> None:
        stmts = [
            "CREATE NODE TABLE Article(pmid STRING, title STRING, PRIMARY KEY (pmid))",
            f"CREATE NODE TABLE Entity(name STRING, label STRING, "
            f"embedding FLOAT[{self.embedding_dim}], PRIMARY KEY (name))",
            "CREATE REL TABLE MENTIONED_IN(FROM Entity TO Article)",
            "CREATE REL TABLE INTERACTS(FROM Entity TO Entity, type STRING, "
            "confidence DOUBLE, status STRING, context STRING)",
        ]
        for stmt in stmts:
            try:
                self.conn.execute(stmt)
            except Exception:  # noqa: BLE001 - table already exists on re-open
                pass

    def _embed(self, triplet_entity_embedding, name: str) -> list[float]:
        if triplet_entity_embedding is not None:
            vec = list(map(float, triplet_entity_embedding))
            if len(vec) == self.embedding_dim:
                return vec
        return [0.0] * self.embedding_dim

    def ingest_triplets(self, triplets: Sequence[Triplet]) -> None:
        for t in triplets:
            if t.pmid:
                self.conn.execute(
                    "MERGE (a:Article {pmid: $pmid}) SET a.title = $title",
                    {"pmid": str(t.pmid), "title": t.context[:120]},
                )
            self.conn.execute(
                "MERGE (h:Entity {name: $n}) SET h.label = $l, h.embedding = $e",
                {"n": t.head.name, "l": t.head.label, "e": self._embed(t.head.embedding, t.head.name)},
            )
            self.conn.execute(
                "MERGE (t:Entity {name: $n}) SET t.label = $l, t.embedding = $e",
                {"n": t.tail.name, "l": t.tail.label, "e": self._embed(t.tail.embedding, t.tail.name)},
            )
            self.conn.execute(
                "MATCH (h:Entity), (t:Entity) WHERE h.name = $n1 AND t.name = $n2 "
                "CREATE (h)-[:INTERACTS {type: $rel, confidence: $score, "
                "status: $status, context: $ctx}]->(t)",
                {"n1": t.head.name, "n2": t.tail.name, "rel": t.relation,
                 "score": float(t.score), "status": t.status, "ctx": t.context},
            )
            if t.pmid:
                self.conn.execute(
                    "MATCH (e:Entity), (a:Article) WHERE e.name = $n AND a.pmid = $pmid "
                    "MERGE (e)-[:MENTIONED_IN]->(a)",
                    {"n": t.head.name, "pmid": str(t.pmid)},
                )
        print(f"Ingested {len(triplets)} triplets into Kùzu.")

    def query(self, cypher: str, params: dict | None = None) -> list[tuple]:
        res = self.conn.execute(cypher, params or {})
        rows = []
        while res.has_next():
            rows.append(tuple(res.get_next()))
        return rows
