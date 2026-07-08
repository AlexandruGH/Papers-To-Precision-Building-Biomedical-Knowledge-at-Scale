"""Remote Neo4j / Aura persistence.

Port of the notebook's ``BioGraphManager`` / ``EnhancedBioGraphManager``. Stores
entity embeddings on nodes so a single database supports both structural (Cypher)
and semantic (vector) queries — the "hybrid" store in GraphRAG.

Get a free instance at https://neo4j.com/cloud/aura/ and set NEO4J_* env vars.
"""

from __future__ import annotations

from typing import Sequence

from biograph.config import Settings
from biograph.schema import Triplet


class Neo4jStore:
    def __init__(self, settings: Settings | None = None,
                 uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        s = settings or Settings.from_env()
        uri = uri or s.neo4j_uri
        user = user or s.neo4j_user
        password = password or s.neo4j_password
        if not (uri and user and password):
            raise ValueError("Neo4j credentials missing. Set NEO4J_URI/USER/PASSWORD.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def ingest_triplets(self, triplets: Sequence[Triplet]) -> None:
        with self.driver.session() as session:
            for t in triplets:
                if t.pmid:
                    session.run(
                        "MERGE (a:Article {pmid: $pmid}) SET a.title = $title",
                        pmid=t.pmid, title=t.context[:120],
                    )
                # Node labels are interpolated (Cypher can't parameterise labels);
                # they come from a fixed, trusted vocabulary so this is safe.
                h_label = _safe_label(t.head.label)
                t_label = _safe_label(t.tail.label)
                session.run(
                    f"""
                    MERGE (h:{h_label} {{name: $hn}})
                      ON CREATE SET h.embedding = $he
                    MERGE (t:{t_label} {{name: $tn}})
                      ON CREATE SET t.embedding = $te
                    MERGE (h)-[r:{t.rel_type}]->(t)
                    SET r.status = $status, r.pmid = $pmid,
                        r.confidence = $score, r.source_context = $ctx
                    """,
                    hn=t.head.name, he=_emb(t.head.embedding),
                    tn=t.tail.name, te=_emb(t.tail.embedding),
                    status=t.status, pmid=t.pmid, score=float(t.score), ctx=t.context,
                )
        print(f"Ingested {len(triplets)} triplets into Neo4j.")

    def query(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as session:
            return session.run(cypher, **params).data()


def _safe_label(label: str) -> str:
    cleaned = "".join(ch for ch in label if ch.isalnum())
    return cleaned or "Entity"


def _emb(embedding):
    return None if embedding is None else list(map(float, embedding))
