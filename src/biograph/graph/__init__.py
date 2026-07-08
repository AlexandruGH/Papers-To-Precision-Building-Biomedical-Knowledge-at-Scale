"""Knowledge-graph construction and storage backends.

* :class:`KnowledgeGraph` — an in-memory ``networkx`` graph that is the canonical
  structure consumed by GraphSAGE and G-Retriever.
* :class:`KuzuStore` — embedded, file-based Cypher store (default persistence).
* :class:`Neo4jStore` — remote Neo4j / Aura backend.
"""

from biograph.graph.knowledge_graph import KnowledgeGraph

__all__ = ["KnowledgeGraph"]


def get_kuzu_store(*args, **kwargs):
    from biograph.graph.kuzu_store import KuzuStore

    return KuzuStore(*args, **kwargs)


def get_neo4j_store(*args, **kwargs):
    from biograph.graph.neo4j_store import Neo4jStore

    return Neo4jStore(*args, **kwargs)
