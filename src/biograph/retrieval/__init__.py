"""Retrieval paradigms over the knowledge graph.

* :func:`hybrid_rag_retrieval` — vector snippets + structural relations (notebook).
* :class:`GRetriever`          — PCST subgraph retrieval + LLM synthesis (this project).
"""

from biograph.retrieval.hybrid_rag import HybridRAG, hybrid_rag_retrieval
from biograph.retrieval.g_retriever import GRetriever, RetrievedSubgraph

__all__ = ["HybridRAG", "hybrid_rag_retrieval", "GRetriever", "RetrievedSubgraph"]
