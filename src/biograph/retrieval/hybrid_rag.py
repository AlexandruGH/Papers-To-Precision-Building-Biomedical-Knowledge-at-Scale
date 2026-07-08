"""Hybrid GraphRAG: fuse structural graph context with semantic vector context.

Port of the notebook's ``hybrid_rag_retrieval``. Two retrieval pathways feed one
prompt:

1. **Structural** — 1-hop relations of the query entity from the graph
   (validated, typed facts).
2. **Semantic**   — literature snippets most similar to the query text
   (raw evidence, recall-oriented).

The union is far stronger than either alone: the graph supplies precision and
provenance, the vector store supplies recall and nuance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from biograph.graph.knowledge_graph import KnowledgeGraph
from biograph.llm import LLMClient
from biograph.schema import Article


@dataclass
class HybridRAG:
    kg: KnowledgeGraph
    articles: Sequence[Article] = field(default_factory=list)
    llm: LLMClient | None = None

    def graph_context(self, entity: str, limit: int = 5) -> list[str]:
        g = self.kg.to_networkx()
        facts: list[tuple[str, float]] = []
        for node in g.nodes:
            if entity.lower() not in node.lower():
                continue
            for _, tgt, d in g.out_edges(node, data=True):
                facts.append((f"- {node} {d.get('relation')} {tgt} "
                              f"(status={d.get('status')}, conf={d.get('score', 0):.2f})",
                              d.get("score", 0.0)))
            for src, _, d in g.in_edges(node, data=True):
                facts.append((f"- {src} {d.get('relation')} {node} "
                              f"(status={d.get('status')}, conf={d.get('score', 0):.2f})",
                              d.get("score", 0.0)))
        facts.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in facts[:limit]]

    def vector_context(self, entity: str, limit: int = 2) -> list[str]:
        # Lightweight lexical fallback (keeps the demo dependency-free). Swap in a
        # FAISS/BiEncoder store for production semantic recall.
        hits = []
        for art in self.articles:
            if entity.lower() in (art.text or "").lower():
                hits.append(f"PMID {art.pmid}: {art.text[:300]}…")
        return hits[:limit]

    def build_prompt(self, entity: str, question: str) -> str:
        graph_ctx = self.graph_context(entity)
        vec_ctx = self.vector_context(entity)
        return f"""QUESTION: {question}

STRUCTURAL KNOWLEDGE GRAPH (validated relationships):
{chr(10).join(graph_ctx) if graph_ctx else 'No graph relations found.'}

SEMANTIC EVIDENCE (retrieved literature snippets):
{chr(10).join(vec_ctx) if vec_ctx else 'No matching snippets found.'}

Answer the question using ONLY the context above. Cite PMIDs.
"""

    def answer(self, entity: str, question: str) -> str:
        prompt = self.build_prompt(entity, question)
        client = self.llm or LLMClient()
        resp = client.complete(prompt)
        if resp.dry_run:
            print("[dry-run: local Ollama LLM unavailable — returning assembled prompt]\n")
        return resp.text


def hybrid_rag_retrieval(kg: KnowledgeGraph, articles, query_entity: str, query_text: str) -> str:
    """Functional entry point mirroring the notebook signature."""
    return HybridRAG(kg=kg, articles=list(articles)).answer(query_entity, query_text)
