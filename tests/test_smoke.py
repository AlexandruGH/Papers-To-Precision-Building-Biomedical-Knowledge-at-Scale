"""Offline smoke tests — no network, no large model downloads.

Run with:  pytest -q
Optional extras (torch/pcst_fast) are skipped gracefully when absent.
"""

from __future__ import annotations

import importlib.util

import pytest

from biograph.graph.knowledge_graph import KnowledgeGraph
from biograph.pipeline import Pipeline
from biograph.retrieval.g_retriever import GRetriever
from biograph.retrieval.hybrid_rag import HybridRAG


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.fixture(scope="module")
def result():
    return Pipeline().run(offline=True)


def test_pipeline_builds_graph(result):
    assert result.kg.num_nodes > 5
    assert result.kg.num_edges > 5
    s = result.kg.summary()
    assert set(s["labels"]) & {"Drug", "Gene", "Disease", "Mutation"}


def test_entity_resolution_runs(result):
    # merges is a dict (possibly empty); must not raise
    assert isinstance(result.merges, dict)


def test_multi_hop(result):
    from biograph.analytics.inferences import multi_hop_pathways

    paths = multi_hop_pathways(result.kg, "Osimertinib", max_hops=2)
    assert all(len(p) <= 2 for p in paths)


def test_hybrid_rag_prompt(result):
    rag = HybridRAG(kg=result.kg, articles=result.articles)
    prompt = rag.build_prompt("EGFR", "What targets EGFR?")
    assert "KNOWLEDGE GRAPH" in prompt


def test_g_retriever_offline(result):
    gr = GRetriever(result.kg)  # hashed encoder; local LLM only if Ollama is up
    out = gr.answer("How does Osimertinib resistance arise?", top_k_nodes=8, top_k_edges=12)
    assert not out["subgraph"].is_empty()
    assert "NODES:" in out["subgraph"].textualize()
    # Without a reachable Ollama server the client is in dry-run mode and echoes
    # the prompt; if a server IS up, it returns a non-empty synthesised answer.
    if out["dry_run"]:
        assert out["answer"] == out["prompt"]
    else:
        assert out["answer"].strip()


def test_g_retriever_connected(result):
    gr = GRetriever(result.kg)
    sub = gr.retrieve("KRAS lung cancer", top_k_nodes=6, top_k_edges=10)
    # every edge endpoint must be present in the node list (well-formed subgraph)
    for h, _, t in sub.edges:
        assert h in sub.nodes and t in sub.nodes


@pytest.mark.skipif(not (_have("torch") and _have("torch_geometric")),
                    reason="torch / torch_geometric not installed")
def test_graphsage_trains_and_predicts(result):
    from biograph.models.graphsage import GraphSAGETrainer

    trainer = GraphSAGETrainer(result.kg, feature_dim=768, hidden_dim=32,
                               out_dim=16, num_layers=2)
    losses = trainer.train_unsupervised(epochs=5, verbose=False)
    assert len(losses) == 5
    hyps = trainer.predict_links("Drug", "Gene", top_k=3)
    assert all(0.0 <= h.score <= 1.0 for h in hyps)


def test_kg_from_triplets_roundtrip():
    from biograph.ingestion.sample_data import sample_triplets

    kg = KnowledgeGraph.from_triplets(sample_triplets())
    assert "EGFR" in kg.nodes()
