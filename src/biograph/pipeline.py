"""End-to-end pipeline: literature → triplets → resolved knowledge graph.

Ties the individual stages into one reproducible object. Two modes:

* ``offline=True`` (default for demos/CI) — uses the bundled sample corpus and
  pre-extracted triplets; no network, no model downloads.
* ``offline=False`` — fetches from PubMed and runs GLiNER extraction; requires
  the ``ingest`` and ``nlp`` extras.

The resulting :class:`KnowledgeGraph` feeds every retriever (hybrid, GraphSAGE,
G-Retriever).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from biograph.config import Config, Settings, load_config
from biograph.extraction.entity_resolution import EntityResolver
from biograph.graph.knowledge_graph import KnowledgeGraph
from biograph.ingestion.sample_data import sample_articles, sample_triplets
from biograph.schema import Article, Triplet


@dataclass
class PipelineResult:
    articles: list[Article]
    triplets: list[Triplet]
    kg: KnowledgeGraph
    merges: dict[str, str] = field(default_factory=dict)

    def describe(self) -> None:
        s = self.kg.summary()
        print(f"Knowledge graph: {s['nodes']} nodes, {s['edges']} edges")
        print(f"  entity types: {s['labels']}")
        print(f"  relations:    {s['relations']}")
        if self.merges:
            print(f"  entity-resolution merges: {len(self.merges)}")


class Pipeline:
    def __init__(self, config: Config | None = None, settings: Settings | None = None):
        self.config = config or load_config()
        self.settings = settings or Settings.from_env()

    def run(self, offline: bool = True, query: Optional[str] = None,
            max_results: Optional[int] = None, resolve: bool = True) -> PipelineResult:
        if offline:
            articles = sample_articles()
            triplets = sample_triplets()
        else:
            articles, triplets = self._run_online(query, max_results)

        merges: dict[str, str] = {}
        if resolve:
            resolver = self._build_resolver(offline)
            triplets, merges = resolver.resolve_triplets(triplets, verbose=True)

        kg = KnowledgeGraph.from_triplets(triplets)
        return PipelineResult(articles=articles, triplets=triplets, kg=kg, merges=merges)

    # ------------------------------------------------------------- internals
    def _run_online(self, query: Optional[str], max_results: Optional[int]):
        from biograph.extraction.gliner_extractor import GLiNERExtractor
        from biograph.ingestion.pubmed import PubMedClient

        ing = self.config.section("ingestion")
        query = query or ing.get("query")
        max_results = max_results or ing.get("max_results", 25)

        client = PubMedClient(self.settings, request_delay_s=ing.get("request_delay_s", 0.2))
        articles = client.search_and_fetch(query, max_results)

        ext = self.config.section("extraction")
        extractor = GLiNERExtractor(
            model_name=ext.get("gliner_model"),
            entity_labels=ext.get("entity_labels"),
            relation_labels=ext.get("relation_labels"),
            threshold=ext.get("threshold", 0.1),
        )
        triplets = extractor.extract_corpus(articles)
        return articles, triplets

    def _build_resolver(self, offline: bool) -> EntityResolver:
        emb = self.config.section("embeddings")
        threshold = emb.get("entity_link_threshold", 0.92)
        if offline:
            # dependency-free hashed encoder
            return EntityResolver(threshold=threshold)
        try:
            return EntityResolver.from_sentence_transformer(emb.get("model"), threshold)
        except Exception as exc:  # noqa: BLE001
            print(f"! sentence-transformers unavailable ({exc}); using hashed encoder.")
            return EntityResolver(threshold=threshold)
