# Architecture & Pipeline

This document walks the data through the pipeline, stage by stage, and explains
the design decisions behind each module.

## 0. Data contracts (`biograph.schema`)

Three dataclasses flow through the whole system:

- **`Article`** — a PubMed record (`pmid`, `title`, `text`, `year`, `journal`).
- **`Entity`** — a canonical graph node (`name`, `label`, optional `embedding`).
- **`Triplet`** — a `(head)-[relation]->(tail)` fact with `score`, `status`
  (evidence strength), `pmid`, and `context` provenance.

The original notebook passed loosely-typed dicts between cells; lifting the
recurring shapes into dataclasses gives every stage a stable, documented
interface and makes the retrievers testable in isolation.

## 1. Ingestion (`biograph.ingestion`)

`PubMedClient` wraps `Bio.Entrez` with NCBI etiquette (contact email, optional
API key for the 10 req/s tier, polite throttling) and returns typed `Article`s.
`sample_data` ships an offline EGFR/lung-cancer corpus + pre-extracted triplets
so every downstream component runs without network or model downloads — the
backbone of reproducible demos and CI.

## 2. Extraction (`biograph.extraction.gliner_extractor`)

**GLiNER** performs *zero-shot* NER + relation extraction: we hand it the entity
types (`Disease/Gene/Drug/Mutation`) and relation types (`treats/inhibits/
causes/targets`) and it returns typed triplets — no task-specific fine-tuning.
Each triplet is tagged with an **evidence status** (`Experimental`, `Clinical`,
`Simulation`, `Mentioned`) inferred from cheap lexical cues in the abstract, so
downstream reasoning can weight a preclinical mouse result differently from an
approved standard of care.

## 3. Entity resolution (`biograph.extraction.entity_resolution`)

The single most important step for graph quality. Surface forms like `EGFR`,
`EGFR receptor`, and `epidermal growth factor receptor` must collapse onto one
node, or multi-hop reasoning fractures across duplicates.

We embed each surface form and merge mentions whose cosine similarity to an
existing canonical vector exceeds a threshold (default 0.92). The encoder is
pluggable: a PubMedBERT bi-encoder in production, or a dependency-free hashed
character-n-gram vectoriser offline (lower recall, but deterministic and
zero-install).

## 4. Knowledge graph (`biograph.graph`)

`KnowledgeGraph` wraps `networkx.MultiDiGraph` and is the **canonical
structure**: triplets are ingested here first, and GraphSAGE + both retrievers
read from it directly. Node attributes carry `label`, `embedding`, and the set
of source `pmids`; edge attributes carry `relation`, `status`, `score`, `pmid`,
`context`.

Two optional persistence backends provide Cypher and durability:
- **`KuzuStore`** — embedded, file-based ("SQLite for graphs"); default.
- **`Neo4jStore`** — remote Neo4j/Aura; stores entity embeddings on nodes so a
  single DB answers both structural (Cypher) and semantic (vector) queries.

## 5. Retrieval & reasoning

- **Hybrid GraphRAG** (`retrieval.hybrid_rag`) — the notebook's baseline: 1-hop
  structural context ∪ semantic snippets → one cited prompt.
- **Analytics** (`analytics.inferences`) — degree centrality ("high-impact
  genes"), bounded multi-hop path enumeration, and `Drug→X→target` resistance
  chains, all as pure `networkx` traversals.
- **GraphSAGE** — see [`graphsage.md`](graphsage.md).
- **G-Retriever** — see [`g_retriever.md`](g_retriever.md).

## 6. LLM synthesis (`biograph.llm`)

A thin, pluggable layer so retrieval stays model-agnostic. The backend is a
**local Ollama** server (OpenAI-compatible endpoint) driven through the
**Instructor** library, so answers are returned as a *validated Pydantic object*
(`BiomedAnswer`: `answer` + `cited_pmids` + `reasoning`) rather than free text —
no cloud API key anywhere. Default model is a small Qwen (`qwen2.5:1.5b`;
configurable, e.g. `qwen2.5:0.5b`/`qwen2.5:3b`). If the server is unreachable it
runs in **dry-run** mode and returns the assembled prompt, so demos need nothing
running.

## Design principles

1. **One canonical graph, many views.** Cypher backends are optional; the GNN
   and retrievers speak `networkx`.
2. **Offline-first.** Every stage has a dependency-free fallback so the whole
   pipeline is runnable and testable with `pip install -e .` alone.
3. **Typed provenance everywhere.** Every fact carries its PMID and evidence
   status from extraction through to the final prompt.
