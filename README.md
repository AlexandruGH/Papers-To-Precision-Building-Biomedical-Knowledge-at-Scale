# biograph — From Papers to Precision

**A biomedical Knowledge-Graph + GraphRAG research pipeline, extended with GraphSAGE and G-Retriever.**

`biograph` turns unstructured scientific literature into a queryable knowledge
graph and layers three complementary retrieval paradigms on top of it. It began
life as a workshop notebook and has been
refactored into a modular, tested Python package with two new graph-native
reasoning components.

<p align="center"><i>PubMed → GLiNER extraction → entity resolution → knowledge graph → {Hybrid RAG · GraphSAGE · G-Retriever}</i></p>

---

## Why this project exists

Classical **Vector RAG** chunks documents and retrieves the chunks most similar
to a query. It fails on *distant knowledge*: if a gene is described on page 1 of
one paper and a drug that acts on it appears in a different paper entirely, a
similarity search will never place them in the same context window. The
connection is real but **latent across documents**.

A **knowledge graph** makes that connection explicit. Once every mention of
`EGFR` collapses onto a single node, a path

```
(Osimertinib) --inhibits--> (EGFR) --causes--> (C797S) --confers--> (resistance)
```

can be *traversed*, even though no single sentence contains all four entities.
This repository implements the full construction pipeline and then asks: **what
is the best way to retrieve and reason over such a graph?** We compare three
answers.

| Paradigm | Retrieves | Strength | Module |
|---|---|---|---|
| **Hybrid GraphRAG** | 1-hop neighbours + similar text snippets | Simple, provenance-rich | `biograph.retrieval.hybrid_rag` |
| **GraphSAGE** | learned structural node embeddings | *Inductive*; enables link prediction / hypothesis generation | `biograph.models.graphsage` |
| **G-Retriever** | a connected, prize-optimal *subgraph* (PCST) | Preserves multi-hop relational context for the LLM | `biograph.retrieval.g_retriever` |

---

## Architecture

```
                ┌───────────────┐     ┌────────────────┐     ┌─────────────────────┐
   PubMed  ──▶  │  ingestion    │ ──▶ │  extraction    │ ──▶ │ entity resolution   │
  (Entrez)      │  (Biopython)  │     │  (GLiNER, 0-shot)│    │ (embedding linking) │
                └───────────────┘     └────────────────┘     └──────────┬──────────┘
                                                                        │  triplets
                                                             ┌──────────▼──────────┐
                                                             │  KnowledgeGraph      │
                                                             │  (networkx core;     │
                                                             │   Kùzu / Neo4j       │
                                                             │   persistence)       │
                                                             └──────────┬──────────┘
                        ┌──────────────────────┬───────────────────────┼────────────────────────┐
                        ▼                       ▼                       ▼                        ▼
                 Hybrid GraphRAG          GraphSAGE (GNN)          G-Retriever              analytics /
              vector + structural   inductive embeddings +   PCST subgraph + LLM         multi-hop / viz
                                      link prediction
```

The **`KnowledgeGraph`** (an in-memory `networkx.MultiDiGraph`) is the
backend-agnostic hub: every downstream component reads from it, so the GNN and
retrievers never have to speak Cypher. Kùzu (embedded) and Neo4j (remote) are
optional persistence/Cypher layers.

---

## Quickstart

```bash
# 1. Install (core deps only → runs the full offline demo, no downloads)
pip install -e .

# 2. Build the graph and query it (uses the bundled EGFR/lung-cancer corpus)
biograph-pipeline

# 3. G-Retriever question answering (PCST subgraph + LLM synthesis)
biograph-gretriever -q "How does resistance to Osimertinib arise?"

# 4. Train GraphSAGE and generate novel Drug→Gene interaction hypotheses
pip install -e ".[gnn]"
biograph-train-sage --epochs 100 --predict Drug Gene
```

Everything above runs **offline** with zero credentials via a bundled sample
corpus and a dependency-free hashed text encoder. Answer synthesis is served by a
**local Qwen model through Ollama** — no cloud API key anywhere. To go live:

```bash
pip install -e ".[all]"                       # PubMed + GLiNER + GNN + LLM + viz
cp configs/env.example .env                    # set ENTREZ_EMAIL (+ optional overrides)

# local, key-free LLM for answer synthesis
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b                           # or qwen3:0.6b (smaller) / qwen3:1.7b

biograph-pipeline --online --query "(Lung Cancer) AND (EGFR) AND (Resistance)" --max-results 50
biograph-gretriever --online -q "..." --encoder pritamdeka/S-PubMedBert-MS-MARCO
```

> **Model tag:** default is `qwen3.5:2b`. Set any tag you have pulled with
> `BIOGRAPH_LLM_MODEL=...` or `g_retriever.llm_model` in the config
> (`ollama list` shows what's available locally).

### Optional dependency groups

| Extra | Enables |
|---|---|
| `nlp` | GLiNER extraction, PubMedBERT/cross-encoder text embeddings |
| `ingest` | PubMed/NCBI Entrez ingestion (Biopython) |
| `graph` | Kùzu (embedded) + Neo4j (remote) persistence |
| `gnn` | GraphSAGE + G-Retriever soft-prompt (PyTorch Geometric, `pcst_fast`) |
| `llm` | Local Qwen answer synthesis via **Ollama + Instructor** (no API key) |
| `viz` | pyvis interactive graphs + seaborn dashboards |

---

## The three retrieval paradigms

### 1. Hybrid GraphRAG (baseline, from the notebook)
Fuses **structural context** (the query entity's validated relations) with
**semantic context** (literature snippets) into a single, provenance-cited
prompt. See [`docs/architecture.md`](docs/architecture.md).

### 2. GraphSAGE — inductive representation learning
Text embeddings ignore an entity's *position* in the graph. GraphSAGE (Hamilton
et al., NeurIPS 2017) learns aggregator functions that build a node's embedding
from its neighbourhood, so representations become structure-aware and — crucially
— *inductive* (new nodes get embeddings without retraining). We train it with the
unsupervised link-prediction objective and reuse the learned scorer for
**hypothesis generation**: high-scoring `Drug→Gene` pairs that are *not yet edges*
are candidate interactions worth experimental follow-up. See
[`docs/graphsage.md`](docs/graphsage.md).

### 3. G-Retriever — retrieve a subgraph, not a list
G-Retriever (He et al., NeurIPS 2024) scores nodes and edges against the query,
then solves a **Prize-Collecting Steiner Tree** to extract a *connected,
size-controlled subgraph* — retaining the relational glue that makes multi-hop
answers possible, which naive top-k retrieval discards. The subgraph is
textualised and passed to an LLM; an optional GraphSAGE **soft graph token**
(graph prompt tuning) can be prepended. See [`docs/g_retriever.md`](docs/g_retriever.md).

---

## Repository layout

```
biomed-graphrag/
├── src/biograph/
│   ├── config.py, schema.py        # typed data contracts + config/secrets
│   ├── pipeline.py                 # end-to-end orchestrator
│   ├── ingestion/                  # PubMed (Entrez) + offline sample corpus
│   ├── extraction/                 # GLiNER zero-shot RE + entity resolution
│   ├── embeddings/                 # bi-/cross-encoder wrappers + theory demos
│   ├── graph/                      # KnowledgeGraph (networkx) + Kùzu + Neo4j
│   ├── models/                     # GraphSAGE + PyG export  ★ new
│   ├── retrieval/                  # hybrid RAG + G-Retriever ★ new
│   ├── analytics/                  # centrality + multi-hop reasoning
│   ├── viz/                        # pyvis interactive graphs
│   ├── llm.py                      # pluggable Claude answer synthesis
│   └── cli/                        # biograph-{pipeline,train-sage,gretriever}
├── configs/                        # default.yaml + env.example
├── docs/                           # research notes per component
├── tests/                          # offline smoke tests (pytest)
└── QIAGEN_Hackathon.ipynb          # original notebook (kept for provenance)
```

## Testing

```bash
pip install -e ".[all]" && pytest -q
```

Tests run fully offline; the GraphSAGE test self-skips if `torch`/`torch_geometric`
are absent, and the PCST retriever falls back to a shortest-path Steiner
heuristic if `pcst_fast` is not installed.

## References

- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017. arXiv:1706.02216
- He et al. *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and QA*. NeurIPS 2024. arXiv:2402.07630
- Zaratiana et al. *GLiNER: Generalist Model for NER using Bidirectional Transformer*. 2023. arXiv:2311.08526
- He et al. *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. ICLR 2021. arXiv:2006.03654

## License

MIT. Biomedical relations extracted by this pipeline are research signals, **not
clinical advice**.
