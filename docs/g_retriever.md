# G-Retriever — Subgraph Retrieval for Graph QA

> He, Tian, Chen, et al. *G-Retriever: Retrieval-Augmented Generation for Textual
> Graph Understanding and Question Answering.* NeurIPS 2024. arXiv:2402.07630

## The problem with top-k retrieval

Naive graph RAG retrieves the *k* nodes (or edges) most similar to the query and
dumps them into the prompt. But a set of disconnected high-similarity nodes has
**lost its relational structure** — the very thing a graph was supposed to
provide. If the answer requires the path

```
Osimertinib → MET (bypass) → HGF
```

top-k might retrieve `Osimertinib` and `HGF` (both query-relevant) while dropping
`MET` (individually less similar), severing the reasoning chain.

## The G-Retriever idea: retrieve a *connected subgraph*

G-Retriever reframes retrieval as an optimisation: find the connected subgraph
that **maximises total query-relevance (prize) minus size (cost)**. That is a
**Prize-Collecting Steiner Tree (PCST)** problem.

### Pipeline (`biograph.retrieval.g_retriever.GRetriever`)

1. **Index.** Every node is textualised as `"name (type)"` and every edge as
   `"head relation tail"`, then embedded in the *same space* as the query. The
   encoder is pluggable (`encode_fn`); offline it defaults to a hashed
   character-n-gram encoder so no downloads are needed.

2. **Score → prizes.** Cosine-similarity to the query, then a **rank-based
   prize**: the top-*k* nodes receive prizes `k, k−1, …, 1`; likewise for edges.
   Rank-based (not raw-similarity) prizes make the sparsity trade-off against
   edge cost interpretable and stable.

3. **Solve PCST.** We use the classic reduction that lets a *node*-prize solver
   handle *edge* prizes: an edge with prize `p > cost` is split through a virtual
   node carrying prize `p − cost`; cheaper edges just carry cost `cost − p`. The
   `pcst_cost` knob (default 0.5) controls sparsity — higher cost ⇒ smaller, more
   focused subgraphs. Solved exactly-ish with the Goemans–Williamson
   approximation in [`pcst_fast`]. Without that library, a **shortest-path
   Steiner heuristic** stitches the prized nodes/edges into one connected
   component — same contract, lower fidelity.

4. **Textualise.** The subgraph is serialised as two CSV-style tables (a `NODES`
   table and an `EDGES` table with `status`/`pmid` provenance) — compact and easy
   for an LLM to parse.

5. **Generate.** The tables + question form the prompt for a **local LLM** —
   a small Qwen served by **Ollama** and driven through **Instructor**
   (`biograph.llm`), so the model returns a *validated* `BiomedAnswer`
   (`answer` + `cited_pmids` + `reasoning`), not free text. No API key. If the
   Ollama server is not running, the retriever runs in dry-run mode and returns
   the prompt itself.

```bash
ollama pull qwen2.5:1.5b        # one-time: pull the local model (no API key)
biograph-gretriever -q "How does Osimertinib resistance arise via bypass signalling?"
```

## Optional: the GNN soft prompt (graph prompt tuning)

The full G-Retriever architecture also encodes the retrieved subgraph with a GNN
and prepends the pooled vector to the LLM input as a **soft token**, giving the
model access to structural signal the text serialisation flattens away.

`GRetriever.graph_soft_prompt(subgraph, trainer)` implements the graph side:
mean-pooling a trained **GraphSAGE** ([`graphsage.md`](graphsage.md)) embedding
over the retrieved nodes. Projecting that vector into a frozen LLM's token space
and training the projector (LoRA-style) is the natural next step for a
fine-tuning setup; the retrieval + textualisation path already works
prompt-only against any hosted LLM.

## Why it complements the other retrievers

| | Hybrid GraphRAG | G-Retriever |
|---|---|---|
| Retrieval unit | 1-hop neighbourhood of one entity | connected subgraph across many |
| Multi-hop paths | only if within 1 hop | **preserved by construction** |
| Size control | fixed limit | **prize/cost optimisation (PCST)** |
| Entry point | a named entity | a free-text question |

Hybrid GraphRAG answers *"what is directly known about X?"*. G-Retriever answers
*"assemble the minimal evidence subgraph that bears on this question"* — the
better fit when the question spans several entities and hops.

[`pcst_fast`]: https://github.com/fraenkel-lab/pcst_fast
