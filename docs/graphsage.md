# GraphSAGE — Inductive Representation Learning on the Knowledge Graph

> Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs.*
> NeurIPS 2017. arXiv:1706.02216

## Motivation

The notebook represents every entity by the **text embedding of its name**. That
has two structural blind spots:

1. **Topology is ignored.** `EGFR` and `KRAS` may sit in very different graph
   neighbourhoods (different drugs, mutations, diseases), yet if their names
   embed similarly the model treats them as interchangeable.
2. **New entities are second-class.** A freshly ingested paper introduces nodes
   that only ever get a text vector — never a representation informed by how they
   connect to the existing graph.

**GraphSAGE** fixes both. Instead of learning one embedding per node
(*transductive*, e.g. node2vec), it learns **aggregator functions** shared across
all nodes:

```
h_v^{k} = σ( W^k · CONCAT( h_v^{k-1},  AGG_{u ∈ N(v)} h_u^{k-1} ) )
```

- `h_v^0` = the node's input features (here: the PubMedBERT text embedding, plus
  a normalised degree and a one-hot entity-type augmentation so featureless nodes
  remain distinguishable).
- `AGG` = a permutation-invariant aggregator (`mean` by default; `max` / `lstm`
  also supported by `SAGEConv`).
- After `K` layers, `h_v^K` summarises the node's `K`-hop neighbourhood.

Because the *functions* are learned (not the embeddings), the model is
**inductive**: it computes embeddings for nodes and subgraphs unseen at training
time — exactly what a continuously growing literature graph needs.

## Implementation (`biograph.models`)

### `pyg_export.kg_to_pyg`
Converts the `KnowledgeGraph` into a PyTorch-Geometric `Data` object:
- `x` — node feature matrix (text embedding ⊕ degree ⊕ one-hot type),
- `edge_index` — **symmetrised** connectivity (GraphSAGE aggregates over
  undirected neighbourhoods even though the KG is directed),
- `y` — entity-type class id (for the optional classification probe).

The row-index ↔ entity-name mapping is preserved so learned vectors can be
written back onto the graph (`node["sage_embedding"]`).

### `graphsage.GraphSAGE`
A stack of `SAGEConv` layers with ReLU + dropout, producing **L2-normalised**
output embeddings (so a dot product is a cosine similarity).

### `graphsage.GraphSAGETrainer`
Two objectives:

- **`train_unsupervised`** — the original GraphSAGE loss via negative sampling:
  neighbouring nodes should score high, random pairs low.
  ```
  L = −log σ(zᵤ·zᵥ)  −  E_{n∼P_neg} log σ(−zᵤ·zₙ)
  ```
  Yields general-purpose structural embeddings and, as a by-product, a **link
  scorer**.
- **`train_node_classification`** — predicts entity type; a sanity probe that
  structure carries signal.

## Hypothesis generation

The unsupervised scorer doubles as a **link-prediction** engine. `predict_links`
ranks all `head_label → tail_label` pairs (default `Drug → Gene`) that are **not
already edges** by `σ(z_head · z_tail)`. High-scoring absent links are candidate
interactions worth wet-lab follow-up:

```bash
biograph-train-sage --epochs 100 --predict Drug Gene --top-k 10
```

On the bundled corpus this surfaces e.g. `Capmatinib → HGF` and `Gefitinib →
MET` — both consistent with the known MET/HGF bypass-resistance axis, despite
never appearing as explicit edges. This is the difference between *retrieving
what the literature says* and *predicting what it implies*.

## Feeding G-Retriever

`GraphSAGETrainer.write_embeddings_to_graph()` stores the learned vectors on the
nodes, and `GRetriever.graph_soft_prompt(subgraph, trainer)` mean-pools them into
a single vector — the **soft graph token** for graph prompt tuning (see
[`g_retriever.md`](g_retriever.md)).

## Scaling notes

The demo trains full-batch (the KG is tiny). For large graphs, swap in PyG's
`NeighborLoader` with the `graphsage.neighbor_samples` fan-out from
`configs/default.yaml` — GraphSAGE was designed precisely for
mini-batch neighbour sampling on graphs too large to fit in memory.
