"""Convert an in-memory :class:`KnowledgeGraph` into PyTorch-Geometric tensors.

The GNN sees the graph as:
* ``x``          — node feature matrix (text embeddings; zero-rows get a small
                   degree/one-hot-type augmentation so featureless nodes are still
                   distinguishable),
* ``edge_index`` — 2 x E connectivity (we symmetrise: GraphSAGE aggregates over
                   undirected neighbourhoods even though the KG is directed),
* ``y``          — entity-type class id (Disease/Gene/Drug/Mutation) for the
                   optional node-classification objective.

The mapping between tensor row indices and entity names is preserved in
``PyGBundle.node_names`` so learned embeddings can be written back to the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from biograph.graph.knowledge_graph import KnowledgeGraph


@dataclass
class PyGBundle:
    data: "object"                 # torch_geometric.data.Data
    node_names: list[str]          # row index -> entity name
    label_names: list[str]         # class id -> entity-type string
    name_to_idx: dict[str, int]

    def idx(self, name: str) -> Optional[int]:
        return self.name_to_idx.get(name)


def kg_to_pyg(kg: KnowledgeGraph, feature_dim: int = 768, augment: bool = True) -> PyGBundle:
    import torch
    from torch_geometric.data import Data

    node_names = kg.nodes()
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    n = len(node_names)

    # --- node features ---
    X, order = kg.embedding_matrix(feature_dim)
    assert order == node_names
    if augment:
        g = kg.to_networkx()
        # append normalised degree + one-hot entity type so zero-embedding nodes
        # are not identical to the model.
        labels = sorted({d.get("label", "Entity") for _, d in g.nodes(data=True)})
        lab_idx = {l: i for i, l in enumerate(labels)}
        extra = np.zeros((n, 1 + len(labels)), dtype=np.float32)
        max_deg = max((d for _, d in g.degree()), default=1) or 1
        for i, name in enumerate(node_names):
            extra[i, 0] = g.degree(name) / max_deg
            extra[i, 1 + lab_idx[g.nodes[name].get("label", "Entity")]] = 1.0
        X = np.hstack([X, extra])

    x = torch.tensor(X, dtype=torch.float)

    # --- edges (symmetrised) ---
    src, dst = [], []
    for u, v, _ in kg.edge_triples():
        i, j = name_to_idx[u], name_to_idx[v]
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)

    # --- labels ---
    g = kg.to_networkx()
    label_names = sorted({g.nodes[nm].get("label", "Entity") for nm in node_names})
    lab_to_id = {l: i for i, l in enumerate(label_names)}
    y = torch.tensor([lab_to_id[g.nodes[nm].get("label", "Entity")] for nm in node_names],
                     dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = n
    return PyGBundle(data=data, node_names=node_names, label_names=label_names,
                     name_to_idx=name_to_idx)
