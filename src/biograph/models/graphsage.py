"""GraphSAGE — inductive representation learning on the knowledge graph.

Why GraphSAGE here?
-------------------
The notebook represents every entity by the *text* embedding of its name. Two
problems follow: (1) the representation ignores the entity's position in the
graph, and (2) a brand-new entity only ever gets a text vector, never a
structural one. GraphSAGE (Hamilton, Ying & Leskovec, NeurIPS 2017) fixes both
by learning **aggregator functions** that build a node's embedding from its
*neighbourhood's* features:

    h_v^{k} = σ( W · CONCAT( h_v^{k-1},  AGG_{u∈N(v)} h_u^{k-1} ) )

Because the aggregators are shared across all nodes, the model is *inductive*:
it produces embeddings for nodes (or whole subgraphs) unseen at training time —
exactly what a continuously growing literature graph needs.

We expose two training objectives:
* **Unsupervised link prediction** — the original GraphSAGE loss (neighbours
  should embed close, random pairs far apart). Yields general-purpose structural
  embeddings and, as a by-product, a link scorer for *hypothesis generation*
  (predicting plausible but not-yet-recorded Drug→Gene interactions).
* **Supervised node classification** — predict the entity type; useful as a
  sanity probe that the structure carries signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

from biograph.graph.knowledge_graph import KnowledgeGraph
from biograph.models.pyg_export import PyGBundle, kg_to_pyg


class GraphSAGE(nn.Module):
    """A stack of SAGEConv layers producing L2-normalised node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, aggr: str = "mean"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.normalize(x, p=2, dim=-1)


@dataclass
class LinkHypothesis:
    head: str
    tail: str
    score: float
    head_label: str
    tail_label: str


class GraphSAGETrainer:
    """Trains a :class:`GraphSAGE` model on a :class:`KnowledgeGraph`."""

    def __init__(self, kg: KnowledgeGraph, *, feature_dim: int = 768, hidden_dim: int = 256,
                 out_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 aggr: str = "mean", lr: float = 0.01, device: str | None = None):
        self.kg = kg
        self.bundle: PyGBundle = kg_to_pyg(kg, feature_dim=feature_dim)
        self.data = self.bundle.data
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data = self.data.to(self.device)
        self.model = GraphSAGE(
            in_dim=self.data.x.size(1), hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, dropout=dropout, aggr=aggr,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------ objectives
    def train_unsupervised(self, epochs: int = 100, verbose: bool = True) -> list[float]:
        """Link-prediction loss: score(u,v) high for edges, low for random pairs."""
        edge_index = self.data.edge_index
        losses = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            z = self.model(self.data.x, edge_index)
            pos = edge_index
            neg = negative_sampling(edge_index, num_nodes=self.data.num_nodes,
                                    num_neg_samples=pos.size(1))
            pos_score = (z[pos[0]] * z[pos[1]]).sum(dim=-1)
            neg_score = (z[neg[0]] * z[neg[1]]).sum(dim=-1)
            loss = (-F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"  epoch {epoch:>4}/{epochs}  loss={loss:.4f}")
        return losses

    def train_node_classification(self, epochs: int = 100, verbose: bool = True) -> list[float]:
        """Auxiliary objective: predict entity type from structure+features."""
        head = nn.Linear(self.model.convs[-1].out_channels,
                         int(self.data.y.max()) + 1).to(self.device)
        opt = torch.optim.Adam(list(self.model.parameters()) + list(head.parameters()), lr=0.01)
        losses = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            z = self.model(self.data.x, self.data.edge_index)
            logits = head(z)
            loss = F.cross_entropy(logits, self.data.y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if verbose and (epoch % max(1, epochs // 10) == 0):
                acc = (logits.argmax(-1) == self.data.y).float().mean()
                print(f"  epoch {epoch:>4}/{epochs}  loss={loss:.4f}  acc={acc:.3f}")
        return losses

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def node_embeddings(self) -> dict[str, "torch.Tensor"]:
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index).cpu()
        return {name: z[i] for i, name in enumerate(self.bundle.node_names)}

    def write_embeddings_to_graph(self) -> None:
        """Persist learned structural embeddings back onto the KnowledgeGraph nodes."""
        emb = self.node_embeddings()
        g = self.kg.to_networkx()
        for name, vec in emb.items():
            g.nodes[name]["sage_embedding"] = vec.numpy()

    @torch.no_grad()
    def predict_links(
        self,
        head_label: str = "Drug",
        tail_label: str = "Gene",
        top_k: int = 10,
        exclude_existing: bool = True,
    ) -> list[LinkHypothesis]:
        """Rank non-existent ``head_label -> tail_label`` pairs by embedding score.

        This is *hypothesis generation*: high-scoring pairs that are NOT already
        edges are candidate interactions worth experimental follow-up.
        """
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index)
        g = self.kg.to_networkx()
        names = self.bundle.node_names
        existing = {(u, v) for u, v, _ in self.kg.edge_triples()}

        heads = [n for n in names if g.nodes[n].get("label") == head_label]
        tails = [n for n in names if g.nodes[n].get("label") == tail_label]

        scored: list[LinkHypothesis] = []
        for h in heads:
            hi = self.bundle.idx(h)
            for t in tails:
                if h == t:
                    continue
                if exclude_existing and ((h, t) in existing or (t, h) in existing):
                    continue
                ti = self.bundle.idx(t)
                s = float(torch.sigmoid((z[hi] * z[ti]).sum()))
                scored.append(LinkHypothesis(h, t, s, head_label, tail_label))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(),
                    "node_names": self.bundle.node_names}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
