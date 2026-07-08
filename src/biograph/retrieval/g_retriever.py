"""G-Retriever — retrieval-augmented generation over a textual graph.

Implements the method of He et al., *"G-Retriever: Retrieval-Augmented Generation
for Textual Graph Understanding and Question Answering"* (NeurIPS 2024,
arXiv:2402.07630), adapted to the biomedical knowledge graph.

Pipeline
--------
1. **Index** every node and edge as a short text and embed it in the same space
   as the query (any ``encode_fn``; a dependency-free hashed encoder is the
   default so this runs offline).
2. **Score** nodes and edges by cosine similarity to the question and turn the
   top-k into *prizes*.
3. **Retrieve a connected subgraph** by solving a **Prize-Collecting Steiner Tree
   (PCST)**: maximise collected prize minus edge cost. Unlike plain top-k
   retrieval this returns a *connected, size-controlled* subgraph — it keeps the
   relational glue that makes multi-hop answers possible while filtering noise.
   Uses ``pcst_fast`` when installed, with a shortest-path Steiner heuristic as
   a fallback.
4. **Textualise** the subgraph into node/edge tables.
5. **Generate** an answer with an LLM (see :mod:`biograph.llm`). Optionally a
   GraphSAGE-derived *soft graph token* can be prepended (graph prompt tuning) —
   see :meth:`GRetriever.graph_soft_prompt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from biograph.graph.knowledge_graph import KnowledgeGraph
from biograph.llm import LLMClient

EncodeFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class RetrievedSubgraph:
    nodes: list[str]
    edges: list[tuple[str, str, str]]  # (head, relation, tail)
    node_meta: dict[str, str] = field(default_factory=dict)  # name -> label
    edge_meta: list[dict] = field(default_factory=list)      # parallel to `edges`

    def textualize(self) -> str:
        """Render the subgraph as two CSV-like tables (the G-Retriever format)."""
        node_lines = ["node_id,name,type"]
        for i, n in enumerate(self.nodes):
            node_lines.append(f"{i},{n},{self.node_meta.get(n, 'Entity')}")
        edge_lines = ["src,relation,dst,status,pmid"]
        for (h, r, t), meta in zip(self.edges, self.edge_meta):
            edge_lines.append(f"{h},{r},{t},{meta.get('status','')},{meta.get('pmid','')}")
        return "NODES:\n" + "\n".join(node_lines) + "\n\nEDGES:\n" + "\n".join(edge_lines)

    def is_empty(self) -> bool:
        return not self.nodes


def _default_encoder() -> EncodeFn:
    from biograph.extraction.entity_resolution import _hashed_ngram_encoder

    return _hashed_ngram_encoder(dim=256)


class GRetriever:
    def __init__(
        self,
        kg: KnowledgeGraph,
        encode_fn: EncodeFn | None = None,
        llm: LLMClient | None = None,
        pcst_cost: float = 0.5,
    ):
        self.kg = kg
        self.g = kg.to_networkx()
        self.encode = encode_fn or _default_encoder()
        self.llm = llm
        self.pcst_cost = pcst_cost

        self.nodes: list[str] = kg.nodes()
        # Directed, de-duplicated edge list with attributes.
        self._edges: list[tuple[str, str, str]] = []
        self._edge_attr: list[dict] = []
        for u, v, d in kg.edge_triples():
            # stored as (head, relation, tail) to match RetrievedSubgraph.edges
            self._edges.append((u, d.get("relation", "related_to"), v))
            self._edge_attr.append(d)

        self._node_emb = self._embed_nodes()
        self._edge_emb = self._embed_edges()

    # ---------------------------------------------------------------- index
    def _embed_nodes(self) -> np.ndarray:
        texts = [f"{n} ({self.g.nodes[n].get('label', 'Entity')})" for n in self.nodes]
        return _normalize(self.encode(texts)) if texts else np.zeros((0, 1))

    def _embed_edges(self) -> np.ndarray:
        texts = [f"{h} {r} {t}" for (h, r, t) in self._edges]
        return _normalize(self.encode(texts)) if texts else np.zeros((0, 1))

    # ------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k_nodes: int = 15, top_k_edges: int = 30) -> RetrievedSubgraph:
        if not self.nodes:
            return RetrievedSubgraph([], [])
        q = _normalize(self.encode([query]))[0]
        node_sims = (self._node_emb @ q) if len(self._node_emb) else np.zeros(0)
        edge_sims = (self._edge_emb @ q) if len(self._edge_emb) else np.zeros(0)

        node_prizes = _rank_prizes(node_sims, top_k_nodes)
        edge_prizes = _rank_prizes(edge_sims, top_k_edges)

        try:
            keep_nodes, keep_edges = self._pcst(node_prizes, edge_prizes)
        except Exception as exc:  # noqa: BLE001 - fall back to a heuristic Steiner subgraph
            print(f"  (pcst_fast unavailable: {exc}; using shortest-path heuristic)")
            keep_nodes, keep_edges = self._heuristic_steiner(node_prizes, edge_prizes)

        return self._assemble(keep_nodes, keep_edges)

    def _pcst(self, node_prizes: np.ndarray, edge_prizes: np.ndarray):
        """Exact-ish PCST via ``pcst_fast`` with the edge-prize→virtual-node trick."""
        import pcst_fast

        name_to_idx = {n: i for i, n in enumerate(self.nodes)}
        # Build an undirected edge multiset keyed by endpoint pair; keep max prize.
        und: dict[tuple[int, int], float] = {}
        und_ref: dict[tuple[int, int], int] = {}
        for ei, (h, _, t) in enumerate(self._edges):
            a, b = sorted((name_to_idx[h], name_to_idx[t]))
            if a == b:
                continue
            if (a, b) not in und or edge_prizes[ei] > und[(a, b)]:
                und[(a, b)] = float(edge_prizes[ei])
                und_ref[(a, b)] = ei

        n = len(self.nodes)
        edges, costs, prizes = [], [], list(map(float, node_prizes))
        virtual_of: dict[int, int] = {}  # virtual node id -> original edge idx
        vnode = n
        cost_e = self.pcst_cost
        for (a, b), p in und.items():
            if p <= cost_e:
                edges.append([a, b])
                costs.append(cost_e - p)
            else:
                # split edge through a prized virtual node
                edges.append([a, vnode])
                costs.append(0.0)
                edges.append([vnode, b])
                costs.append(0.0)
                prizes.append(p - cost_e)
                virtual_of[vnode] = und_ref[(a, b)]
                vnode += 1

        if not edges:
            keep = [i for i in range(n) if node_prizes[i] > 0]
            return keep, []

        v_sel, e_sel = pcst_fast.pcst_fast(
            np.array(edges, dtype=np.int64), np.array(prizes, dtype=np.float64),
            np.array(costs, dtype=np.float64), -1, 1, "gw", 0,
        )
        keep_nodes = [i for i in v_sel if i < n]
        keep_edges = []
        for ei in e_sel:
            u, w = edges[ei]
            if u >= n or w >= n:  # touches a virtual node → maps to a real edge
                vn = u if u >= n else w
                if vn in virtual_of:
                    keep_edges.append(virtual_of[vn])
            else:
                # recover the original directed edge idx for this undirected pair
                a, b = sorted((u, w))
                if (a, b) in und_ref:
                    keep_edges.append(und_ref[(a, b)])
        return sorted(set(keep_nodes)), sorted(set(keep_edges))

    def _heuristic_steiner(self, node_prizes: np.ndarray, edge_prizes: np.ndarray):
        """Connected fallback: seed with prized nodes/edges, then stitch components
        together with shortest paths on the underlying undirected graph."""
        import networkx as nx

        name_to_idx = {n: i for i, n in enumerate(self.nodes)}
        keep_edges = [ei for ei in range(len(self._edges)) if edge_prizes[ei] > 0]
        seed_nodes = {i for i in range(len(self.nodes)) if node_prizes[i] > 0}
        for ei in keep_edges:
            h, _, t = self._edges[ei]
            seed_nodes.add(name_to_idx[h])
            seed_nodes.add(name_to_idx[t])

        und = self.g.to_undirected()
        seed_names = [self.nodes[i] for i in seed_nodes]
        connect_edges = set(keep_edges)
        for a, b in zip(seed_names, seed_names[1:]):
            try:
                path = nx.shortest_path(und, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for x, y in zip(path, path[1:]):
                for ei, (h, _, t) in enumerate(self._edges):
                    if {h, t} == {x, y}:
                        connect_edges.add(ei)
                        seed_nodes.add(name_to_idx[h])
                        seed_nodes.add(name_to_idx[t])
        return sorted(seed_nodes), sorted(connect_edges)

    def _assemble(self, keep_nodes, keep_edges) -> RetrievedSubgraph:
        node_names = [self.nodes[i] for i in keep_nodes]
        node_set = set(node_names)
        edges, edge_meta = [], []
        for ei in keep_edges:
            h, r, t = self._edges[ei]
            edges.append((h, r, t))
            edge_meta.append(self._edge_attr[ei])
            node_set.update([h, t])
        node_names = sorted(node_set)
        node_meta = {n: self.g.nodes[n].get("label", "Entity") for n in node_names}
        return RetrievedSubgraph(node_names, edges, node_meta, edge_meta)

    # -------------------------------------------------------------- generate
    def build_prompt(self, query: str, sub: RetrievedSubgraph) -> str:
        return (
            f"You are answering a question over a biomedical knowledge graph.\n"
            f"Use ONLY the retrieved subgraph below. Reason over multi-hop paths "
            f"when relevant, and cite PMIDs from the edge table.\n\n"
            f"=== RETRIEVED SUBGRAPH ===\n{sub.textualize()}\n\n"
            f"=== QUESTION ===\n{query}\n\n=== ANSWER ===\n"
        )

    def answer(self, query: str, top_k_nodes: int = 15, top_k_edges: int = 30) -> dict:
        sub = self.retrieve(query, top_k_nodes, top_k_edges)
        prompt = self.build_prompt(query, sub)
        client = self.llm or LLMClient()
        resp = client.complete(prompt)
        return {
            "query": query,
            "subgraph": sub,
            "prompt": prompt,
            "answer": resp.text,
            "dry_run": resp.dry_run,
            "model": resp.model,
        }

    # --------------------------------------------------- optional GNN prompt
    def graph_soft_prompt(self, sub: RetrievedSubgraph, trainer) -> "object":
        """Mean-pool a GraphSAGE embedding over the retrieved subgraph.

        In the full G-Retriever architecture this vector is projected and
        prepended to the LLM input as a *soft token* (graph prompt tuning),
        letting the model attend to graph structure the text serialisation
        loses. Requires a trained :class:`GraphSAGETrainer`.
        """
        import torch

        emb = trainer.node_embeddings()
        vecs = [emb[n] for n in sub.nodes if n in emb]
        if not vecs:
            return None
        return torch.stack(vecs).mean(dim=0)


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _rank_prizes(sims: np.ndarray, top_k: int) -> np.ndarray:
    """Assign descending prizes (top_k, top_k-1, …, 1) to the top-k by similarity."""
    prizes = np.zeros(len(sims), dtype=np.float32)
    if len(sims) == 0 or top_k <= 0:
        return prizes
    k = min(top_k, len(sims))
    top = np.argsort(-sims)[:k]
    for rank, idx in enumerate(top):
        if sims[idx] <= 0:
            continue
        prizes[idx] = float(k - rank)
    return prizes
