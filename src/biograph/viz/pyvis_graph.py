"""Interactive HTML visualisations of the knowledge graph with pyvis.

Consolidates the notebook's several pyvis cells into two reusable renderers:
one for the whole graph, one for a G-Retriever subgraph (to *see* what the
retriever selected for a given question).
"""

from __future__ import annotations

from typing import Optional

from biograph.graph.knowledge_graph import KnowledgeGraph

_COLORS = {"Disease": "#ff7675", "Gene": "#74b9ff", "Drug": "#55efc4",
           "Mutation": "#ffeaa7", "Entity": "#dfe6e9"}


def render_graph(kg: KnowledgeGraph, out_path: str = "outputs/graph.html",
                 max_edges: int = 200, height: str = "650px") -> str:
    """Render the full (or truncated) knowledge graph to a standalone HTML file."""
    from pyvis.network import Network
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    net = Network(height=height, width="100%", bgcolor="#222222", font_color="white",
                  directed=True, cdn_resources="remote")
    net.toggle_physics(True)

    g = kg.to_networkx()
    added: set[str] = set()
    for i, (h, t, d) in enumerate(g.edges(data=True)):
        if i >= max_edges:
            break
        for name in (h, t):
            if name not in added:
                lbl = g.nodes[name].get("label", "Entity")
                net.add_node(name, label=name, color=_COLORS.get(lbl, _COLORS["Entity"]),
                             title=f"Type: {lbl}\nPMIDs: {sorted(g.nodes[name].get('pmids', []))}")
                added.add(name)
        evidence = d.get("context", "")[:200]
        net.add_edge(h, t, label=d.get("relation", "").upper(),
                     title=f"{d.get('status','')} | {d.get('pmid','')}\n{evidence}")
    net.write_html(out_path, notebook=False)
    print(f"Wrote interactive graph → {out_path}")
    return out_path


def render_subgraph(sub, kg: Optional[KnowledgeGraph] = None,
                    out_path: str = "outputs/subgraph.html", height: str = "600px") -> str:
    """Visualise a :class:`RetrievedSubgraph` (G-Retriever output)."""
    from pyvis.network import Network
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    net = Network(height=height, width="100%", bgcolor="#1a1a2e", font_color="white",
                  directed=True, cdn_resources="remote")
    for n in sub.nodes:
        lbl = sub.node_meta.get(n, "Entity")
        net.add_node(n, label=n, color=_COLORS.get(lbl, _COLORS["Entity"]), title=f"Type: {lbl}")
    for (h, r, t), meta in zip(sub.edges, sub.edge_meta):
        net.add_edge(h, t, label=r.upper(), title=f"{meta.get('status','')} | {meta.get('pmid','')}")
    net.write_html(out_path, notebook=False)
    print(f"Wrote retrieved subgraph → {out_path}")
    return out_path
