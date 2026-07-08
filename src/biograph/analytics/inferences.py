"""Structural inferences over the knowledge graph.

These are the notebook's "Graph Inferences & Analytics" and "Multi-hop Reasoning"
sections, re-expressed as pure ``networkx`` traversals so they run on the
in-memory :class:`KnowledgeGraph` without any Cypher backend.

The multi-hop routines are the concrete answer to *"Why GraphRAG wins on distant
knowledge"*: a path ``Drug -> Gene -> Disease`` links facts that never co-occur
in a single text chunk and are therefore invisible to plain vector RAG.
"""

from __future__ import annotations

from typing import Optional

from biograph.graph.knowledge_graph import KnowledgeGraph


def high_impact_entities(kg: KnowledgeGraph, top_k: int = 10) -> list[tuple[str, int]]:
    """Most-connected entities — proxies for research centrality."""
    return kg.degree_ranking(top_k)


def multi_hop_pathways(
    kg: KnowledgeGraph,
    source: str,
    max_hops: int = 2,
    source_label: Optional[str] = "Drug",
    limit: int = 20,
) -> list[list[tuple[str, str, str]]]:
    """Enumerate directed paths of up to ``max_hops`` edges from matching sources.

    Each path is a list of ``(head, relation, tail)`` steps. Used to surface
    indirect effects, e.g. ``Osimertinib --(causes)--> AXL --(...)--> ...``.
    """
    g = kg.to_networkx()
    seeds = [
        n for n in g.nodes
        if source.lower() in n.lower()
        and (source_label is None or g.nodes[n].get("label") == source_label)
    ]
    paths: list[list[tuple[str, str, str]]] = []

    def walk(node: str, trail: list[tuple[str, str, str]]):
        if len(paths) >= limit or len(trail) >= max_hops:
            return
        for _, nbr, data in g.out_edges(node, data=True):
            step = (node, data.get("relation", "?"), nbr)
            if step in trail:  # avoid trivial cycles
                continue
            new_trail = trail + [step]
            paths.append(new_trail)
            walk(nbr, new_trail)

    for seed in seeds:
        walk(seed, [])
    # keep only the longest, most informative paths first
    paths.sort(key=len, reverse=True)
    return paths[:limit]


def resistance_chains(kg: KnowledgeGraph, limit: int = 10) -> list[tuple[str, str, str, str]]:
    """Find ``Drug -> X -> Disease/Gene`` chains flagged by a 'causes' hop.

    Approximates the notebook's "Potential Resistance Chains" discovery query.
    Returns ``(drug, intermediate, target, target_label)`` tuples.
    """
    g = kg.to_networkx()
    out: list[tuple[str, str, str, str]] = []
    for drug in [n for n, d in g.nodes(data=True) if d.get("label") == "Drug"]:
        for _, mid, e1 in g.out_edges(drug, data=True):
            for _, tgt, e2 in g.out_edges(mid, data=True):
                if tgt == drug:
                    continue
                if e1.get("relation") == "causes" or e2.get("relation") == "causes":
                    out.append((drug, mid, tgt, g.nodes[tgt].get("label", "Entity")))
                if len(out) >= limit:
                    return out
    return out


def print_report(kg: KnowledgeGraph, focus_drug: str = "Osimertinib") -> None:
    """Console intelligence report mirroring the notebook's analytics cells."""
    print("--- High-Impact Entities (degree centrality) ---")
    for name, deg in high_impact_entities(kg):
        print(f"  {deg:>3}  {name} [{kg.node_label(name)}]")

    print(f"\n--- Multi-hop Reasoning: indirect effects for {focus_drug} ---")
    for path in multi_hop_pathways(kg, focus_drug, max_hops=2):
        chain = "  ".join(f"({h}) -{r}-> ({t})" for h, r, t in path)
        print(f"  {chain}")

    print("\n--- Potential Resistance Chains (Drug -> X -> target) ---")
    for drug, mid, tgt, lab in resistance_chains(kg):
        print(f"  {drug} may influence {tgt} ({lab}) via {mid}")
