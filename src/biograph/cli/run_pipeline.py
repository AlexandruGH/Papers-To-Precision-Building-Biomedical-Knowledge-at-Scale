"""CLI: build the knowledge graph and run hybrid GraphRAG + analytics.

    biograph-pipeline                       # offline demo (bundled corpus)
    biograph-pipeline --online --query "..." --max-results 30
    biograph-pipeline --viz --ask "resistance mechanisms" --entity Osimertinib
"""

from __future__ import annotations

import argparse

from biograph.analytics.inferences import print_report
from biograph.config import load_config
from biograph.pipeline import Pipeline
from biograph.retrieval.hybrid_rag import HybridRAG


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the biomedical KG and query it.")
    p.add_argument("--config", default=None, help="Path to a YAML config.")
    p.add_argument("--online", action="store_true", help="Fetch from PubMed + run GLiNER.")
    p.add_argument("--query", default=None, help="PubMed search query (online mode).")
    p.add_argument("--max-results", type=int, default=None)
    p.add_argument("--no-resolve", action="store_true", help="Skip entity resolution.")
    p.add_argument("--entity", default="Osimertinib", help="Focus entity for hybrid RAG.")
    p.add_argument("--ask", default="What are the resistance mechanisms?",
                   help="Question for hybrid RAG synthesis.")
    p.add_argument("--viz", action="store_true", help="Write an interactive HTML graph.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = Pipeline(config=load_config(args.config))
    result = pipeline.run(
        offline=not args.online, query=args.query,
        max_results=args.max_results, resolve=not args.no_resolve,
    )
    print("\n=== PIPELINE SUMMARY ===")
    result.describe()

    print("\n=== GRAPH ANALYTICS ===")
    print_report(result.kg, focus_drug=args.entity)

    print("\n=== HYBRID GraphRAG ===")
    rag = HybridRAG(kg=result.kg, articles=result.articles)
    out = rag.answer(args.entity, args.ask)
    print(out)

    if args.viz:
        from biograph.viz.pyvis_graph import render_graph

        render_graph(result.kg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
