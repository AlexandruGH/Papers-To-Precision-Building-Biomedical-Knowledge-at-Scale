"""CLI: answer a question with G-Retriever (PCST subgraph + LLM synthesis).

    biograph-gretriever --question "How does Osimertinib resistance arise?"
    biograph-gretriever -q "..." --top-k-nodes 20 --viz

Runs offline with a hashed encoder and a dry-run LLM by default. For real answer
synthesis, run a local Ollama model (no API key) — `ollama pull qwen2.5:1.5b`
and `pip install -e ".[llm]"`. Install biograph[nlp] to use a PubMedBERT encoder
for retrieval.
"""

from __future__ import annotations

import argparse

from biograph.config import load_config
from biograph.pipeline import Pipeline
from biograph.retrieval.g_retriever import GRetriever


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G-Retriever question answering over the KG.")
    p.add_argument("--config", default=None)
    p.add_argument("--online", action="store_true")
    p.add_argument("-q", "--question", default="What are the resistance mechanisms of Osimertinib?")
    p.add_argument("--top-k-nodes", type=int, default=None)
    p.add_argument("--top-k-edges", type=int, default=None)
    p.add_argument("--encoder", default=None, help="sentence-transformers model for retrieval.")
    p.add_argument("--viz", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    gr_cfg = cfg.section("g_retriever")

    result = Pipeline(config=cfg).run(offline=not args.online)
    result.describe()

    encode_fn = None
    if args.encoder:
        from biograph.embeddings.encoders import BiEncoder

        enc = BiEncoder(args.encoder)
        encode_fn = lambda xs: enc.encode(list(xs))  # noqa: E731

    from biograph.llm import LLMClient

    retriever = GRetriever(
        result.kg, encode_fn=encode_fn,
        llm=LLMClient(model=gr_cfg.get("llm_model"), max_tokens=gr_cfg.get("llm_max_tokens", 1024)),
        pcst_cost=gr_cfg.get("pcst_cost", 0.5),
    )
    out = retriever.answer(
        args.question,
        top_k_nodes=args.top_k_nodes or gr_cfg.get("top_k_nodes", 15),
        top_k_edges=args.top_k_edges or gr_cfg.get("top_k_edges", 30),
    )

    sub = out["subgraph"]
    print(f"\n=== RETRIEVED SUBGRAPH ({len(sub.nodes)} nodes, {len(sub.edges)} edges) ===")
    print(sub.textualize())

    print(f"\n=== ANSWER (model={out['model']}, dry_run={out['dry_run']}) ===")
    if out["dry_run"]:
        print("[Ollama unavailable — showing the prompt the local LLM would receive. "
              "Start `ollama serve` and pull the model for real synthesis.]\n")
    print(out["answer"])

    if args.viz:
        from biograph.viz.pyvis_graph import render_subgraph

        render_subgraph(sub, result.kg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
