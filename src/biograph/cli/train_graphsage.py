"""CLI: train GraphSAGE on the knowledge graph and generate link hypotheses.

    biograph-train-sage                        # offline demo
    biograph-train-sage --epochs 200 --objective unsupervised
    biograph-train-sage --predict Drug Gene --top-k 15

Requires the ``gnn`` extra:  pip install biograph[gnn]
"""

from __future__ import annotations

import argparse

from biograph.config import load_config
from biograph.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GraphSAGE for structural embeddings.")
    p.add_argument("--config", default=None)
    p.add_argument("--online", action="store_true")
    p.add_argument("--objective", choices=["unsupervised", "classification"],
                   default="unsupervised")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--predict", nargs=2, metavar=("HEAD_LABEL", "TAIL_LABEL"),
                   default=["Drug", "Gene"], help="Entity types for link hypotheses.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--save", default=None, help="Path to save the trained model.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    sage_cfg = cfg.section("graphsage")

    result = Pipeline(config=cfg).run(offline=not args.online)
    result.describe()

    from biograph.models.graphsage import GraphSAGETrainer  # lazy: needs torch

    trainer = GraphSAGETrainer(
        result.kg,
        feature_dim=cfg.get("embeddings.dim", 768),
        hidden_dim=sage_cfg.get("hidden_dim", 256),
        out_dim=sage_cfg.get("out_dim", 128),
        num_layers=sage_cfg.get("num_layers", 2),
        dropout=sage_cfg.get("dropout", 0.3),
        aggr=sage_cfg.get("aggr", "mean"),
        lr=sage_cfg.get("lr", 0.01),
    )
    epochs = args.epochs or sage_cfg.get("epochs", 100)

    print(f"\n=== Training GraphSAGE ({args.objective}, {epochs} epochs) ===")
    if args.objective == "unsupervised":
        trainer.train_unsupervised(epochs=epochs)
    else:
        trainer.train_node_classification(epochs=epochs)
    trainer.write_embeddings_to_graph()

    head_label, tail_label = args.predict
    print(f"\n=== Link Hypotheses: novel {head_label} → {tail_label} interactions ===")
    for h in trainer.predict_links(head_label, tail_label, top_k=args.top_k):
        print(f"  {h.score:.3f}  {h.head} → {h.tail}")

    if args.save:
        trainer.save(args.save)
        print(f"\nSaved model → {args.save}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
