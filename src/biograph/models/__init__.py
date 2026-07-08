"""Graph neural networks (GraphSAGE) and PyG interop.

Imports are lazy: the heavy ``torch`` / ``torch_geometric`` stack is only pulled
in when you actually touch these symbols, so the rest of ``biograph`` stays
importable with just the core dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["GraphSAGE", "GraphSAGETrainer", "kg_to_pyg", "PyGBundle"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute loading
    if name in {"GraphSAGE", "GraphSAGETrainer"}:
        from biograph.models import graphsage

        return getattr(graphsage, name)
    if name in {"kg_to_pyg", "PyGBundle"}:
        from biograph.models import pyg_export

        return getattr(pyg_export, name)
    raise AttributeError(f"module 'biograph.models' has no attribute {name!r}")
