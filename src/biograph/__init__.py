"""biograph — biomedical Knowledge-Graph construction and graph-native retrieval.

The package turns unstructured scientific literature into a queryable knowledge
graph and layers three retrieval paradigms on top of it:

* **Hybrid GraphRAG** — vector similarity + Cypher traversal (from the notebook).
* **GraphSAGE**       — inductive GNN node embeddings for link prediction.
* **G-Retriever**     — PCST subgraph retrieval + LLM answer synthesis.

See ``README.md`` for the research narrative and ``docs/`` for architecture notes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from biograph.config import Settings, load_config  # noqa: E402
from biograph.schema import Article, Entity, Triplet  # noqa: E402

__all__ = ["Settings", "load_config", "Article", "Entity", "Triplet", "__version__"]
