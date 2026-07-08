"""Configuration loading: YAML files for hyper-parameters, env vars for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _load_dotenv() -> None:
    """Populate os.environ from a repo-root .env file if present.

    We avoid a hard dependency on python-dotenv: a tiny parser is enough for the
    ``KEY=value`` files we ship, and it keeps the core install light.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    """Secrets and endpoints pulled from the environment."""

    entrez_email: str = "researcher@example.com"
    entrez_api_key: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    # Local LLM via Ollama (OpenAI-compatible endpoint) — no API key.
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:2b"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            entrez_email=os.environ.get("ENTREZ_EMAIL", "researcher@example.com"),
            entrez_api_key=os.environ.get("ENTREZ_API_KEY") or None,
            neo4j_uri=os.environ.get("NEO4J_URI") or None,
            neo4j_user=os.environ.get("NEO4J_USER") or None,
            neo4j_password=os.environ.get("NEO4J_PASSWORD") or None,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            llm_model=os.environ.get("BIOGRAPH_LLM_MODEL", "qwen3.5:2b"),
        )


@dataclass
class Config:
    """Nested pipeline hyper-parameters (a thin typed view over the YAML)."""

    raw: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config, defaulting to ``configs/default.yaml``."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(raw=data)
