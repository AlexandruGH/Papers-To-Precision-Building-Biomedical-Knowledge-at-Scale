"""Local LLM answer-synthesis via Ollama + Instructor (no API key required).

Both the hybrid RAG and the G-Retriever assemble a context-rich prompt and hand
it to a language model. We isolate that call here so the retrieval logic stays
model-agnostic and testable.

Backend
-------
A **local Ollama** server exposing an OpenAI-compatible endpoint
(``http://localhost:11434/v1``), driven through the **Instructor** library so
answers come back as a *validated Pydantic object* (``BiomedAnswer``) rather than
free text — the model is forced to separate its prose answer from the list of
cited PMIDs. Default model is a small Qwen (``qwen3.5:2b``); swap it for any tag
you have pulled (e.g. ``qwen3:0.6b`` for the smallest, ``qwen3:1.7b``) via config
or ``BIOGRAPH_LLM_MODEL``.

No cloud credentials are used anywhere. If Ollama / Instructor are unavailable
(server down, deps not installed) the client runs in **dry-run** mode and returns
the assembled prompt, so demos and tests still work.

Setup:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3.5:2b
    pip install -e ".[llm]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from biograph.config import Settings

# The structured response schema is defined only if pydantic is present (it ships
# with Instructor). Guarded so the core install imports this module cleanly.
try:
    from pydantic import BaseModel, Field

    class BiomedAnswer(BaseModel):
        """Schema Instructor forces the local model to fill."""

        answer: str = Field(description="Concise answer grounded in the provided context.")
        cited_pmids: list[str] = Field(
            default_factory=list, description="PMIDs from the context that support the answer."
        )
        reasoning: str = Field(
            default="", description="Brief chain of graph hops / evidence used."
        )

        def to_text(self) -> str:
            cites = f"\n\nCited PMIDs: {', '.join(self.cited_pmids)}" if self.cited_pmids else ""
            reason = f"\n\nReasoning: {self.reasoning}" if self.reasoning else ""
            return f"{self.answer}{reason}{cites}"

except ImportError:  # pragma: no cover - pydantic absent in the minimal install
    BaseModel = None  # type: ignore
    BiomedAnswer = None  # type: ignore


_SYSTEM = (
    "You are a rigorous biomedical research assistant. Ground every claim strictly "
    "in the provided context, reason over multi-hop relationships when relevant, and "
    "cite the PMIDs that support your answer. If the context is insufficient, say so."
)


@dataclass
class LLMResponse:
    text: str
    model: str
    dry_run: bool = False
    structured: Optional["BiomedAnswer"] = None


class LLMClient:
    """Structured local inference through Ollama + Instructor."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.settings = settings or Settings.from_env()
        self.model = model or self.settings.llm_model
        self.base_url = base_url or self.settings.ollama_base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None  # instructor-patched OpenAI client

        try:
            import instructor
            from openai import OpenAI

            # api_key is required by the SDK but ignored by Ollama — no secret.
            oai = OpenAI(base_url=self.base_url, api_key="ollama")
            self._client = instructor.from_openai(oai, mode=instructor.Mode.JSON)
        except ImportError:
            print("! instructor/openai not installed; LLM layer runs in dry-run mode "
                  "(install with `pip install -e \".[llm]\"`).")

    @property
    def available(self) -> bool:
        return self._client is not None and BiomedAnswer is not None

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a structured biomedical answer from a local Qwen via Ollama.

        Falls back to dry-run (returns the prompt) if the client is unavailable or
        the local server cannot be reached.
        """
        if not self.available:
            return LLMResponse(text=prompt, model=self.model, dry_run=True)
        try:
            result: "BiomedAnswer" = self._client.chat.completions.create(
                model=self.model,
                response_model=BiomedAnswer,
                max_retries=2,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system or _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return LLMResponse(text=result.to_text(), model=self.model,
                               dry_run=False, structured=result)
        except Exception as exc:  # noqa: BLE001 - server down / model not pulled / parse fail
            print(f"! Ollama call failed ({exc}); returning prompt (dry-run). "
                  f"Is `ollama serve` running and `{self.model}` pulled?")
            return LLMResponse(text=prompt, model=self.model, dry_run=True)
