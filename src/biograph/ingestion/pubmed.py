"""Programmatic access to PubMed abstracts via NCBI Entrez (Biopython).

Refactored from notebook cell ``fetch_abstracts``. Improvements:
* honours NCBI etiquette (email + optional API key, request throttling),
* returns typed :class:`~biograph.schema.Article` objects,
* degrades gracefully when Biopython / network are unavailable so the rest of
  the pipeline can run against the bundled sample corpus.
"""

from __future__ import annotations

import time
from typing import Iterable

from biograph.config import Settings
from biograph.schema import Article


class PubMedClient:
    """Thin wrapper around ``Bio.Entrez`` with typed output."""

    def __init__(self, settings: Settings | None = None, request_delay_s: float = 0.2):
        self.settings = settings or Settings.from_env()
        self.request_delay_s = request_delay_s
        try:
            from Bio import Entrez  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "biopython is required for PubMed ingestion. "
                "Install with `pip install biograph[ingest]`, or use "
                "`biograph.ingestion.sample_articles()` for offline runs."
            ) from exc
        from Bio import Entrez

        self._Entrez = Entrez
        Entrez.email = self.settings.entrez_email
        if self.settings.entrez_api_key:
            Entrez.api_key = self.settings.entrez_api_key

    def search_ids(self, query: str, max_results: int = 25) -> list[str]:
        handle = self._Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = self._Entrez.read(handle)
        handle.close()
        return list(record.get("IdList", []))

    def fetch(self, pmids: Iterable[str]) -> list[Article]:
        articles: list[Article] = []
        for pmid in pmids:
            try:
                handle = self._Entrez.efetch(
                    db="pubmed", id=pmid, rettype="abstract", retmode="xml"
                )
                record = self._Entrez.read(handle)
                handle.close()
                citation = record["PubmedArticle"][0]["MedlineCitation"]
                art = citation["Article"]
                abstract = " ".join(
                    str(chunk) for chunk in art.get("Abstract", {}).get("AbstractText", [""])
                )
                year = None
                try:
                    year = int(art["Journal"]["JournalIssue"]["PubDate"].get("Year"))
                except (KeyError, TypeError, ValueError):
                    pass
                articles.append(
                    Article(
                        pmid=str(pmid),
                        title=str(art.get("ArticleTitle", "")),
                        text=abstract,
                        year=year,
                        journal=str(art["Journal"].get("Title", "")) or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad record shouldn't abort the batch
                print(f"  ! skipped PMID {pmid}: {exc}")
            time.sleep(self.request_delay_s)
        return articles

    def search_and_fetch(self, query: str, max_results: int = 25) -> list[Article]:
        print(f"Searching PubMed: {query!r} (max {max_results})")
        ids = self.search_ids(query, max_results)
        print(f"  found {len(ids)} PMIDs; fetching abstracts…")
        return self.fetch(ids)


def fetch_abstracts(query: str, count: int = 25, settings: Settings | None = None) -> list[Article]:
    """Convenience one-shot search+fetch (mirrors the notebook helper)."""
    return PubMedClient(settings).search_and_fetch(query, count)
