"""Literature ingestion (PubMed) and offline sample corpora."""

from biograph.ingestion.pubmed import PubMedClient, fetch_abstracts
from biograph.ingestion.sample_data import sample_articles, sample_triplets

__all__ = ["PubMedClient", "fetch_abstracts", "sample_articles", "sample_triplets"]
