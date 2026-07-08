"""Zero-shot biomedical entity & relation extraction, plus entity resolution."""

from biograph.extraction.entity_resolution import EntityResolver
from biograph.extraction.gliner_extractor import GLiNERExtractor

__all__ = ["GLiNERExtractor", "EntityResolver"]
