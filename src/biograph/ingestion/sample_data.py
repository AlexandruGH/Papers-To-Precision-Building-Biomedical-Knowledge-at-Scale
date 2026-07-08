"""Offline biomedical sample corpus.

Lets every downstream component (graph build, GraphSAGE, G-Retriever) run with
zero network access and no large model downloads — essential for reproducible
demos, CI, and teaching. The facts are simplified but scientifically plausible,
centred on the EGFR / lung-cancer resistance story from the notebook.
"""

from __future__ import annotations

from biograph.schema import Article, Entity, Triplet

_ARTICLES = [
    Article("30000001", "Osimertinib in EGFR-mutant NSCLC",
            "Osimertinib is a third-generation TKI that inhibits EGFR bearing the "
            "T790M mutation and is approved as standard of care for advanced NSCLC.",
            2018, "N Engl J Med"),
    Article("30000002", "C797S and acquired resistance",
            "Acquired resistance to Osimertinib is frequently driven by the EGFR "
            "C797S mutation, which prevents covalent binding of the drug.",
            2019, "Cancer Discov"),
    Article("30000003", "MET amplification bypass",
            "MET amplification activates bypass signalling and causes resistance to "
            "EGFR inhibitors; combination with a MET inhibitor such as Capmatinib "
            "restores sensitivity in vitro.",
            2020, "Clin Cancer Res"),
    Article("30000004", "Gefitinib and the T790M gatekeeper",
            "Gefitinib, a first-generation EGFR TKI, treats EGFR-mutant lung cancer "
            "but tumours acquire the T790M gatekeeper mutation causing resistance.",
            2010, "Lancet Oncol"),
    Article("30000005", "KRAS as a downstream driver",
            "EGFR signalling activates KRAS; KRAS mutations cause resistance and "
            "drive lung adenocarcinoma independently of EGFR.",
            2017, "Nature"),
    Article("30000006", "Sotorasib targets KRAS G12C",
            "Sotorasib inhibits the KRAS G12C mutation and treats non-small cell "
            "lung cancer harbouring that alteration.",
            2021, "N Engl J Med"),
    Article("30000007", "AXL-mediated escape",
            "Upregulation of AXL causes resistance to Osimertinib; preclinical "
            "murine models show that inhibiting AXL restores drug response.",
            2019, "Nat Commun"),
    Article("30000008", "MET–HGF axis",
            "MET activation via HGF promotes bypass resistance; Capmatinib inhibits "
            "MET and is approved for MET-altered NSCLC.",
            2020, "J Thorac Oncol"),
]

# Pre-extracted (head, relation, tail) triplets keyed by evidence PMID. Types:
# Disease / Gene / Drug / Mutation. This is the output a GLiNER pass would yield.
_RAW = [
    ("Osimertinib", "Drug", "inhibits", "EGFR", "Gene", "Clinical", "30000001"),
    ("Osimertinib", "Drug", "treats", "NSCLC", "Disease", "Clinical", "30000001"),
    ("EGFR", "Gene", "causes", "T790M", "Mutation", "Mentioned", "30000001"),
    ("C797S", "Mutation", "causes", "Osimertinib", "Drug", "Experimental", "30000002"),
    ("EGFR", "Gene", "causes", "C797S", "Mutation", "Mentioned", "30000002"),
    ("MET", "Gene", "causes", "Osimertinib", "Drug", "Experimental", "30000003"),
    ("Capmatinib", "Drug", "inhibits", "MET", "Gene", "Experimental", "30000003"),
    ("Gefitinib", "Drug", "treats", "Lung Cancer", "Disease", "Clinical", "30000004"),
    ("Gefitinib", "Drug", "targets", "EGFR", "Gene", "Clinical", "30000004"),
    ("T790M", "Mutation", "causes", "Gefitinib", "Drug", "Mentioned", "30000004"),
    ("EGFR", "Gene", "targets", "KRAS", "Gene", "Mentioned", "30000005"),
    ("KRAS", "Gene", "causes", "Lung Adenocarcinoma", "Disease", "Mentioned", "30000005"),
    ("Sotorasib", "Drug", "inhibits", "KRAS", "Gene", "Clinical", "30000006"),
    ("Sotorasib", "Drug", "treats", "NSCLC", "Disease", "Clinical", "30000006"),
    ("KRAS", "Gene", "causes", "G12C", "Mutation", "Mentioned", "30000006"),
    ("AXL", "Gene", "causes", "Osimertinib", "Drug", "Experimental", "30000007"),
    ("MET", "Gene", "targets", "HGF", "Gene", "Mentioned", "30000008"),
    ("Capmatinib", "Drug", "inhibits", "MET", "Gene", "Clinical", "30000008"),
]


def sample_articles() -> list[Article]:
    """Return the bundled offline corpus."""
    return [Article(**a.to_dict()) for a in _ARTICLES]


def sample_triplets() -> list[Triplet]:
    """Return pre-extracted triplets (as if produced by the GLiNER stage)."""
    out: list[Triplet] = []
    for head, h_lab, rel, tail, t_lab, status, pmid in _RAW:
        art = next((a for a in _ARTICLES if a.pmid == pmid), None)
        out.append(
            Triplet(
                head=Entity(head, h_lab),
                tail=Entity(tail, t_lab),
                relation=rel,
                score=0.9,
                status=status,
                pmid=pmid,
                context=(art.text if art else "")[:250],
            )
        )
    return out
