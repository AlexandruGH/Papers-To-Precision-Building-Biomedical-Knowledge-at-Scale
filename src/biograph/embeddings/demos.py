"""Runnable reproductions of the notebook's Section-1 embedding demonstrations.

Each function is self-contained and prints a side-by-side comparison, exactly as
the notebook cells did — but importable and testable. They require the ``nlp``
extra (``pip install biograph[nlp]``).
"""

from __future__ import annotations


def demo_biencoder_vs_crossencoder() -> None:
    """Show why a fast bi-encoder needs a cross-encoder re-rank for precision."""
    from biograph.embeddings.encoders import BiEncoder, CrossEncoderReranker

    question = "How many hours does a cat usually sleep?"
    documents = [
        "The dog is man's best friend and sleeps 10 hours.",
        "A domestic feline spends on average between 12 and 16 hours a day resting.",
        "The National Bank sleeps on a gold reserve.",
        "Cats are active hunters and don't sleep at all during the night hours.",
        "The construction worker spent 8 hours at the site before going to sleep.",
    ]
    print(f"Question: {question}\n")

    bi = BiEncoder("all-MiniLM-L6-v2")
    print("--- BI-ENCODER (cosine similarity) ---")
    for idx, score in bi.rank(question, documents):
        print(f"  {score:+.3f}  {documents[idx]}")

    ce = CrossEncoderReranker()
    print("\n--- CROSS-ENCODER (re-ranked) ---")
    for idx, score in ce.rank(question, documents):
        print(f"  {score:+.3f}  {documents[idx]}")


def demo_negation_trap() -> None:
    """A logic/negation case where a plain bi-encoder is fooled by lexical overlap.

    Uses an NLI cross-encoder (DeBERTa) which reasons about entailment rather than
    surface similarity — the model that "surpassed human performance on SuperGLUE".
    """
    from biograph.embeddings.encoders import BiEncoder, CrossEncoderReranker

    query = "Does my insurance cover damages if I hit an animal on the highway?"
    correct = "The auto policy covers repairs in a collision with wildlife on public roads."
    trap = ("If you are hit on the highway, the insurance does not cover damages if you "
            "were transporting a pet animal.")
    docs = [correct, trap]

    bi = BiEncoder("all-MiniLM-L6-v2")
    print("--- BI-ENCODER (keyword-biased) ---")
    for idx, score in bi.rank(query, docs):
        print(f"  {score:+.3f}  {docs[idx]}")

    nli = CrossEncoderReranker("cross-encoder/nli-deberta-v3-base")
    print("\n--- DeBERTa NLI cross-encoder (logic-aware) ---")
    for idx, score in nli.rank(query, docs):
        print(f"  {score:+.3f}  {docs[idx]}")


if __name__ == "__main__":  # pragma: no cover
    demo_biencoder_vs_crossencoder()
    print("\n" + "=" * 60 + "\n")
    demo_negation_trap()
