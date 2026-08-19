"""Local, dependency-light text-similarity primitives shared by every evaluator.

Design note (see docs/GroundControl_Design.md, LLD 2.5 "Evaluation as a
runtime gate"): the *same* metric definitions are used by two judge
configurations — a cheap online judge that gates answers at request time,
and a stronger offline judge used for the golden-set benchmark. Both
configurations in this codebase are built on top of the helpers below, which
compute TF-IDF cosine similarity between texts. This is a real, correct,
fully local approximation of the "embedding-overlap heuristic" the design
doc calls for — it requires no network access and no live LLM, so evaluators
are deterministic and testable in a sandbox with no model weights downloaded.

Upgrade path: swap `tfidf_cosine` / `max_sentence_similarity` for a call to
`ragas` metrics (faithfulness, answer_relevancy, context_precision/recall,
answer_correctness) backed by an `OllamaProvider` LLM judge and a real
sentence-embedding model (`sentence-transformers`, already a project
dependency) once `OLLAMA_BASE_URL` is reachable and embeddings can be
computed live. The `Evaluator` protocol and `EvalResult` schema do not need
to change for that upgrade — only the scoring implementation inside each
evaluator class.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences using simple punctuation rules."""
    if not text or not text.strip():
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts under a shared TF-IDF fit.

    Returns 0.0 for empty/degenerate input rather than raising, since eval
    samples may legitimately contain empty answers or contexts.
    """
    a, b = (text_a or "").strip(), (text_b or "").strip()
    if not a or not b:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([a, b])
        if matrix.shape[1] == 0:
            return 0.0
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    except ValueError:
        # Empty vocabulary (e.g. both texts are pure stopwords/punctuation).
        return 0.0
    return float(max(0.0, min(1.0, sim)))


def max_similarity_to_any(text: str, candidates: list[str]) -> float:
    """Max cosine similarity between `text` and any of `candidates`."""
    if not candidates:
        return 0.0
    return max((tfidf_cosine(text, c) for c in candidates), default=0.0)


def mean_sentence_support(answer: str, contexts: list[str], threshold: float = 0.15) -> float:
    """Fraction of answer sentences that are "supported" by at least one context.

    This is the local stand-in for ragas' faithfulness metric, which
    normally decomposes the answer into atomic claims via an LLM and checks
    each claim against the context with an NLI judge. Here, each sentence is
    treated as a claim and "supported" means its max TF-IDF cosine
    similarity to any context chunk clears `threshold`.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    if not contexts:
        return 0.0
    supported = sum(1 for s in sentences if max_similarity_to_any(s, contexts) >= threshold)
    return supported / len(sentences)
