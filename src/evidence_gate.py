"""
Step 8: Evidence Validation -- the deterministic anti-hallucination gate.

This is the single most important module in the "near-zero hallucination"
requirement. It is intentionally NOT the LLM's job to decide whether it has
enough evidence -- LLMs are unreliable judges of their own uncertainty. This
gate runs BEFORE the LLM is invoked, using two independent, non-generative
signals:

  1. Reranker/similarity score of the best-matching chunk.
  2. Lexical (Jaccard) overlap between the question and best chunk, as a
     cheap independent sanity check that doesn't depend on the same model
     that produced signal (1).

If either check fails, we short-circuit to "Not found in the retrieved 3GPP
documents" and never call the LLM for a generative answer -- so there is
nothing for the LLM to hallucinate about that specific question.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


_STOPWORDS = {
    "what", "does", "do", "is", "are", "the", "a", "an", "of", "in", "on",
    "to", "for", "and", "or", "how", "when", "where", "which", "who", "why",
    "describe", "explain", "tell", "me", "about", "with", "this", "that",
}


def lexical_overlap(question: str, text: str) -> float:
    """Query-term coverage: what fraction of the question's SIGNIFICANT terms
    (stopwords removed) appear in the chunk text. Deliberately not full
    Jaccard -- Jaccard's shared denominator unfairly penalizes long chunks
    matched against short questions, which would make this gate reject
    correct long-chunk answers."""
    q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower())) - _STOPWORDS
    d_terms = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    if not q_terms or not d_terms:
        return 0.0
    return len(q_terms & d_terms) / len(q_terms)


def validate_evidence(question: str, reranked_results):
    """
    reranked_results: list of (score, chunk) sorted best-first.

    Returns (is_sufficient: bool, reason: str, evidence: list[chunk])
    """
    if not reranked_results:
        return False, "No candidate chunks retrieved at all.", []

    top_score, top_chunk = reranked_results[0]
    overlap = lexical_overlap(question, top_chunk["text"])

    if top_score < config.MIN_EVIDENCE_SCORE:
        return False, (
            f"Top retrieval score {top_score:.3f} is below the minimum "
            f"confidence threshold ({config.MIN_EVIDENCE_SCORE})."
        ), []

    if overlap < config.MIN_LEXICAL_OVERLAP:
        return False, (
            f"Best-matching chunk has insufficient lexical overlap with the "
            f"question ({overlap:.3f} < {config.MIN_LEXICAL_OVERLAP}); likely "
            f"a spurious semantic match."
        ), []

    # Evidence accepted. Keep only chunks that clear a lower per-chunk bar
    # so weak secondary hits don't get cited alongside strong ones.
    evidence = [c for score, c in reranked_results if score >= config.MIN_EVIDENCE_SCORE * 0.6]
    return True, "Evidence sufficient.", evidence
