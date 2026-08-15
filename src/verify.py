"""
Step 10 (added beyond the original diagram): Post-Generation Verification.

Even with grounded prompting and an evidence gate, an LLM can still drift off
the provided context. This module closes the loop by checking, for every
claimed sentence the LLM produced, that it has real lexical/semantic support
in the chunk it cites. Unsupported claims are stripped from the final answer
rather than silently shipped -- this is what lets you defend a "near-zero
hallucination" claim with evidence rather than just a prompt instruction.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MIN_CLAIM_SUPPORT = 0.15  # jaccard overlap between claim sentence and its cited chunk


def _overlap(a: str, b: str) -> float:
    a_terms = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    b_terms = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    if not a_terms or not b_terms:
        return 0.0
    return len(a_terms & b_terms) / len(a_terms | b_terms)


def verify_claims(llm_output: dict, evidence_by_id: dict):
    """
    Returns (verified_output, verification_report)
    verified_output has unsupported claims stripped from `claims` and
    `answer` rebuilt from only the supported claims.
    """
    claims = llm_output.get("claims", [])
    supported, unsupported = [], []

    for claim in claims:
        chunk = evidence_by_id.get(claim.get("chunk_id"))
        if chunk is None:
            unsupported.append({**claim, "reason": "cited chunk_id not in retrieved evidence"})
            continue
        score = _overlap(claim["text"], chunk["text"])
        if score >= MIN_CLAIM_SUPPORT:
            supported.append({**claim, "support_score": round(score, 3),
                               "ts_number": chunk["ts_number"], "release": chunk["release"],
                               "clause_id": chunk["clause_id"], "clause_title": chunk["clause_title"],
                               "page": chunk.get("page")})
        else:
            unsupported.append({**claim, "reason": f"low lexical support ({score:.3f})"})

    if not supported:
        rebuilt_answer = ""
        insufficient = True
    else:
        parts = []
        for c in supported:
            page_str = f", p.{c['page']}" if c.get("page") is not None else ""
            parts.append(f"{c['text']} [{c['ts_number']} §{c['clause_id']}{page_str}]")
        rebuilt_answer = " ".join(parts)
        insufficient = False

    verified_output = {
        "answer": rebuilt_answer,
        "claims": supported,
        "insufficient_evidence": insufficient,
    }
    report = {
        "n_claims_total": len(claims),
        "n_supported": len(supported),
        "n_stripped_unsupported": len(unsupported),
        "unsupported_details": unsupported,
    }
    return verified_output, report
