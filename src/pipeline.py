"""
End-to-end orchestration of the full RAG pipeline:

  question -> retrieve -> rerank -> evidence gate -> LLM -> verify -> answer

This is the module the chat app (app.py) calls. It returns a rich result
dict so the UI/CLI can show not just the answer but *why* it trusts it
(scores, citations, gate decision) -- important for demoing the
anti-hallucination behaviour to graders.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import Retriever
from src.evidence_gate import validate_evidence
from src.llm_client import generate_answer
from src.verify import verify_claims


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str) -> dict:
        reranked, ts_filter = self.retriever.retrieve(question)

        is_sufficient, gate_reason, evidence = validate_evidence(question, reranked)
        if not is_sufficient:
            return {
                "question": question,
                "answer": "Not found in the retrieved 3GPP documents for this question.",
                "status": "REFUSED_INSUFFICIENT_EVIDENCE",
                "gate_reason": gate_reason,
                "ts_filter_detected": ts_filter,
                "citations": [],
                "retrieval_debug": [
                    {"score": s, "chunk_id": c["chunk_id"], "ts": c["ts_number"],
                     "clause": c["breadcrumb"]} for s, c in reranked
                ],
            }

        evidence_by_id = {c["chunk_id"]: c for c in evidence}
        llm_output = generate_answer(question, evidence)
        verified_output, verification_report = verify_claims(llm_output, evidence_by_id)

        if verified_output["insufficient_evidence"] or not verified_output["answer"]:
            return {
                "question": question,
                "answer": "Not found in the retrieved 3GPP documents for this question.",
                "status": "REFUSED_NO_VERIFIED_CLAIMS",
                "gate_reason": "LLM output had no claims that passed post-generation verification.",
                "ts_filter_detected": ts_filter,
                "citations": [],
                "verification_report": verification_report,
            }

        citations = [
            {"ts_number": c["ts_number"], "release": c["release"],
             "clause": f"{c['clause_id']} {c['clause_title']}".strip(),
             "page": c.get("page"), "chunk_id": c["chunk_id"]}
            for c in verified_output["claims"]
        ]
        # de-dup citations
        seen = set()
        dedup_citations = []
        for c in citations:
            key = (c["ts_number"], c["clause"], c["page"])
            if key not in seen:
                seen.add(key)
                dedup_citations.append(c)

        return {
            "question": question,
            "answer": verified_output["answer"],
            "status": "ANSWERED",
            "gate_reason": gate_reason,
            "ts_filter_detected": ts_filter,
            "citations": dedup_citations,
            "verification_report": verification_report,
        }
