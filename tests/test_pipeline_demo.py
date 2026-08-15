"""
Automated tests for the hallucination-gate behaviour. Run with:
    cd 3gpp_rag_chatbot
    RAG_EMBEDDING_BACKEND=tfidf RAG_LLM_BACKEND=echo python3 -m pytest tests/ -v

These tests exercise the FULL pipeline (ingest -> index -> retrieve -> gate ->
generate -> verify) against the synthetic sample corpus in data/sample/, so
they run with zero external downloads or network access -- suitable for CI.

For your project report, this file (and its pass/fail output) is your
evidence that the "near-zero hallucination" claim is enforced by code, not
just prompted for.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.ingest import ingest_directory, save_chunks
from src.build_index import EmbeddingBackend, build_vector_index, build_bm25_index, save_metadata
from src.pipeline import RAGPipeline
import config


@pytest.fixture(scope="module", autouse=True)
def build_demo_index():
    """(Re)build the demo index from the synthetic sample corpus once per test run."""
    os.environ["RAG_EMBEDDING_BACKEND"] = "tfidf"
    os.environ["RAG_LLM_BACKEND"] = "echo"
    chunks = ingest_directory(config.SAMPLE_DIR)
    save_chunks(chunks)
    embedder = EmbeddingBackend(mode="tfidf")
    build_vector_index(chunks, embedder)
    build_bm25_index(chunks)
    save_metadata(chunks)
    yield


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline()


# ---- In-domain questions: MUST be answered with a correct TS citation ----

@pytest.mark.parametrize("question,expected_ts", [
    ("What does the AMF do?", "TS 23.501"),
    ("Describe the PDU session establishment procedure", None),  # spans both TS
    ("What is network slicing?", "TS 23.501"),
    ("What happens during initial registration in NAS?", "TS 24.501"),
])
def test_in_domain_questions_are_answered_and_cited(pipeline, question, expected_ts):
    result = pipeline.answer(question)
    assert result["status"] == "ANSWERED", (
        f"Expected an answer for in-domain question {question!r}, got refusal: "
        f"{result.get('gate_reason')}"
    )
    assert len(result["citations"]) > 0, "Answered but produced no citations -- bug."
    if expected_ts:
        ts_numbers = {c["ts_number"] for c in result["citations"]}
        assert expected_ts in ts_numbers, f"Expected a {expected_ts} citation, got {ts_numbers}"


# ---- Out-of-domain / unsupported questions: MUST be refused, not guessed ----

@pytest.mark.parametrize("question", [
    "What is the capital of France?",
    "Who won the 2022 football World Cup?",
    "What is the boiling point of mercury?",
    "Describe the plot of Romeo and Juliet.",
])
def test_out_of_domain_questions_are_refused(pipeline, question):
    result = pipeline.answer(question)
    assert result["status"] in ("REFUSED_INSUFFICIENT_EVIDENCE", "REFUSED_NO_VERIFIED_CLAIMS"), (
        f"Out-of-domain question {question!r} should be refused, but got: {result['status']} "
        f"-> {result['answer']}"
    )
    assert "Not found" in result["answer"]
    assert len(result["citations"]) == 0


# ---- Every citation returned must trace back to a real chunk in the index ----

def test_every_citation_traces_to_a_real_indexed_chunk(pipeline):
    import json
    with open(config.METADATA_PATH) as f:
        all_chunk_ids = {c["chunk_id"] for c in json.load(f)}

    for question in ["What does the AMF do?", "What is network slicing?"]:
        result = pipeline.answer(question)
        for c in result["citations"]:
            assert c["chunk_id"] in all_chunk_ids, (
                f"Citation references chunk_id {c['chunk_id']} not present in the index -- "
                f"this would be an unverifiable / fabricated citation."
            )
