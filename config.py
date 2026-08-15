"""
Central configuration for the 3GPP RAG chatbot.

BACKEND MODES
-------------
EMBEDDING_BACKEND:
    "tfidf"                -> zero-download, pure scikit-learn TF-IDF vectors.
                               Used for local testing / demo without internet access.
    "sentence-transformers" -> real dense embeddings (recommended: BAAI/bge-large-en-v1.5).
                               Requires internet access to download the model once.

LLM_BACKEND:
    "echo"    -> deterministic mock LLM that just extracts/quotes the retrieved
                 evidence. Zero downloads, zero GPU. Used to prove the pipeline
                 wiring (retrieval -> gate -> generation -> citation) end-to-end.
    "ollama"  -> real local LLM (Llama 3 / Mistral) served via Ollama
                 (https://ollama.com). Requires `ollama serve` running locally
                 and a model pulled, e.g. `ollama pull llama3.1:8b`.

For the graded submission, switch both backends to the "real" option and index
your actual 3GPP corpus. The "tfidf" / "echo" combo exists purely so the full
pipeline can be demonstrated and unit-tested without network/model access.
"""

import os

# ---- Backends -----------------------------------------------------------
EMBEDDING_BACKEND = os.environ.get("RAG_EMBEDDING_BACKEND", "tfidf")   # "tfidf" | "sentence-transformers"
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "echo")                # "echo" | "ollama" | "groq"

SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-large-en-v1.5"
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("RAG_OLLAMA_HOST", "http://localhost:11434")

# Groq: free-tier hosted API for real Llama models, no local GPU/download
# needed -- used for the deployed public web app. Get a free key at
# https://console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("RAG_GROQ_MODEL", "openai/gpt-oss-20b")

# ---- Paths ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")          # put your real .docx/.pdf 3GPP specs here
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")    # synthetic demo specs (no download needed)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")

CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
TFIDF_INDEX_PATH = os.path.join(INDEX_DIR, "tfidf.pkl")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# ---- Chunking ---------------------------------------------------------
MAX_CHUNK_TOKENS = 400          # sub-chunk clauses longer than this
CHUNK_OVERLAP_TOKENS = 40

# ---- Retrieval ----------------------------------------------------------
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_RERANKED = 4

# ---- Evidence validation (the anti-hallucination gate) --------------------
# If the top reranked score is below this, we refuse to answer instead of
# calling the LLM.
#
# IMPORTANT: this threshold is on the SAME SCALE as whatever reranker/scorer
# is active (config.EMBEDDING_BACKEND / presence of a CrossEncoder), and that
# scale differs a lot between backends:
#   - demo mode (tfidf embeddings, no CrossEncoder available offline)
#     -> reranker falls back to lexical Jaccard overlap, typically 0.03-0.20
#   - production mode (sentence-transformers + cross-encoder/ms-marco-MiniLM)
#     -> cross-encoder outputs raw logits, typically -5 to +5, NOT 0-1
# The value below is calibrated for demo mode only (see
# tests/test_pipeline_demo.py, which prints the score separation). When you
# switch to production backends, RECALIBRATE this against a labelled dev set
# of known-good vs known-bad questions for your actual corpus -- do not reuse
# this number.
MIN_EVIDENCE_SCORE = 0.10
# Minimum lexical overlap (Jaccard on significant terms) between question and
# best chunk as a second, independent check (belt-and-braces against a single
# scorer being fooled).
MIN_LEXICAL_OVERLAP = 0.08

# 3GPP abbreviation expansion table (subset -- extend from TS 21.905 in prod)
ABBREVIATIONS = {
    "AMF": "Access and Mobility Management Function",
    "SMF": "Session Management Function",
    "UPF": "User Plane Function",
    "PDU": "Protocol Data Unit",
    "gNB": "next generation Node B",
    "NAS": "Non-Access Stratum",
    "RRC": "Radio Resource Control",
    "UE": "User Equipment",
    "5GC": "5G Core Network",
    "PLMN": "Public Land Mobile Network",
    "QoS": "Quality of Service",
    "NR": "New Radio",
}
