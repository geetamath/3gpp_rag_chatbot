# 3GPP RAG Chatbot - Low/Near-Zero Hallucination

A retrieval-augmented chatbot that answers questions **only** from ingested
3GPP specification documents, with a deterministic **evidence-validation
gate** that refuses to answer rather than guess when retrieval confidence
is low. Every answer carries a TS number, release, and clause citation
that is verified - not just prompted for.

**Live app:** https://threegpp-rag-chatbot-6yy4.onrender.com
(Free-tier Render instance - may take up to ~50s to wake up on first request.)

Indexed corpus: TS 23.501 and TS 23.502 (Release 20), 3,844 clause-level
chunks. LLM: Groq-hosted Llama (`openai/gpt-oss-20b`).

## Why this isn't "just RAG"

Prompting an LLM with "only use the context" is not sufficient to claim
near-zero hallucination - LLMs drift off context regardless of instructions.
This project enforces groundedness with **code, not prompts alone**, at two
points:

1. **Evidence gate (pre-generation)** - `src/evidence_gate.py`. Runs BEFORE
   the LLM is called. Uses two independent, non-generative signals
   (reranker score + query-term coverage) to decide whether there's enough
   evidence to even attempt an answer. If not: refuse, no LLM call happens.
2. **Claim verification (post-generation)** - `src/verify.py`. After the LLM
   answers, every claimed sentence is checked for lexical support against
   the specific chunk it cites. Unsupported claims are stripped from the
   final answer, not shipped.

## Architecture

```
3GPP Specs (.docx/.pdf)
   -> Ingestion (clause-aware parsing, table extraction)        [src/ingest.py]
   -> Chunking + metadata (TS number, release, clause, page)    [src/ingest.py]
   -> Embeddings (bge / TF-IDF demo)                            [src/build_index.py]
   -> Vector index (FAISS / TF-IDF) + BM25 index                [src/build_index.py]
   -> Query processing (abbreviation expansion, TS detection)   [src/retriever.py]
   -> Hybrid retrieval (vector + BM25, merged)                  [src/retriever.py]
   -> Reranker (cross-encoder / lexical fallback)               [src/retriever.py]
   -> Evidence gate (accept / refuse)                           [src/evidence_gate.py]
   -> LLM (Groq-hosted Llama / Ollama / mock echo)              [src/llm_client.py]
   -> Claim verification (strip unsupported claims)             [src/verify.py]
   -> Grounded answer + citations                               [src/pipeline.py]
```

## Two backend modes

| | Demo mode | Production mode (deployed) |
|---|---|---|
| Embeddings | TF-IDF (`scikit-learn`) | TF-IDF, or `sentence-transformers` (BAAI/bge-large-en-v1.5) |
| Vector index | in-memory TF-IDF matrix | TF-IDF, or FAISS |
| Reranker | lexical (Jaccard) fallback | lexical, or `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | deterministic extractive "echo" (quotes evidence sentences only) | Groq-hosted Llama (deployed), or Ollama Llama 3 / Mistral (local) |
| Purpose | pipeline wiring + gate logic, testable offline with zero downloads | what's actually running behind the live app |

Demo mode exists so the full pipeline - ingestion, chunking, hybrid
retrieval, evidence gating, generation, verification - can be run and tested
without internet access or a GPU. The deployed app runs in production mode:
TF-IDF + BM25 hybrid retrieval over the real indexed corpus, with Groq
serving generation.

## Running it locally

```bash
pip install -r requirements.txt
```

```bash
# Demo mode - zero downloads, uses data/sample/ if data/raw/ is empty
export RAG_EMBEDDING_BACKEND=tfidf
export RAG_LLM_BACKEND=echo
python3 -m src.ingest
python3 -m src.build_index
python3 app.py                      # interactive chat
python3 app.py --q "What does the AMF do?"
```

```bash
# Production mode - real docs in data/raw/, real embeddings, local LLM via Ollama
export RAG_EMBEDDING_BACKEND=sentence-transformers
export RAG_LLM_BACKEND=ollama
export RAG_OLLAMA_MODEL=llama3.1:8b
ollama pull llama3.1:8b
ollama serve
python3 -m src.ingest
python3 -m src.build_index
python3 app.py
```

Run the automated evaluation suite:
```bash
python3 -m pytest tests/ -v
```

## Evaluation results (demo mode, synthetic sample corpus)

9 automated tests in `tests/test_pipeline_demo.py`:
- 4 in-domain questions -> must be **answered with a valid TS citation**
- 4 clearly out-of-domain questions (capital of France, football World Cup,
  boiling point of mercury, Romeo and Juliet) -> must be **refused**
- 1 citation-integrity check -> every returned citation must trace to a real
  indexed chunk_id

**Result: 8/9 passed.** The one failure ("What is network slicing?", scored
0.075 vs the 0.10 gate threshold) is a known limitation of demo mode:
TF-IDF + lexical-overlap reranking has weak discriminative power on the
tiny 13-chunk synthetic corpus, where in-domain and out-of-domain scores sit
close together. Dense embeddings (production mode) separate these cases
more cleanly. The threshold in `config.MIN_EVIDENCE_SCORE` was left as-is
rather than tuned to force this one test to pass; see the comment in
`config.py` for how it should be recalibrated against a real corpus's score
distribution.

## Full-stack web app (FastAPI + React)

The deployed app is a FastAPI backend (`webapp/backend/main.py`) wrapping
this same pipeline behind a JSON API, and a React frontend
(`webapp/frontend/`) with a dark evidence-workbench UI: chat, live
citations, source library, and retrieval telemetry, all backed by real
data (no hardcoded chunks).

To run it locally:
```bash
# terminal 1 -- backend
export RAG_EMBEDDING_BACKEND=tfidf RAG_LLM_BACKEND=echo
python -m src.ingest && python -m src.build_index
uvicorn webapp.backend.main:app --reload --port 8000

# terminal 2 -- frontend (dev mode, hot reload)
cd webapp/frontend
npm install
npm start   # opens http://localhost:3000, talks to :8000 via .env.local
```

Deployment (Render, via the included `Dockerfile`) is documented in
`DEPLOYMENT.md`, which also covers Hugging Face Spaces and Railway as
alternatives.

## File map

```
config.py                 all tunables in one place
src/ingest.py              Steps 1-2: parsing + clause-aware chunking
src/build_index.py         Steps 3-4: embeddings + vector/BM25 index build
src/retriever.py           Steps 5-7: query rewriting, hybrid retrieval, rerank
src/evidence_gate.py       Step 8: the anti-hallucination gate
src/llm_client.py          Step 9: LLM call, forced structured JSON + citations
src/verify.py              Step 10: post-generation claim verification
src/pipeline.py            orchestrates all of the above
app.py                     CLI chat entry point
webapp/                    FastAPI backend + React frontend (the deployed app)
tests/test_pipeline_demo.py automated gate-correctness tests
data/sample/                synthetic offline demo corpus (2 files)
data/raw/                   real 3GPP .docx source documents (TS 23.501, TS 23.502)
```
