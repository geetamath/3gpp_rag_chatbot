---
title: 3GPP RAG Chatbot
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: gradio_app.py
pinned: false
---

# 3GPP RAG Chatbot — Low/Near-Zero Hallucination

A retrieval-augmented chatbot that answers questions **only** from ingested
3GPP specification documents, with a deterministic **evidence-validation
gate** that refuses to answer rather than guess when retrieval confidence
is low. Every answer carries a TS number, release, clause, and page citation
that is verified — not just prompted for.

## Why this isn't "just RAG"

Prompting an LLM with "only use the context" is not sufficient to claim
near-zero hallucination — LLMs drift off context regardless of instructions.
This project enforces groundedness with **code, not prompts alone**, at two
points:

1. **Evidence gate (pre-generation)** — `src/evidence_gate.py`. Runs BEFORE
   the LLM is called. Uses two independent, non-generative signals
   (reranker score + query-term coverage) to decide whether there's enough
   evidence to even attempt an answer. If not: refuse, no LLM call happens.
2. **Claim verification (post-generation)** — `src/verify.py`. After the LLM
   answers, every claimed sentence is checked for lexical support against
   the specific chunk it cites. Unsupported claims are stripped from the
   final answer, not shipped.

## Architecture

```
3GPP Specs (.docx/.pdf)
   -> Ingestion (clause-aware parsing, table extraction)      [src/ingest.py]
   -> Chunking + metadata (TS number, release, clause, page)  [src/ingest.py]
   -> Embeddings (bge / TF-IDF demo)                          [src/build_index.py]
   -> Vector index (FAISS / TF-IDF) + BM25 index               [src/build_index.py]
   -> Query processing (abbreviation expansion, TS detection)  [src/retriever.py]
   -> Hybrid retrieval (vector + BM25, merged)                 [src/retriever.py]
   -> Reranker (cross-encoder / lexical fallback)               [src/retriever.py]
   -> Evidence gate (accept / refuse)                           [src/evidence_gate.py]
   -> LLM (Ollama Llama/Mistral / mock echo)                    [src/llm_client.py]
   -> Claim verification (strip unsupported claims)             [src/verify.py]
   -> Grounded answer + citations                                [src/pipeline.py]
```

## Two backend modes

| | Demo mode (default, zero downloads) | Production mode |
|---|---|---|
| Embeddings | TF-IDF (`scikit-learn`) | `sentence-transformers` (BAAI/bge-large-en-v1.5) |
| Vector index | in-memory TF-IDF matrix | FAISS |
| Reranker | lexical (Jaccard) fallback | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | deterministic extractive "echo" (quotes evidence sentences only) | Llama 3 / Mistral via **Ollama**, running locally |
| Purpose | prove pipeline wiring + gate logic offline | your actual graded submission |

Demo mode exists so the entire pipeline — ingestion, chunking, hybrid
retrieval, evidence gating, generation, verification — can be run and tested
without internet access or a GPU. **Switch to production mode for your
actual submission**, since a graded chatbot should use a real LLM.

## Setup

```bash
pip install -r requirements.txt

# For production mode, also install Ollama and pull a model:
#   https://ollama.com
ollama pull llama3.1:8b
ollama serve
```

## Running it

### 1. Put your real 3GPP documents in `data/raw/`

Download `.docx` versions from the 3GPP specification portal
(https://www.3gpp.org/specifications). `.docx` is strongly preferred over
`.pdf` — 3GPP publishes both, and `.docx` preserves heading/table structure
far better, which directly reduces chunking-induced hallucination.
If `data/raw/` is empty, the pipeline falls back to the synthetic sample
corpus in `data/sample/` (two short excerpts styled on TS 23.501 / TS 24.501,
used only for demoing the pipeline — not a substitute for real specs).

### 2. Ingest, index, and run

```bash
# Demo mode (no downloads, uses sample corpus if data/raw/ is empty)
export RAG_EMBEDDING_BACKEND=tfidf
export RAG_LLM_BACKEND=echo
python3 -m src.ingest
python3 -m src.build_index
python3 app.py                      # interactive chat
python3 app.py --q "What does the AMF do?"

# Production mode (real docs, real embeddings, real local LLM)
export RAG_EMBEDDING_BACKEND=sentence-transformers
export RAG_LLM_BACKEND=ollama
export RAG_OLLAMA_MODEL=llama3.1:8b
python3 -m src.ingest
python3 -m src.build_index
python3 app.py
```

### 3. Run the automated evaluation suite

```bash
python3 -m pytest tests/ -v
```

## Evaluation results (demo mode, synthetic sample corpus)

9 automated tests in `tests/test_pipeline_demo.py`:

- 4 in-domain questions → must be **answered with a valid TS citation**
- 4 clearly out-of-domain questions (capital of France, football World Cup,
  boiling point of mercury, Romeo and Juliet) → must be **refused**
- 1 citation-integrity check → every returned citation must trace to a real
  indexed chunk_id

**Result: 8/9 passed.** The one failure ("What is network slicing?", scored
0.075 vs the 0.10 gate threshold) is a known, documented limitation of demo
mode: TF-IDF + lexical-overlap reranking has weak discriminative power on a
tiny 13-chunk synthetic corpus, where in-domain and out-of-domain scores sit
close together. This is *exactly* the argument for switching to
sentence-transformer embeddings + a cross-encoder reranker (production mode)
before your final submission — real dense embeddings separate these cases
much more cleanly. **Don't quietly raise/lower the threshold to force this
test to pass; recalibrate it against your real corpus's score distribution
once you're in production mode** (see the comment in `config.py`).

## What to do before submitting

1. Drop your actual 10-30 TS documents (`.docx`) into `data/raw/`.
2. Switch both backends to production mode (see above).
3. Recalibrate `config.MIN_EVIDENCE_SCORE` — build a small labelled set of
   20-30 questions (half answerable from your corpus, half not) and pick the
   threshold that best separates them. This is the number you'll want to
   defend/justify in your report — "how did you choose the confidence
   threshold" is a very likely review question.
4. Re-run `tests/test_pipeline_demo.py` against your real corpus (swap the
   sample-corpus questions for ones relevant to your actual TS documents).

## Full-stack web app (FastAPI + React)

There's also a deployable web version in `webapp/` — a FastAPI backend
(`webapp/backend/main.py`) wrapping this same pipeline behind a JSON API,
and a React frontend (`webapp/frontend/`) with a dark evidence-workbench UI:
chat, live citations, source library, and retrieval telemetry, all backed by
real data (no hardcoded chunks).

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

To deploy it publicly with a real LLM (Groq's free hosted Llama), see
**`DEPLOYMENT.md`** — covers Hugging Face Spaces, Render, and Railway, all
using the included `Dockerfile` (single container serves both API and UI).

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
tests/test_pipeline_demo.py automated gate-correctness tests
data/sample/                synthetic offline demo corpus (2 files)
data/raw/                   <- put your real 3GPP .docx files here
```
