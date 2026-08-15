# Deployment Guide

This app is a **single Docker container** — the FastAPI backend serves both
the JSON API (`/api/*`) and the built React frontend (everything else) on
one port. That means one deploy = one public link.

Before deploying anywhere:
1. Get a free Groq API key: https://console.groq.com (takes ~1 minute)
2. Decide whether you're deploying with the synthetic sample corpus (works
   immediately, good for testing the deploy itself) or your real 3GPP docs
   (drop `.docx` files into `data/raw/` before building the image).

**Never commit your `GROQ_API_KEY` to git.** Every option below sets it as a
platform secret/environment variable instead.

---

## Option A — Hugging Face Spaces (recommended: easiest, free, no card needed)

1. Create a new Space at https://huggingface.co/new-space → SDK: **Docker**.
2. Push this whole `3gpp_rag_chatbot/` folder as the Space's repo (it already
   has a `Dockerfile` at the root, which is what HF Spaces looks for).
3. In the Space's **Settings → Variables and secrets**, add:
   - `RAG_LLM_BACKEND` = `groq`
   - `RAG_GROQ_MODEL` = `llama-3.1-8b-instant`
   - `GROQ_API_KEY` = *(your key, as a Secret, not a public Variable)*
   - `RAG_EMBEDDING_BACKEND` = `tfidf` (keep this unless your Space has
     enough RAM for sentence-transformers — HF Spaces free CPU tier is
     usually fine for tfidf; switch later once real docs are in and it's working)
4. The Space builds the Dockerfile automatically. Once it shows "Running",
   your public link is `https://huggingface.co/spaces/<you>/<space-name>`.

---

## Option B — Render.com

1. Push this repo to GitHub.
2. In Render: **New → Web Service** → connect the repo → it should
   auto-detect `render.yaml` (included in this project).
3. In the service's **Environment** tab, set `GROQ_API_KEY` (the `sync: false`
   line in `render.yaml` means Render won't ask you to commit it — you set
   it directly in the dashboard).
4. Deploy. Free tier spins down when idle and cold-starts on the next
   request (~30-60s) — fine for a project demo, mention this if presenting live.
5. Your link: `https://<service-name>.onrender.com`

---

## Option C — Railway.app

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**. It detects the
   `Dockerfile` automatically.
3. In **Variables**, add `RAG_LLM_BACKEND=groq`, `RAG_GROQ_MODEL=llama-3.1-8b-instant`,
   `GROQ_API_KEY=<your key>`, `RAG_EMBEDDING_BACKEND=tfidf`.
4. Railway assigns a public domain automatically under **Settings → Networking
   → Generate Domain**.

---

## Building locally to sanity-check before you deploy (optional but recommended)

```bash
docker build -t 3gpp-rag .
docker run -p 8000:8000 \
  -e RAG_LLM_BACKEND=groq \
  -e GROQ_API_KEY=your_key_here \
  3gpp-rag
```
Then open http://localhost:8000 — you should see the same UI you'll get from
the public deployment. This catches Dockerfile/build issues before you burn
a deploy cycle on a hosting platform.

## Using your real 3GPP corpus instead of the sample docs

The `Dockerfile` builds the index **at image-build time** from whatever's in
`data/raw/`. So:
1. Put your `.docx` 3GPP spec files into `data/raw/` before building/pushing.
2. If you also want real embeddings (recommended for the graded submission,
   not just the sample-corpus demo), you need to switch the Dockerfile's
   pip install line back to the full `requirements.txt` (which includes
   `sentence-transformers`/`faiss-cpu` and pulls in PyTorch + several GB of
   CUDA wheels — expect a much bigger/slower build and image, and make sure
   your host has the disk space and RAM for it), then build with:
   ```bash
   docker build --build-arg RAG_EMBEDDING_BACKEND=sentence-transformers -t 3gpp-rag .
   ```
   By default (no changes), the image stays on the lightweight
   `requirements-web.txt` + TF-IDF, which is much cheaper to build and run
   and is a perfectly reasonable choice paired with Groq for the LLM.
3. Recalibrate `config.MIN_EVIDENCE_SCORE` against your real corpus before
   the final deploy — see the note in `config.py` and the main `README.md`.
