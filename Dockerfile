# --- Stage 1: build the React frontend ------------------------------------
FROM node:20-slim AS frontend-build
WORKDIR /app/webapp/frontend
COPY webapp/frontend/package.json ./
RUN npm install
COPY webapp/frontend/ ./
RUN npm run build

# --- Stage 2: Python backend + built frontend ------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

# System deps for python-docx/pypdf (none heavy needed) + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend Python deps first (better layer caching)
# NOTE: uses requirements-web.txt, a MINIMAL set for the deployed web app
# (Groq LLM + TF-IDF embeddings by default) -- NOT the full requirements.txt,
# which includes sentence-transformers/faiss-cpu/gradio and would pull in
# several GB of PyTorch + CUDA wheels that this deployment doesn't use.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code
COPY config.py .
COPY src/ ./src/
COPY webapp/backend/ ./webapp/backend/
COPY data/sample/ ./data/sample/
# data/raw/ is where you'd normally add real 3GPP docs at build time --
# create it empty here so ingest.py's directory-listing check doesn't fail
RUN mkdir -p data/raw data/processed data/index

# Bring in the built frontend from stage 1
COPY --from=frontend-build /app/webapp/frontend/build ./webapp/frontend/build

# Build the index at image-build time so the container starts ready to serve.
# Uses demo mode by default (zero downloads at build time, matches the
# minimal requirements-web.txt above). If you want real sentence-transformer
# embeddings, you must ALSO change the pip install step above to use the
# full requirements.txt (which pulls in PyTorch + several GB of CUDA wheels)
# -- don't just flip this ARG without doing that, the import will fail.
ARG RAG_EMBEDDING_BACKEND=tfidf
ARG RAG_LLM_BACKEND=echo
ENV RAG_EMBEDDING_BACKEND=${RAG_EMBEDDING_BACKEND}
ENV RAG_LLM_BACKEND=${RAG_LLM_BACKEND}
RUN python -m src.ingest && python -m src.build_index

# Runtime env (LLM backend/API keys are typically overridden at deploy time,
# e.g. Render/HF Spaces "Secrets" -- do NOT bake real API keys into the image)
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn webapp.backend.main:app --host 0.0.0.0 --port ${PORT}"]
