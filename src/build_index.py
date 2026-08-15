"""
Step 3 + 4: Embedding Model and Vector Database.

Builds two indexes from the chunk store:
  1. A dense vector index (FAISS, or TF-IDF-as-vectors in demo mode) for
     semantic retrieval.
  2. A BM25 index for lexical retrieval (critical for 3GPP text: message
     names, IE names and clause numbers are exact-match keyword signals
     that dense embeddings alone often under-rank).

Both are combined at query time in retriever.py (hybrid retrieval).
"""

import os
import sys
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ingest import load_chunks


def _expand_abbrevs(text: str) -> str:
    """Append expanded forms of known 3GPP abbreviations so embeddings/BM25
    can match a user's plain-English phrasing against acronym-heavy spec text."""
    extra = []
    upper = text.upper()
    for abbr, full in config.ABBREVIATIONS.items():
        if abbr.upper() in upper or full.lower() in text.lower():
            extra.append(full)
    if extra:
        return text + "\n" + " ".join(extra)
    return text


class EmbeddingBackend:
    def __init__(self, mode=config.EMBEDDING_BACKEND):
        self.mode = mode
        if mode == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        elif mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self._fitted = False
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {mode}")

    def fit(self, texts):
        if self.mode == "tfidf":
            self.vectorizer.fit(texts)
            self._fitted = True

    def encode(self, texts):
        if self.mode == "sentence-transformers":
            import numpy as np
            embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(embs, dtype="float32")
        else:
            if not self._fitted:
                self.fit(texts)
            mat = self.vectorizer.transform(texts)
            import numpy as np
            dense = mat.toarray().astype("float32")
            # L2-normalize so inner product == cosine similarity, matching
            # how the FAISS/real-embedding path is used downstream.
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms


def build_vector_index(chunks, embedder: EmbeddingBackend):
    texts = [_expand_abbrevs(c["text"]) for c in chunks]
    vectors = embedder.encode(texts)

    if config.EMBEDDING_BACKEND == "sentence-transformers":
        import faiss
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner product on normalized vecs = cosine sim
        index.add(vectors)
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        print(f"[index] FAISS index -> {config.FAISS_INDEX_PATH} ({index.ntotal} vectors)")
    else:
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        with open(config.TFIDF_INDEX_PATH, "wb") as f:
            pickle.dump({"vectorizer": embedder.vectorizer, "vectors": vectors}, f)
        print(f"[index] TF-IDF index -> {config.TFIDF_INDEX_PATH} ({vectors.shape[0]} vectors)")


def build_bm25_index(chunks):
    from rank_bm25 import BM25Okapi
    tokenized = [_expand_abbrevs(c["text"]).lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[index] BM25 index -> {config.BM25_INDEX_PATH} ({len(chunks)} docs)")


def save_metadata(chunks):
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[index] metadata -> {config.METADATA_PATH}")


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found. Run `python -m src.ingest` first.")

    embedder = EmbeddingBackend()
    build_vector_index(chunks, embedder)
    build_bm25_index(chunks)
    save_metadata(chunks)

    if config.EMBEDDING_BACKEND == "tfidf":
        # persist the fitted vectorizer object reference used at query time
        pass


if __name__ == "__main__":
    main()
