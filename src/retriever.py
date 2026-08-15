"""
Steps 5-7: Query Processing, Hybrid Retrieval, Reranking.
"""

import os
import sys
import json
import pickle
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.build_index import EmbeddingBackend, _expand_abbrevs

TS_QUERY_RE = re.compile(r"\b(?:TS|TR)\s?(\d{2}\.\d{3})\b", re.IGNORECASE)


def rewrite_query(question: str) -> str:
    """Query rewriting for retrieval: expand abbreviations found in the
    question so we can match acronym-heavy spec prose."""
    return _expand_abbrevs(question)


def detect_ts_filter(question: str):
    m = TS_QUERY_RE.search(question)
    if m:
        return f"TS {m.group(1)}"
    return None


class Retriever:
    def __init__(self):
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = EmbeddingBackend()

        if config.EMBEDDING_BACKEND == "sentence-transformers":
            import faiss
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        else:
            with open(config.TFIDF_INDEX_PATH, "rb") as f:
                store = pickle.load(f)
            self.embedder.vectorizer = store["vectorizer"]
            self.embedder._fitted = True
            self.doc_vectors = store["vectors"]

        with open(config.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        # Reranker: cross-encoder in production, lightweight lexical overlap
        # scorer in demo mode (no model download required).
        self.reranker = None
        if config.EMBEDDING_BACKEND == "sentence-transformers":
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"[retriever] WARNING: reranker unavailable ({e}); "
                      f"falling back to lexical rerank")

    # ---- retrieval components ----------------------------------------

    def _vector_search(self, query: str, ts_filter=None, k=config.TOP_K_VECTOR):
        q_vec = self.embedder.encode([query])
        if config.EMBEDDING_BACKEND == "sentence-transformers":
            scores, idxs = self.index.search(q_vec, k * 3 if ts_filter else k)
            scores, idxs = scores[0], idxs[0]
        else:
            import numpy as np
            sims = (self.doc_vectors @ q_vec[0])
            idxs = np.argsort(-sims)[: k * 3 if ts_filter else k]
            scores = sims[idxs]

        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            c = self.chunks[idx]
            if ts_filter and c["ts_number"] != ts_filter:
                continue
            results.append((float(score), c))
            if len(results) >= k:
                break
        return results

    def _bm25_search(self, query: str, ts_filter=None, k=config.TOP_K_BM25):
        tokenized_q = query.lower().split()
        scores = self.bm25.get_scores(tokenized_q)
        import numpy as np
        idxs = np.argsort(-scores)[: k * 3 if ts_filter else k]
        results = []
        for idx in idxs:
            c = self.chunks[idx]
            if ts_filter and c["ts_number"] != ts_filter:
                continue
            results.append((float(scores[idx]), c))
            if len(results) >= k:
                break
        return results

    def _lexical_overlap(self, query: str, text: str) -> float:
        q_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        d_terms = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        if not q_terms or not d_terms:
            return 0.0
        return len(q_terms & d_terms) / len(q_terms | d_terms)

    def _rerank(self, query: str, candidates):
        """candidates: list of (score, chunk) already de-duplicated by chunk_id."""
        if not candidates:
            return []

        if self.reranker is not None:
            pairs = [(query, c["text"]) for _, c in candidates]
            ce_scores = self.reranker.predict(pairs)
            scored = list(zip(ce_scores, [c for _, c in candidates]))
        else:
            # demo-mode lexical rerank, normalized to a roughly comparable
            # 0-1 range so MIN_EVIDENCE_SCORE behaves sensibly either way.
            scored = [(self._lexical_overlap(query, c["text"]), c) for _, c in candidates]

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: config.TOP_K_RERANKED]

    # ---- public API -----------------------------------------------------

    def retrieve(self, question: str):
        query = rewrite_query(question)
        ts_filter = detect_ts_filter(question)

        vec_results = self._vector_search(query, ts_filter)
        bm25_results = self._bm25_search(query, ts_filter)

        # union + de-dup by chunk_id, keeping the max signal from either arm
        merged = {}
        for score, c in vec_results + bm25_results:
            cid = c["chunk_id"]
            if cid not in merged or score > merged[cid][0]:
                merged[cid] = (score, c)

        reranked = self._rerank(query, list(merged.values()))
        return reranked, ts_filter
