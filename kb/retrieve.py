import os
import json
from typing import List, Dict

import joblib
import numpy as np

OUT_DIR = os.path.join("kb", "index")
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(OUT_DIR, "tfidf_matrix.joblib")

_vectorizer = None
_matrix = None
_chunks: List[Dict] = []


def _load():
    global _vectorizer, _matrix, _chunks
    if _vectorizer is not None:
        return

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MATRIX_PATH):
        raise RuntimeError("RAG TF-IDF index not found. Run: python kb/build_index.py (locally) and commit kb/index/")

    _chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _chunks.append(json.loads(line))

    _vectorizer = joblib.load(VECTORIZER_PATH)
    _matrix = joblib.load(MATRIX_PATH)


def retrieve(query: str, k: int = 6) -> List[Dict]:
    _load()
    q = (query or "").strip()
    if not q:
        return []

    qv = _vectorizer.transform([q])           # (1, vocab)
    scores = (qv @ _matrix.T).toarray()[0]    # cosine-ish since tfidf is L2-normalized by default

    if scores.size == 0:
        return []

    k = max(1, min(int(k), 20))
    # top-k indices
    idxs = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
    idxs = idxs[np.argsort(-scores[idxs])]

    results = []
    for idx in idxs.tolist():
        c = _chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "source": c.get("source"),
            "chunk_index": c.get("chunk_index"),
            "text": c.get("text"),
        })
    return results
