import os
import json
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.join("kb", "index")
INDEX_PATH = os.path.join(OUT_DIR, "docs.faiss")
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_index = None
_chunks: List[Dict] = []
_model = None


def _load():
    global _index, _chunks, _model
    if _index is not None:
        return
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("RAG index not found. Run: python kb/build_index.py")

    _index = faiss.read_index(INDEX_PATH)
    _chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _chunks.append(json.loads(line))

    _model = SentenceTransformer(MODEL_NAME)


def retrieve(query: str, k: int = 6) -> List[Dict]:
    _load()
    q = (query or "").strip()
    if not q:
        return []

    q_emb = _model.encode([q], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = _index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(_chunks):
            continue
        c = _chunks[idx]
        results.append(
            {
                "score": float(score),
                "source": c.get("source"),
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text"),
            }
        )
    return results
