import os
import json
import glob
import re
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAW_DIR = os.path.join("kb", "raw")
OUT_DIR = os.path.join("kb", "index")
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUT_DIR, "docs.faiss")
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
META_PATH = os.path.join(OUT_DIR, "meta.json")

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".log", ".csv", ".json", ".yaml", ".yml", ".html", ".htm"}


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def strip_html(text: str) -> str:
    # Very simple HTML stripper (good enough for many Abaqus HTML exports)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # keep paragraph-ish structure lightly, then compress
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def iter_source_files(raw_dir: str) -> List[str]:
    paths = []
    for p in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True):
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in TEXT_EXTS:
                paths.append(p)
    return sorted(paths)


def main():
    files = iter_source_files(RAW_DIR)
    if not files:
        raise SystemExit(f"No text-like files found in {RAW_DIR}. Add .txt/.md/.html etc first.")

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Found {len(files)} files")

    model = SentenceTransformer(MODEL_NAME)

    all_chunks: List[Dict] = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        raw = read_text_file(fp)
        if ext in {".html", ".htm"}:
            raw = strip_html(raw)

        chunks = chunk_text(raw, CHUNK_CHARS, CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "id": f"{os.path.relpath(fp, RAW_DIR)}::{i}",
                "source": os.path.relpath(fp, RAW_DIR),
                "chunk_index": i,
                "text": c
            })

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if normalized
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "dim": dim,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_files": len(files),
        "num_chunks": len(all_chunks),
        "raw_dir": RAW_DIR,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" -", INDEX_PATH)
    print(" -", CHUNKS_PATH)
    print(" -", META_PATH)


if __name__ == "__main__":
    main()
