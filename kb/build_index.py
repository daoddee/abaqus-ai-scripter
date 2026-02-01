import os
import json
import glob
import re
from typing import List, Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

RAW_DIR = os.path.join("kb", "raw")
OUT_DIR = os.path.join("kb", "index")
os.makedirs(OUT_DIR, exist_ok=True)

CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(OUT_DIR, "tfidf_matrix.joblib")
META_PATH = os.path.join(OUT_DIR, "meta.json")

CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".log", ".csv", ".json", ".yaml", ".yml", ".html", ".htm"}


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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
        raise SystemExit(
            "No text-like files found in kb/raw. Copy your Abaqus docs/text files into kb/raw first."
        )

    print("Found %d files" % len(files))

    all_chunks: List[Dict] = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        raw = read_text_file(fp)
        if ext in {".html", ".htm"}:
            raw = strip_html(raw)

        chunks = chunk_text(raw, CHUNK_CHARS, CHUNK_OVERLAP)
        rel = os.path.relpath(fp, RAW_DIR)
        for i, c in enumerate(chunks):
            all_chunks.append(
                {"id": "%s::%d" % (rel, i), "source": rel, "chunk_index": i, "text": c}
            )

    print("Total chunks: %d" % len(all_chunks))

    texts = [c["text"] for c in all_chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=200000,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf, MATRIX_PATH)

    meta = {
        "backend": "tfidf",
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_files": len(files),
        "num_chunks": len(all_chunks),
        "raw_dir": RAW_DIR,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" - %s" % CHUNKS_PATH)
    print(" - %s" % VECTORIZER_PATH)
    print(" - %s" % MATRIX_PATH)
    print(" - %s" % META_PATH)


if __name__ == "__main__":
    main()
