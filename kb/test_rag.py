import sys
from kb.retrieve import retrieve

def main():
    if len(sys.argv) < 2:
        print("Usage: python kb/test_rag.py \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    hits = retrieve(query, k=6)

    print("\nQUERY:", query)
    print("=" * 80)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.4f} source={h['source']} chunk={h['chunk_index']}")
        print("-" * 80)
        print(h["text"][:1200])
        if len(h["text"]) > 1200:
            print("... (truncated)")
    print("\nDone.")

if __name__ == "__main__":
    main()
