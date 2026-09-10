#!/usr/bin/env python3
"""Build hybrid BM25 + TF-IDF index from data/corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import load_corpus, chunks_to_records
from src.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "corpus",
        help="Corpus directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "index",
        help="Output index directory",
    )
    args = parser.parse_args()

    chunks = load_corpus(args.corpus)
    if not chunks:
        raise SystemExit(f"No documents found in {args.corpus}")

    records = chunks_to_records(chunks)
    retriever = HybridRetriever(records)
    retriever.save(args.out)

    meta = {
        "n_docs": len({c.doc_id for c in chunks}),
        "n_chunks": len(chunks),
        "corpus": str(args.corpus),
        "index_dir": str(args.out),
        "retriever": "hybrid_bm25_tfidf_rrf",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Index written to {args.out}")


if __name__ == "__main__":
    main()
