"""Hybrid BM25 + TF-IDF dense retriever with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import TfidfEmbedder, tokenize


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "score": float(self.score),
            "rank": int(self.rank),
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """RRF: score(d) = sum 1/(k + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_key in enumerate(ranked, start=1):
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[dict], embedder: TfidfEmbedder | None = None):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.id_to_chunk = {c["chunk_id"]: c for c in chunks}
        texts = [c["text"] for c in chunks]
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.embedder = embedder or TfidfEmbedder()
        self.dense_matrix = self.embedder.fit_transform(texts)

    def _bm25_rank(self, query: str, top_n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_n]
        return [self.chunk_ids[i] for i in order if scores[i] > 0] or [
            self.chunk_ids[i] for i in order
        ]

    def _dense_rank(self, query: str, top_n: int) -> list[str]:
        q = self.embedder.transform([query])[0]
        sims = self.dense_matrix @ q
        order = np.argsort(sims)[::-1][:top_n]
        return [self.chunk_ids[i] for i in order]

    def retrieve(
        self, query: str, top_k: int = 5, candidate_pool: int = 30
    ) -> list[RetrievedChunk]:
        bm25_list = self._bm25_rank(query, candidate_pool)
        dense_list = self._dense_rank(query, candidate_pool)
        fused = reciprocal_rank_fusion([bm25_list, dense_list])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            c = self.id_to_chunk[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=c["doc_id"],
                    title=c["title"],
                    text=c["text"],
                    score=score,
                    rank=rank,
                )
            )
        return results

    def save(self, index_dir: Path | str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, indent=2), encoding="utf-8"
        )
        joblib.dump(
            {
                "embedder": self.embedder,
                "dense_matrix": self.dense_matrix,
                "chunk_ids": self.chunk_ids,
            },
            index_dir / "dense.joblib",
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "HybridRetriever":
        index_dir = Path(index_dir)
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        blob = joblib.load(index_dir / "dense.joblib")
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.chunk_ids = blob["chunk_ids"]
        obj.id_to_chunk = {c["chunk_id"]: c for c in chunks}
        obj.embedder = blob["embedder"]
        obj.dense_matrix = blob["dense_matrix"]
        tokenized = [tokenize(c["text"]) for c in chunks]
        obj.bm25 = BM25Okapi(tokenized)
        return obj
