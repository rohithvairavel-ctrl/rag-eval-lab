"""Offline extractive + template generator (no API keys)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import tokenize
from .retriever import RetrievedChunk


@dataclass
class GenerationResult:
    answer: str
    method: str
    support_sentences: list[str]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "method": self.method,
            "support_sentences": self.support_sentences,
        }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


class ExtractiveGenerator:
    """Select top sentences from retrieved chunks most similar to the question."""

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                answer="I don't know based on the retrieved context.",
                method="extractive_empty",
                support_sentences=[],
            )

        sentences: list[str] = []
        for c in contexts:
            for s in _split_sentences(c.text):
                sentences.append(s)
        if not sentences:
            text = contexts[0].text.strip()
            return GenerationResult(
                answer=text[:500],
                method="extractive_chunk_fallback",
                support_sentences=[text[:300]],
            )

        vect = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))
        mat = vect.fit_transform(sentences + [question])
        q = mat[-1]
        sims = (mat[:-1] @ q.T).toarray().ravel()
        order = np.argsort(sims)[::-1]

        picked: list[str] = []
        seen = set()
        for idx in order:
            s = sentences[int(idx)]
            key = s.lower()[:80]
            if key in seen:
                continue
            if sims[idx] <= 0:
                break
            seen.add(key)
            picked.append(s)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            picked = sentences[: self.max_sentences]

        sources = ", ".join(sorted({c.doc_id for c in contexts[:3]}))
        lead = f"Based on the retrieved notes ({sources}): "
        answer = lead + " ".join(picked)
        return GenerationResult(
            answer=answer,
            method="extractive_tfidf",
            support_sentences=picked,
        )
