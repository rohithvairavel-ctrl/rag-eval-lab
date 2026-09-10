"""TF-IDF dense vectors + optional sentence-transformers if installed."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


class TfidfEmbedder:
    """Offline dense-ish embeddings via TF-IDF (L2-normalized)."""

    def __init__(self, max_features: int = 8192, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            tokenizer=tokenize,
            token_pattern=None,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(list(texts))
        self._fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit before transform")
        mat = self.vectorizer.transform(list(texts))
        return normalize(mat, norm="l2", axis=1).toarray().astype(np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


def try_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a sentence-transformers model if importable, else None."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name)
    except Exception:
        return None
