"""RAGAS-inspired offline metrics: faithfulness, context precision, answer relevance."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import tokenize


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
        "was", "were", "be", "as", "by", "with", "that", "this", "it", "from",
        "at", "into", "about", "how", "what", "when", "which", "who", "whom",
    }
    return {t for t in tokenize(text) if t not in stop and len(t) > 2}


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tfidf_cosine(a: str, b: str) -> float:
    vect = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))
    try:
        mat = vect.fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(mat[0], mat[1])[0, 0])


def answer_relevance(question: str, answer: str) -> float:
    """How well the answer addresses the question (lexical + TF-IDF hybrid)."""
    if not answer.strip():
        return 0.0
    q, a = _tokens(question), _tokens(answer)
    coverage = (len(q & a) / len(q)) if q else 0.0
    sim = _tfidf_cosine(question, answer)
    return float(0.45 * coverage + 0.55 * sim)


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer content words supported by retrieved context (NLI-lite)."""
    if not answer.strip():
        return 0.0
    ctx = " ".join(contexts)
    a_toks = _tokens(answer)
    c_toks = _tokens(ctx)
    if not a_toks:
        return 0.0
    supported = len(a_toks & c_toks) / len(a_toks)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
    if not sents:
        return float(supported)
    sent_scores = []
    for s in sents:
        st = _tokens(s)
        if not st:
            continue
        sent_scores.append(len(st & c_toks) / len(st))
    sent_mean = float(np.mean(sent_scores)) if sent_scores else supported
    return float(0.6 * supported + 0.4 * sent_mean)


def context_precision(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]
) -> float:
    """Precision of retrieved docs w.r.t. labeled relevant doc ids (deduped order)."""
    if not retrieved_doc_ids:
        return 0.0
    relevant = set(relevant_doc_ids)
    seen = set()
    ordered: list[str] = []
    for d in retrieved_doc_ids:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    hits = sum(1 for d in ordered if d in relevant)
    return hits / len(ordered)


def context_recall(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]
) -> float:
    if not relevant_doc_ids:
        return 1.0
    retrieved = set(retrieved_doc_ids)
    relevant = set(relevant_doc_ids)
    return len(retrieved & relevant) / len(relevant)


def context_relevance_score(question: str, contexts: Sequence[str]) -> float:
    """Blend token coverage and TF-IDF cosine between question and each chunk."""
    if not contexts:
        return 0.0
    q = _tokens(question)
    scores = []
    for c in contexts:
        ctoks = _tokens(c)
        cover = (len(q & ctoks) / len(q)) if q else 0.0
        sim = _tfidf_cosine(question, c)
        scores.append(0.5 * cover + 0.5 * sim)
    return float(np.mean(scores))


def hit_rate_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """1 if any relevant doc appears in retrieved list, else 0."""
    if not relevant_doc_ids:
        return 1.0
    return 1.0 if set(retrieved_doc_ids) & set(relevant_doc_ids) else 0.0


def aggregate_metrics(rows: list[dict]) -> dict:
    keys = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "context_relevance",
        "answer_relevance",
        "answer_token_f1",
        "hit_rate",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0
    out["n_examples"] = len(rows)
    return out
