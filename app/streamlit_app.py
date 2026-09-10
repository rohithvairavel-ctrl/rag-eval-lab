#!/usr/bin/env python3
"""Streamlit demo: ask questions, inspect retrieval, view eval metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    answer_relevance,
    context_relevance_score,
    faithfulness,
)
from src.pipeline import RAGPipeline

INDEX_DIR = ROOT / "index"
METRICS_PATH = ROOT / "reports" / "metrics.json"
EVAL_PATH = ROOT / "data" / "eval_set.json"


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    if not (INDEX_DIR / "chunks.json").exists():
        raise FileNotFoundError(
            "Index missing. Run: python scripts/build_index.py"
        )
    return RAGPipeline.from_index(INDEX_DIR)


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def main() -> None:
    st.set_page_config(
        page_title="RAG Eval Lab",
        page_icon="🔎",
        layout="wide",
    )
    st.title("🔎 RAG Eval Lab")
    st.caption(
        "Offline-first hybrid retrieval (BM25 + TF-IDF) · extractive QA · "
        "RAGAS-inspired evaluation harness"
    )

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    tab_ask, tab_eval = st.tabs(["Ask a question", "Eval set metrics"])

    with tab_ask:
        col_q, col_k = st.columns([4, 1])
        with col_q:
            question = st.text_input(
                "Question",
                value="What is the bias-variance tradeoff?",
                help="Ask about DS/ML interview concepts in the corpus.",
            )
        with col_k:
            top_k = st.slider("top_k", 1, 10, 5)

        if st.button("Retrieve & Answer", type="primary") or question:
            resp = pipeline.ask(question, top_k=top_k)
            ctx_texts = [c.text for c in resp.contexts]
            live = {
                "faithfulness": faithfulness(resp.answer, ctx_texts),
                "context_relevance": context_relevance_score(question, ctx_texts),
                "answer_relevance": answer_relevance(question, resp.answer),
            }

            m1, m2, m3 = st.columns(3)
            m1.metric("Faithfulness", f"{live['faithfulness']:.3f}")
            m2.metric("Context relevance", f"{live['context_relevance']:.3f}")
            m3.metric("Answer relevance", f"{live['answer_relevance']:.3f}")

            st.subheader("Answer")
            st.write(resp.answer)
            st.caption(f"Generator: `{resp.generation.method}`")

            st.subheader("Retrieved chunks")
            for c in resp.contexts:
                with st.expander(
                    f"#{c.rank} · {c.title} (`{c.doc_id}`) · RRF={c.score:.4f}"
                ):
                    st.write(c.text)

    with tab_eval:
        metrics = load_metrics()
        if metrics is None:
            st.warning("No reports/metrics.json yet. Run: python scripts/evaluate.py")
        else:
            summary = metrics["summary"]
            st.subheader("Aggregate scores (offline harness)")
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)
            c1.metric("Faithfulness", f"{summary['faithfulness']:.4f}")
            c2.metric("Context precision", f"{summary['context_precision']:.4f}")
            c3.metric("Context recall", f"{summary['context_recall']:.4f}")
            c4.metric("Context relevance", f"{summary['context_relevance']:.4f}")
            c5.metric("Answer relevance", f"{summary['answer_relevance']:.4f}")
            c6.metric("Answer token F1", f"{summary['answer_token_f1']:.4f}")
            if "hit_rate" in summary:
                st.metric("Hit rate@k", f"{summary['hit_rate']:.4f}")

            st.caption(
                f"n={summary['n_examples']} · generated_at={metrics.get('generated_at', '?')} · "
                f"retriever={metrics.get('retriever')}"
            )

            st.subheader("Per-example breakdown")
            rows = metrics.get("per_example", [])
            table = [
                {
                    "id": r["id"],
                    "question": r["question"][:60] + ("…" if len(r["question"]) > 60 else ""),
                    "faithfulness": round(r["faithfulness"], 3),
                    "ctx_precision": round(r["context_precision"], 3),
                    "ctx_recall": round(r["context_recall"], 3),
                    "ans_relevance": round(r["answer_relevance"], 3),
                    "token_f1": round(r["answer_token_f1"], 3),
                }
                for r in rows
            ]
            st.dataframe(table, use_container_width=True)

            if EVAL_PATH.exists():
                with st.expander("Eval set schema"):
                    st.code(
                        EVAL_PATH.read_text(encoding="utf-8")[:1200] + "\n…",
                        language="json",
                    )


if __name__ == "__main__":
    main()
