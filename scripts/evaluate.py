#!/usr/bin/env python3
"""Run offline RAG evaluation harness → reports/metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    aggregate_metrics,
    answer_relevance,
    context_precision,
    context_recall,
    context_relevance_score,
    faithfulness,
    hit_rate_at_k,
    token_f1,
)
from src.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--index", type=Path, default=ROOT / "index")
    parser.add_argument("--eval-set", type=Path, default=ROOT / "data" / "eval_set.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "metrics.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not (args.index / "chunks.json").exists():
        raise SystemExit(
            f"Index not found at {args.index}. Run: python scripts/build_index.py"
        )

    eval_data = json.loads(args.eval_set.read_text(encoding="utf-8"))
    items = eval_data["items"]
    pipeline = RAGPipeline.from_index(args.index)

    rows = []
    for item in items:
        q = item["question"]
        gold = item["gold_answer"]
        relevant = item["relevant_doc_ids"]
        resp = pipeline.ask(q, top_k=args.top_k)
        retrieved_docs = [c.doc_id for c in resp.contexts]
        ctx_texts = [c.text for c in resp.contexts]

        row = {
            "id": item["id"],
            "question": q,
            "gold_answer": gold,
            "predicted_answer": resp.answer,
            "retrieved_doc_ids": retrieved_docs,
            "retrieved_chunk_ids": [c.chunk_id for c in resp.contexts],
            "relevant_doc_ids": relevant,
            "faithfulness": faithfulness(resp.answer, ctx_texts),
            "context_precision": context_precision(retrieved_docs, relevant),
            "context_recall": context_recall(retrieved_docs, relevant),
            "context_relevance": context_relevance_score(q, ctx_texts),
            "answer_relevance": answer_relevance(q, resp.answer),
            "answer_token_f1": token_f1(resp.answer, gold),
            "hit_rate": hit_rate_at_k(retrieved_docs, relevant),
        }
        rows.append(row)

    summary = aggregate_metrics(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": str(args.index),
        "eval_set": str(args.eval_set),
        "top_k": args.top_k,
        "retriever": "hybrid_bm25_tfidf_rrf",
        "generator": "extractive_tfidf",
        "summary": summary,
        "per_example": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = args.out.with_suffix(".md")
    lines = [
        "# RAG Eval Metrics",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Examples: **{summary['n_examples']}** · top_k={args.top_k}",
        "",
        "| Metric | Score |",
        "|--------|------:",
        f"| Faithfulness | {summary['faithfulness']:.4f} |",
        f"| Context Precision | {summary['context_precision']:.4f} |",
        f"| Context Recall | {summary['context_recall']:.4f} |",
        f"| Context Relevance | {summary['context_relevance']:.4f} |",
        f"| Answer Relevance | {summary['answer_relevance']:.4f} |",
        f"| Answer Token F1 | {summary['answer_token_f1']:.4f} |",
        f"| Hit Rate@k | {summary['hit_rate']:.4f} |",
        "",
        "## Per-example highlights",
        "",
    ]
    for r in rows:
        lines.append(
            f"- **{r['id']}** faithfulness={r['faithfulness']:.2f} "
            f"ctx_p={r['context_precision']:.2f} ctx_r={r['context_recall']:.2f} "
            f"ans_rel={r['answer_relevance']:.2f} f1={r['answer_token_f1']:.2f} "
            f"hit={r['hit_rate']:.0f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary": summary, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
