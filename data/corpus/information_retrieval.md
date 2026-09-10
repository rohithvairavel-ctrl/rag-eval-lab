# Information Retrieval Metrics
Recall@k: fraction of relevant docs found in top-k. Precision@k: fraction of top-k that are relevant.
MRR (mean reciprocal rank) averages 1/rank of first relevant doc. nDCG accounts for graded relevance and position.
MAP averages precision at each relevant hit. Choose metrics that match user behavior (first-click vs exhaustive).
For RAG, retrieval recall@k is a leading indicator of answer quality.
