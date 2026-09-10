# Reranking in Retrieval
A two-stage pipeline retrieves many candidates cheaply (BM25/ANN), then reranks a shortlist with a stronger cross-encoder.
Cross-encoders jointly encode query+document for higher accuracy at higher latency.
Reciprocal Rank Fusion (RRF) merges ranked lists from hybrid retrievers without score calibration.
Reranking often yields larger gains than swapping embedding models alone.
