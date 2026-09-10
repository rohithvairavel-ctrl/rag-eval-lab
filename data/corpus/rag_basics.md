# Retrieval-Augmented Generation (RAG)
RAG retrieves relevant documents and conditions a generator on that context, reducing hallucination and enabling private/knowledge updates without retraining.
Pipeline: chunk corpus → embed → index → retrieve top-k → prompt LLM with context → generate answer.
Failure modes: bad chunking, retrieval misses, context stuffing, and unfaithful generation.
Evaluation should cover retrieval quality (recall@k, MRR) and generation quality (faithfulness, answer relevance).
