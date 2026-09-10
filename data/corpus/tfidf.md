# TF-IDF
TF-IDF weights a term by its frequency in a document times the log inverse document frequency across the corpus.
It downweights ubiquitous terms (the, and) and upweights distinctive terms. Classic baseline for search and classification.
Sparse high-dimensional vectors; cosine similarity ranks documents. Still competitive for keyword-heavy queries.
Modern dense embeddings capture synonyms better; hybrid TF-IDF/BM25 + dense often wins.
