# Document Chunking for RAG
Chunking splits long documents into passages that fit embedding and context windows while preserving coherence.
Fixed-size windows with overlap are simple. Semantic chunking splits on headings, paragraphs, or embedding similarity breakpoints.
Too-small chunks lose context; too-large chunks dilute relevance and waste tokens.
Store chunk metadata (doc_id, title, section) for citation and filtering.
