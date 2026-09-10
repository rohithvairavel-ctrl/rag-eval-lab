# Cosine Similarity
Cosine similarity is the cosine of the angle between two vectors: (A·B) / (||A|| ||B||). Range is [-1, 1] for unrestricted vectors, often [0, 1] for TF-IDF.
It ignores magnitude and focuses on orientation — useful for text where document length varies.
In retrieval, dense embeddings are often L2-normalized so cosine equals dot product.
Related metrics: Euclidean distance, Manhattan, and learned similarity functions.
