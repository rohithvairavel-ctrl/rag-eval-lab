# Attention vs RNNs
RNNs process tokens sequentially, making long-range dependency learning hard and preventing full parallelization.
Attention computes pairwise interactions across the sequence in parallel, improving gradient flow to distant tokens.
Transformers largely replaced RNNs for NLP at scale, though RNNs remain useful for small streaming models.
Hybrid models and state-space models (S4, Mamba) revisit efficient sequence modeling.
