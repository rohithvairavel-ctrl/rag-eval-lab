# Transformers and Attention
Transformers replace recurrence with self-attention, enabling parallel sequence processing and long-range dependencies.
Scaled dot-product attention: softmax(QKᵀ / √d) V. Multi-head attention runs several projections in parallel.
Positional encodings inject order information. Encoder-decoder and decoder-only (GPT-style) variants dominate NLP and multimodal models.
Complexity is O(n²) in sequence length; sparse and linear attention variants reduce this cost.
