# Tokenization
Tokenizers split text into subword units for model input. BPE, WordPiece, and Unigram are common algorithms.
Vocabulary size trades coverage vs rare-token fragmentation. Special tokens mark BOS/EOS/PAD/UNK.
Token counts determine cost and context limits for LLMs. Multilingual tokenizers balance languages unevenly.
Preprocessing (lowercasing, Unicode normalization) must match training-time tokenization.
