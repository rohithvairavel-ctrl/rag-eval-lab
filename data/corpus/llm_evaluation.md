# Evaluating LLM and RAG Systems
LLM eval includes automatic metrics (BLEU, ROUGE, BERTScore), LLM-as-judge, and human review.
RAG-specific metrics: context relevance/precision, faithfulness (answer grounded in context), answer relevance to the question.
RAGAS-style pipelines score each dimension. Offline labeled sets enable regression testing without paid APIs via heuristics or NLI models.
Always report retrieval@k alongside generation metrics — generation can't fix bad retrieval.
