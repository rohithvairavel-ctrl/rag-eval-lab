"""End-to-end RAG pipeline: retrieve + generate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .generator import ExtractiveGenerator, GenerationResult
from .retriever import HybridRetriever, RetrievedChunk


@dataclass
class RAGResponse:
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    generation: GenerationResult

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": [c.to_dict() for c in self.contexts],
            "generation": self.generation.to_dict(),
        }


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: ExtractiveGenerator | None = None):
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()

    @classmethod
    def from_index(cls, index_dir: Path | str) -> "RAGPipeline":
        return cls(HybridRetriever.load(index_dir))

    def ask(self, question: str, top_k: int = 5) -> RAGResponse:
        contexts = self.retriever.retrieve(question, top_k=top_k)
        gen = self.generator.generate(question, contexts)
        return RAGResponse(
            question=question,
            answer=gen.answer,
            contexts=contexts,
            generation=gen,
        )
