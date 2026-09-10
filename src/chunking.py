"""Simple paragraph/heading-aware chunking for short FAQ docs."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return fallback.replace("_", " ").title()


def chunk_document(path: Path, max_chars: int = 900, overlap: int = 80) -> list[Chunk]:
    """Chunk a markdown/text file into overlapping windows on paragraph boundaries."""
    raw = path.read_text(encoding="utf-8").strip()
    doc_id = path.stem
    title = _title_from_text(raw, doc_id)
    body = re.sub(r"^#+\s+.*\n+", "", raw, count=1).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        paragraphs = [body] if body else [raw]

    chunks: list[Chunk] = []
    buf = ""
    part = 0
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}__{part}",
                    doc_id=doc_id,
                    title=title,
                    text=buf,
                    source_path=str(path),
                )
            )
            part += 1
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + "\n\n" + para
            else:
                buf = para
        else:
            for i in range(0, len(para), max_chars - overlap):
                piece = para[i : i + max_chars]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}__{part}",
                        doc_id=doc_id,
                        title=title,
                        text=piece,
                        source_path=str(path),
                    )
                )
                part += 1
            buf = ""
    if buf:
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}__{part}",
                doc_id=doc_id,
                title=title,
                text=buf,
                source_path=str(path),
            )
        )
    for c in chunks:
        c.text = f"{c.title}. {c.text}"
    return chunks


def load_corpus(corpus_dir: Path | str) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    paths = sorted(
        list(corpus_dir.glob("*.md"))
        + list(corpus_dir.glob("*.txt"))
    )
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_document(path))
    return chunks


def chunks_to_records(chunks: Iterable[Chunk]) -> list[dict]:
    return [c.to_dict() for c in chunks]
