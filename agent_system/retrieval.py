"""A dependency-free BM25 retriever for local text and Markdown notes."""

from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from math import log


class LocalRetriever:
    STOP_WORDS = {
        "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
        "do", "does", "for", "from", "how", "i", "in", "is", "it", "of",
        "on", "or", "that", "the", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "with",
    }

    def __init__(self, corpus_path: Path, *, limit: int = 4) -> None:
        self.limit = limit
        paths = sorted(
            path for path in corpus_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ) if corpus_path.is_dir() else [corpus_path]
        self.documents: list[dict[str, str | Counter[str] | int]] = []
        for path in paths:
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            for index, paragraph in enumerate(path.read_text(encoding="utf-8").split("\n\n"), 1):
                paragraph = paragraph.strip()
                if paragraph:
                    self.documents.append({
                        "id": f"{path.name}#{index}",
                        "source": str(path),
                        "text": paragraph,
                        "terms": Counter(self._terms(paragraph)),
                        "length": len(self._terms(paragraph)),
                    })

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            word for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 2 and word not in LocalRetriever.STOP_WORDS
        }

    def search(self, query: str) -> list[dict[str, str | int]]:
        terms = self._terms(query)
        if not terms or not self.documents:
            return []

        # BM25 rewards rare, repeated query terms while normalizing for passage length.
        count = len(self.documents)
        avg_length = sum(int(doc["length"]) for doc in self.documents) / count
        doc_freq = {
            term: sum(1 for doc in self.documents if term in doc["terms"])
            for term in terms
        }
        ranked: list[tuple[float, dict[str, str | Counter[str] | int]]] = []
        for document in self.documents:
            frequencies = document["terms"]
            length = int(document["length"])
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = log(1 + (count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / avg_length)
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: -item[0])
        return [
            {
                "id": str(document["id"]),
                "source": str(document["source"]),
                "text": str(document["text"]),
                "score": round(score, 3),
            }
            for score, document in ranked[: self.limit]
        ]
