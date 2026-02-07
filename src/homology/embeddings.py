"""Generate embeddings for notes. Supports OpenAI API and local sentence-transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from homology.vault import Note


class Embedder(ABC):
    """Base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) array of embeddings for the given texts."""

    def embed_notes(self, notes: list[Note]) -> np.ndarray:
        """Embed a list of notes using their content."""
        texts = [_note_to_text(n) for n in notes]
        return self.embed(texts)


class OpenAIEmbedder(Embedder):
    """Embed via OpenAI's API (requires OPENAI_API_KEY env var)."""

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([item.embedding for item in response.data])
        return np.array(all_embeddings)


class LocalEmbedder(Embedder):
    """Embed locally using sentence-transformers (no API key needed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def get_embedder(backend: str = "openai", **kwargs) -> Embedder:
    """Factory for embedder backends.

    Args:
        backend: "openai" or "local"
    """
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    elif backend == "local":
        return LocalEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")


def _note_to_text(note: Note) -> str:
    """Convert a note into a string suitable for embedding.

    Uses title + content. Truncates to ~8000 tokens worth of chars as a rough
    safeguard for API limits.
    """
    text = f"{note.title}\n\n{note.content}"
    # Rough truncation: ~4 chars per token, 8191 token limit
    max_chars = 8000 * 4
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
