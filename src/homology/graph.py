"""Build explicit link and semantic similarity graphs from vault notes."""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from homology.vault import Note


def build_link_graph(notes: list[Note]) -> nx.Graph:
    """Build an undirected graph from explicit wikilinks between notes.

    Nodes are note slugs (lowercase filenames without extension).
    An edge exists between two notes if one links to the other.
    """
    slug_set = {n.slug for n in notes}
    slug_to_title = {n.slug: n.title for n in notes}

    G = nx.Graph()
    for note in notes:
        G.add_node(note.slug, title=slug_to_title.get(note.slug, note.slug))

    for note in notes:
        for target_raw in note.outgoing_links:
            target_slug = target_raw.lower().strip()
            if target_slug in slug_set and target_slug != note.slug:
                G.add_edge(note.slug, target_slug)

    return G


def build_semantic_graph(
    notes: list[Note],
    embeddings: np.ndarray,
    threshold: float = 0.0,
) -> nx.Graph:
    """Build a weighted graph from cosine similarity of note embeddings.

    Args:
        notes: List of notes (used for node labels).
        embeddings: (N, D) array of note embeddings.
        threshold: Minimum cosine similarity to include an edge.
            Default 0.0 keeps all non-negative similarity edges.

    Returns:
        Weighted undirected graph. Edge weight = cosine similarity.
    """
    sim_matrix = cosine_similarity(embeddings)

    G = nx.Graph()
    for note in notes:
        G.add_node(note.slug, title=note.title)

    n = len(notes)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                G.add_edge(notes[i].slug, notes[j].slug, weight=sim)

    return G


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the full pairwise cosine similarity matrix."""
    return cosine_similarity(embeddings)
