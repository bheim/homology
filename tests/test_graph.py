"""Tests for graph construction."""

import numpy as np
import pytest

from homology.graph import build_link_graph, build_semantic_graph, similarity_matrix
from homology.vault import Note


def _make_notes():
    """Create simple test notes without actual files."""
    from pathlib import Path
    from unittest.mock import MagicMock

    notes = []
    specs = [
        ("A.md", "Note A", ["B", "C"], []),
        ("B.md", "Note B", ["A"], []),
        ("C.md", "Note C", ["A"], []),
        ("D.md", "Note D", [], []),  # isolated
    ]
    for filename, title, links, _ in specs:
        note = Note(
            path=Path(filename),
            title=title,
            content=f"Content of {title}",
            outgoing_links=links,
        )
        notes.append(note)
    return notes


def test_build_link_graph_nodes():
    notes = _make_notes()
    G = build_link_graph(notes)
    assert set(G.nodes) == {"a", "b", "c", "d"}


def test_build_link_graph_edges():
    notes = _make_notes()
    G = build_link_graph(notes)
    assert G.has_edge("a", "b")
    assert G.has_edge("a", "c")
    assert not G.has_edge("b", "c")
    assert G.degree("d") == 0  # isolated


def test_build_semantic_graph():
    notes = _make_notes()
    # Create embeddings where A and D are most similar
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.9, 0.1, 0.0],  # similar to A
    ])
    G = build_semantic_graph(notes, embeddings, threshold=0.5)
    # A-D should be connected (high similarity)
    assert G.has_edge("a", "d")
    # B-C should not (orthogonal)
    assert not G.has_edge("b", "c")


def test_semantic_graph_threshold():
    notes = _make_notes()
    embeddings = np.random.default_rng(42).random((4, 8))
    G_low = build_semantic_graph(notes, embeddings, threshold=0.0)
    G_high = build_semantic_graph(notes, embeddings, threshold=0.99)
    assert G_low.number_of_edges() >= G_high.number_of_edges()


def test_similarity_matrix_shape():
    emb = np.random.default_rng(0).random((5, 10))
    sim = similarity_matrix(emb)
    assert sim.shape == (5, 5)
    # Diagonal should be ~1
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)
