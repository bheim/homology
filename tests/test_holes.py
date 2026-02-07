"""Tests for structural hole detection."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from homology.holes import find_cluster_gaps, find_structural_holes
from homology.vault import Note


def _make_scenario():
    """Create a scenario with an obvious structural hole.

    Notes A, B, C form one cluster (linked).
    Notes D, E form another cluster (linked).
    A and D are semantically very similar but not linked — the structural hole.
    """
    notes = [
        Note(path=Path("A.md"), title="A", content="topic alpha", outgoing_links=["B", "C"]),
        Note(path=Path("B.md"), title="B", content="topic alpha related", outgoing_links=["A"]),
        Note(path=Path("C.md"), title="C", content="topic alpha adjacent", outgoing_links=["A"]),
        Note(path=Path("D.md"), title="D", content="topic alpha far", outgoing_links=["E"]),
        Note(path=Path("E.md"), title="E", content="topic beta", outgoing_links=["D"]),
    ]

    # Embeddings: A and D are very similar, others less so
    embeddings = np.array([
        [1.0, 0.0, 0.0],   # A
        [0.7, 0.7, 0.0],   # B
        [0.6, 0.0, 0.8],   # C
        [0.95, 0.05, 0.0],  # D — similar to A
        [0.0, 0.0, 1.0],   # E
    ])
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    link_graph = nx.Graph()
    for n in notes:
        link_graph.add_node(n.slug)
    link_graph.add_edge("a", "b")
    link_graph.add_edge("a", "c")
    link_graph.add_edge("d", "e")

    return notes, embeddings, link_graph


def test_finds_obvious_hole():
    notes, embeddings, link_graph = _make_scenario()
    holes = find_structural_holes(
        notes, embeddings, link_graph, top_k=10, min_similarity=0.3
    )
    # A-D should be the top hole (high similarity, disconnected)
    assert len(holes) > 0
    top = holes[0]
    pair = {top.note_a, top.note_b}
    assert pair == {"a", "d"}
    assert top.graph_distance is None  # disconnected
    assert top.semantic_similarity > 0.9


def test_directly_linked_not_returned():
    notes, embeddings, link_graph = _make_scenario()
    holes = find_structural_holes(
        notes, embeddings, link_graph, top_k=100, min_similarity=0.0
    )
    # A-B are directly linked, should not appear
    pairs = [{h.note_a, h.note_b} for h in holes]
    assert {"a", "b"} not in pairs
    assert {"d", "e"} not in pairs


def test_min_similarity_filters():
    notes, embeddings, link_graph = _make_scenario()
    holes_low = find_structural_holes(
        notes, embeddings, link_graph, top_k=100, min_similarity=0.0
    )
    holes_high = find_structural_holes(
        notes, embeddings, link_graph, top_k=100, min_similarity=0.9
    )
    assert len(holes_low) >= len(holes_high)


def test_cluster_gaps_basic():
    notes, embeddings, link_graph = _make_scenario()
    gaps = find_cluster_gaps(notes, embeddings, link_graph, top_k=5)
    # Should find at least one gap between the two clusters
    assert len(gaps) >= 1
    gap = gaps[0]
    assert gap.cross_link_count == 0  # no links between clusters
    assert gap.bridge_score > 0


def test_empty_vault():
    holes = find_structural_holes([], np.array([]).reshape(0, 0), nx.Graph())
    assert holes == []
