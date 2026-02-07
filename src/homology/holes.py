"""Detect structural holes in the knowledge graph.

A structural hole exists where two notes (or clusters of notes) are
semantically related but not connected in the explicit link graph.
Bridging these holes would increase the coherence of the knowledge structure.

This draws on Burt's structural holes theory and Evans et al.'s work on
how bridging across knowledge clusters enables novel combinations.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from homology.vault import Note


@dataclass
class StructuralHole:
    """A detected gap between two notes worth bridging."""

    note_a: str  # slug
    note_b: str  # slug
    semantic_similarity: float  # how related the content is
    graph_distance: int | None  # shortest path in link graph (None = disconnected)
    bridge_score: float  # composite score: higher = more valuable to bridge

    @property
    def title_a(self) -> str:
        return self._titles.get(self.note_a, self.note_a)

    @property
    def title_b(self) -> str:
        return self._titles.get(self.note_b, self.note_b)

    _titles: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._titles is None:
            object.__setattr__(self, "_titles", {})


@dataclass
class ClusterGap:
    """A gap between two communities of notes."""

    cluster_a: list[str]  # slugs
    cluster_b: list[str]  # slugs
    mean_cross_similarity: float  # avg semantic similarity between clusters
    cross_link_count: int  # number of explicit links between clusters
    bridge_score: float
    best_pair: tuple[str, str]  # most similar cross-cluster pair


def find_structural_holes(
    notes: list[Note],
    embeddings: np.ndarray,
    link_graph: nx.Graph,
    top_k: int = 20,
    min_similarity: float = 0.3,
) -> list[StructuralHole]:
    """Find note pairs that are semantically close but topologically distant.

    The bridge_score captures the value of connecting two notes:
        bridge_score = semantic_similarity * distance_factor

    where distance_factor rewards pairs that are far apart or disconnected
    in the explicit link graph.

    Args:
        notes: Parsed vault notes.
        embeddings: (N, D) embedding matrix aligned with notes.
        link_graph: Graph built from wikilinks.
        top_k: Number of top holes to return.
        min_similarity: Minimum cosine similarity to consider a pair.

    Returns:
        Sorted list of structural holes, highest bridge_score first.
    """
    if len(notes) < 2:
        return []

    slug_to_title = {n.slug: n.title for n in notes}
    slugs = [n.slug for n in notes]
    n = len(notes)

    sim_matrix = cosine_similarity(embeddings)

    # Precompute shortest path lengths for all pairs in the link graph
    # For disconnected pairs, we use None
    path_lengths: dict[tuple[str, str], int | None] = {}
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = nx.shortest_path_length(link_graph, slugs[i], slugs[j])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = None
            path_lengths[(slugs[i], slugs[j])] = d

    holes: list[StructuralHole] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim < min_similarity:
                continue

            dist = path_lengths.get((slugs[i], slugs[j]))

            # Already directly linked — not a hole
            if dist is not None and dist <= 1:
                continue

            # Bridge score: similarity weighted by topological distance
            if dist is None:
                # Disconnected components — maximum distance factor
                distance_factor = 3.0
            else:
                # Connected but distant — log scale reward
                distance_factor = min(np.log2(dist + 1), 3.0)

            bridge_score = sim * distance_factor

            hole = StructuralHole(
                note_a=slugs[i],
                note_b=slugs[j],
                semantic_similarity=sim,
                graph_distance=dist,
                bridge_score=bridge_score,
            )
            object.__setattr__(hole, "_titles", slug_to_title)
            holes.append(hole)

    holes.sort(key=lambda h: h.bridge_score, reverse=True)
    return holes[:top_k]


def find_cluster_gaps(
    notes: list[Note],
    embeddings: np.ndarray,
    link_graph: nx.Graph,
    top_k: int = 10,
    resolution: float = 1.0,
) -> list[ClusterGap]:
    """Detect gaps between communities in the link graph.

    Uses Louvain community detection on the link graph, then measures
    semantic similarity between communities. High similarity + few links
    = a cluster-level structural hole.

    Args:
        notes: Parsed vault notes.
        embeddings: (N, D) embedding matrix aligned with notes.
        link_graph: Graph built from wikilinks.
        top_k: Number of top cluster gaps to return.
        resolution: Louvain resolution parameter (higher = smaller clusters).
    """
    slugs = [n.slug for n in notes]
    slug_to_idx = {s: i for i, s in enumerate(slugs)}

    # Community detection
    communities = nx.community.louvain_communities(
        link_graph, resolution=resolution, seed=42
    )

    # Filter to communities that actually contain notes we have embeddings for
    communities = [
        [s for s in comm if s in slug_to_idx] for comm in communities
    ]
    communities = [c for c in communities if len(c) >= 2]

    if len(communities) < 2:
        return []

    sim_matrix = cosine_similarity(embeddings)

    gaps: list[ClusterGap] = []
    for ci in range(len(communities)):
        for cj in range(ci + 1, len(communities)):
            ca, cb = communities[ci], communities[cj]
            idx_a = [slug_to_idx[s] for s in ca]
            idx_b = [slug_to_idx[s] for s in cb]

            # Mean cross-cluster similarity
            cross_sims = sim_matrix[np.ix_(idx_a, idx_b)]
            mean_sim = float(cross_sims.mean())

            # Count cross-cluster links
            cross_links = sum(
                1 for a in ca for b in cb if link_graph.has_edge(a, b)
            )

            # Best pair
            best_flat = int(cross_sims.argmax())
            best_i, best_j = divmod(best_flat, len(idx_b))
            best_pair = (ca[best_i], cb[best_j])

            # Score: high similarity + few links = big gap worth bridging
            link_density = cross_links / (len(ca) * len(cb))
            bridge_score = mean_sim * (1.0 - link_density)

            gaps.append(
                ClusterGap(
                    cluster_a=list(ca),
                    cluster_b=list(cb),
                    mean_cross_similarity=mean_sim,
                    cross_link_count=cross_links,
                    bridge_score=bridge_score,
                    best_pair=best_pair,
                )
            )

    gaps.sort(key=lambda g: g.bridge_score, reverse=True)
    return gaps[:top_k]
