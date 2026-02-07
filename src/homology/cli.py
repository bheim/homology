"""CLI entrypoint for homology."""

from __future__ import annotations

from pathlib import Path

import click
import networkx as nx
from rich.console import Console
from rich.table import Table

from homology.embeddings import get_embedder
from homology.graph import build_link_graph
from homology.holes import find_cluster_gaps, find_structural_holes
from homology.vault import load_vault

console = Console()


@click.group()
def main():
    """homology — find structural holes in your Obsidian knowledge graph."""


@main.command()
@click.argument("vault", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--backend",
    type=click.Choice(["openai", "local"]),
    default="openai",
    help="Embedding backend.",
)
@click.option("--top-k", default=20, help="Number of top holes to show.")
@click.option(
    "--min-similarity",
    default=0.3,
    help="Minimum cosine similarity to consider a pair.",
)
@click.option("--model", default=None, help="Embedding model name override.")
def holes(
    vault: str,
    backend: str,
    top_k: int,
    min_similarity: float,
    model: str | None,
):
    """Find structural holes — semantically related but unlinked note pairs."""
    vault_path = Path(vault)

    with console.status("Loading vault..."):
        notes = load_vault(vault_path)
    console.print(f"Loaded [bold]{len(notes)}[/bold] notes from {vault_path}")

    if len(notes) < 2:
        console.print("[yellow]Need at least 2 notes to find holes.[/yellow]")
        return

    embedder_kwargs = {}
    if model:
        embedder_kwargs["model" if backend == "openai" else "model_name"] = model

    with console.status(f"Generating embeddings ({backend})..."):
        embedder = get_embedder(backend, **embedder_kwargs)
        embeddings = embedder.embed_notes(notes)
    console.print(f"Embeddings: {embeddings.shape}")

    with console.status("Building link graph..."):
        link_graph = build_link_graph(notes)
    console.print(
        f"Link graph: {link_graph.number_of_nodes()} nodes, "
        f"{link_graph.number_of_edges()} edges"
    )

    with console.status("Detecting structural holes..."):
        results = find_structural_holes(
            notes, embeddings, link_graph, top_k=top_k, min_similarity=min_similarity
        )

    if not results:
        console.print("[green]No structural holes found — your vault is well-connected![/green]")
        return

    table = Table(title="Structural Holes", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Note A", style="cyan")
    table.add_column("Note B", style="cyan")
    table.add_column("Similarity", justify="right")
    table.add_column("Link Distance", justify="right")
    table.add_column("Bridge Score", justify="right", style="bold green")

    for i, hole in enumerate(results, 1):
        dist_str = str(hole.graph_distance) if hole.graph_distance is not None else "∞"
        table.add_row(
            str(i),
            hole.title_a,
            hole.title_b,
            f"{hole.semantic_similarity:.3f}",
            dist_str,
            f"{hole.bridge_score:.3f}",
        )

    console.print(table)


@main.command()
@click.argument("vault", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--backend",
    type=click.Choice(["openai", "local"]),
    default="openai",
    help="Embedding backend.",
)
@click.option("--top-k", default=10, help="Number of top cluster gaps to show.")
@click.option("--resolution", default=1.0, help="Louvain community resolution.")
@click.option("--model", default=None, help="Embedding model name override.")
def clusters(
    vault: str,
    backend: str,
    top_k: int,
    resolution: float,
    model: str | None,
):
    """Find gaps between note communities — cluster-level structural holes."""
    vault_path = Path(vault)

    with console.status("Loading vault..."):
        notes = load_vault(vault_path)
    console.print(f"Loaded [bold]{len(notes)}[/bold] notes from {vault_path}")

    if len(notes) < 4:
        console.print("[yellow]Need at least 4 notes for cluster analysis.[/yellow]")
        return

    embedder_kwargs = {}
    if model:
        embedder_kwargs["model" if backend == "openai" else "model_name"] = model

    with console.status(f"Generating embeddings ({backend})..."):
        embedder = get_embedder(backend, **embedder_kwargs)
        embeddings = embedder.embed_notes(notes)

    with console.status("Building link graph..."):
        link_graph = build_link_graph(notes)

    with console.status("Detecting cluster gaps..."):
        gaps = find_cluster_gaps(
            notes, embeddings, link_graph, top_k=top_k, resolution=resolution
        )

    if not gaps:
        console.print("[green]No cluster gaps found.[/green]")
        return

    slug_to_title = {n.slug: n.title for n in notes}

    for i, gap in enumerate(gaps, 1):
        console.print(f"\n[bold]Gap #{i}[/bold]  (bridge score: {gap.bridge_score:.3f})")
        console.print(f"  Cross-similarity: {gap.mean_cross_similarity:.3f}")
        console.print(f"  Cross-links: {gap.cross_link_count}")

        a_titles = [slug_to_title.get(s, s) for s in gap.cluster_a[:5]]
        b_titles = [slug_to_title.get(s, s) for s in gap.cluster_b[:5]]
        suffix_a = f" (+{len(gap.cluster_a) - 5} more)" if len(gap.cluster_a) > 5 else ""
        suffix_b = f" (+{len(gap.cluster_b) - 5} more)" if len(gap.cluster_b) > 5 else ""

        console.print(f"  Cluster A: {', '.join(a_titles)}{suffix_a}")
        console.print(f"  Cluster B: {', '.join(b_titles)}{suffix_b}")

        best_a, best_b = gap.best_pair
        console.print(
            f"  Best bridge: [cyan]{slug_to_title.get(best_a, best_a)}[/cyan] ↔ "
            f"[cyan]{slug_to_title.get(best_b, best_b)}[/cyan]"
        )


@main.command()
@click.argument("vault", type=click.Path(exists=True, file_okay=False))
def stats(vault: str):
    """Print basic statistics about the vault's link structure."""
    vault_path = Path(vault)

    notes = load_vault(vault_path)
    link_graph = build_link_graph(notes)

    console.print(f"Notes: {len(notes)}")
    console.print(f"Explicit links: {link_graph.number_of_edges()}")

    components = list(nx.connected_components(link_graph))
    console.print(f"Connected components: {len(components)}")

    isolates = list(nx.isolates(link_graph))
    console.print(f"Isolated notes (no links): {len(isolates)}")

    if link_graph.number_of_edges() > 0:
        density = nx.density(link_graph)
        console.print(f"Graph density: {density:.4f}")


if __name__ == "__main__":
    main()
