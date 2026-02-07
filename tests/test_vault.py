"""Tests for vault parsing."""

from pathlib import Path
from textwrap import dedent

import pytest

from homology.vault import Note, load_vault, parse_note


@pytest.fixture
def tmp_vault(tmp_path):
    """Create a minimal Obsidian vault with a few linked notes."""
    (tmp_path / "Epistemology.md").write_text(
        dedent("""\
        ---
        title: Epistemology
        tags: [philosophy]
        ---
        # Epistemology

        The study of knowledge. Closely related to [[Philosophy of Science]].
        See also [[Bayesian Inference]] for a formal framework.
        """)
    )
    (tmp_path / "Philosophy of Science.md").write_text(
        dedent("""\
        # Philosophy of Science

        How we know what we know about the natural world.
        Links to [[Epistemology]] and [[Scientific Method]].
        """)
    )
    (tmp_path / "Bayesian Inference.md").write_text(
        dedent("""\
        # Bayesian Inference

        Updating beliefs with evidence. P(H|E) = P(E|H)P(H)/P(E).
        """)
    )
    (tmp_path / "Network Science.md").write_text(
        dedent("""\
        # Network Science

        The study of complex networks. [[Graph Theory]] is foundational.
        """)
    )
    # Hidden dir — should be skipped
    hidden = tmp_path / ".obsidian"
    hidden.mkdir()
    (hidden / "config.md").write_text("should be skipped")

    return tmp_path


def test_parse_note_frontmatter(tmp_vault):
    note = parse_note(tmp_vault / "Epistemology.md")
    assert note.title == "Epistemology"
    assert note.frontmatter["tags"] == ["philosophy"]
    assert "---" not in note.content


def test_parse_note_wikilinks(tmp_vault):
    note = parse_note(tmp_vault / "Epistemology.md")
    assert "Philosophy of Science" in note.outgoing_links
    assert "Bayesian Inference" in note.outgoing_links


def test_parse_note_no_frontmatter(tmp_vault):
    note = parse_note(tmp_vault / "Network Science.md")
    assert note.title == "Network Science"  # falls back to filename stem
    assert note.frontmatter == {}
    assert "Graph Theory" in note.outgoing_links


def test_load_vault_skips_hidden(tmp_vault):
    notes = load_vault(tmp_vault)
    slugs = {n.slug for n in notes}
    assert "config" not in slugs
    assert len(notes) == 4


def test_load_vault_not_found():
    with pytest.raises(FileNotFoundError):
        load_vault("/nonexistent/vault")


def test_note_slug(tmp_vault):
    note = parse_note(tmp_vault / "Philosophy of Science.md")
    assert note.slug == "philosophy of science"


def test_wikilink_with_alias(tmp_path):
    (tmp_path / "test.md").write_text("See [[Target Note|my alias]] for more.")
    note = parse_note(tmp_path / "test.md")
    assert "Target Note" in note.outgoing_links


def test_wikilink_with_heading(tmp_path):
    (tmp_path / "test.md").write_text("See [[Target Note#Section]] for details.")
    note = parse_note(tmp_path / "test.md")
    assert "Target Note" in note.outgoing_links
