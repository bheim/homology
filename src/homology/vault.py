"""Parse an Obsidian vault: extract note content, metadata, and wikilinks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Matches [[target]] and [[target|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Matches YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


@dataclass
class Note:
    """A single Obsidian markdown note."""

    path: Path
    title: str
    content: str  # body text with frontmatter stripped
    frontmatter: dict = field(default_factory=dict)
    outgoing_links: list[str] = field(default_factory=list)  # wikilink targets

    @property
    def slug(self) -> str:
        """Canonical identifier: filename without extension, lowercased."""
        return self.path.stem.lower()


def parse_note(path: Path) -> Note:
    """Parse a single markdown file into a Note."""
    raw = path.read_text(encoding="utf-8")

    # Extract frontmatter
    frontmatter: dict = {}
    body = raw
    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            frontmatter = {}
        body = raw[fm_match.end() :]

    # Extract wikilinks
    outgoing = _WIKILINK_RE.findall(body)
    # Normalize: strip heading/block refs (e.g. "Note#heading" -> "Note")
    outgoing = [link.split("#")[0].strip() for link in outgoing]
    outgoing = [link for link in outgoing if link]  # drop empty

    title = frontmatter.get("title", path.stem)

    return Note(
        path=path,
        title=title,
        content=body.strip(),
        frontmatter=frontmatter,
        outgoing_links=outgoing,
    )


def load_vault(vault_path: str | Path) -> list[Note]:
    """Recursively load all markdown notes from an Obsidian vault directory.

    Skips hidden directories (e.g. .obsidian, .trash).
    """
    vault = Path(vault_path)
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {vault}")

    notes: list[Note] = []
    for md_file in sorted(vault.rglob("*.md")):
        # Skip hidden dirs
        if any(part.startswith(".") for part in md_file.relative_to(vault).parts):
            continue
        notes.append(parse_note(md_file))

    return notes
