"""Knowledge Base parser and chunking engine.

Parses Markdown documents with YAML frontmatter, extracts structured metadata,
and chunks documents by semantic heading boundaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DocumentMetadata:
    document_id: str
    title: str
    status: str  # e.g., 'active', 'superseded', 'draft'
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    superseded_date: Optional[str] = None
    audience: str = "customer"  # 'customer', 'internal'
    policy_authority: str = "official"  # 'official', 'none'
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True
    filename: str = ""
    raw_frontmatter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    document_id: str
    document_title: str
    heading: str
    content: str
    full_text: str
    status: str
    policy_authority: str
    customer_answering: bool
    audience: str
    effective_date: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active_official(self) -> bool:
        return (
            self.status == "active"
            and self.policy_authority == "official"
            and self.customer_answering
        )

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded" or bool(self.superseded_by)

    @property
    def source_tag(self) -> str:
        """Returns standard citation format: 'filename (Heading)'."""
        if self.heading and self.heading != self.document_title:
            return f"{self.filename} ({self.heading})"
        return self.filename


class KnowledgeBase:
    """Manages parsing, chunking, and metadata extraction for knowledge base documents."""

    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks: List[Chunk] = []
        self._load_and_chunk()

    def _load_and_chunk(self) -> None:
        """Loads all markdown files and chunks them hierarchically by headings."""
        md_files = sorted(self.kb_dir.glob("*.md"))
        for file_path in md_files:
            self._process_file(file_path)

    def _parse_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extracts YAML frontmatter and remaining markdown body."""
        frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = frontmatter_pattern.match(text)
        if match:
            fm_yaml = match.group(1)
            body = text[match.end():]
            try:
                fm_data = yaml.safe_load(fm_yaml) or {}
            except Exception:
                fm_data = {}
            return fm_data, body
        return {}, text

    def _process_file(self, file_path: Path) -> None:
        filename = file_path.name
        raw_text = file_path.read_text(encoding="utf-8")
        fm_data, body = self._parse_frontmatter(raw_text)

        doc_id = str(fm_data.get("document_id", file_path.stem))
        title = str(fm_data.get("title", file_path.stem))
        status = str(fm_data.get("status", "active")).lower()
        effective_date = str(fm_data.get("effective_date", "")) if fm_data.get("effective_date") else None
        last_reviewed = str(fm_data.get("last_reviewed", "")) if fm_data.get("last_reviewed") else None
        superseded_date = str(fm_data.get("superseded_date", "")) if fm_data.get("superseded_date") else None
        audience = str(fm_data.get("audience", "customer")).lower()
        policy_authority = str(fm_data.get("policy_authority", "official")).lower()
        supersedes = str(fm_data.get("supersedes", "")) if fm_data.get("supersedes") else None
        superseded_by = str(fm_data.get("superseded_by", "")) if fm_data.get("superseded_by") else None

        # Determine if doc should be used for customer answering
        customer_answering = fm_data.get("customer_answering")
        if customer_answering is None:
            customer_answering = (policy_authority == "official" and status == "active")
        else:
            customer_answering = bool(customer_answering)

        doc_meta = DocumentMetadata(
            document_id=doc_id,
            title=title,
            status=status,
            effective_date=effective_date,
            last_reviewed=last_reviewed,
            superseded_date=superseded_date,
            audience=audience,
            policy_authority=policy_authority,
            supersedes=supersedes,
            superseded_by=superseded_by,
            customer_answering=customer_answering,
            filename=filename,
            raw_frontmatter=fm_data,
        )
        self.documents[filename] = doc_meta

        # Chunk the markdown body by heading sections
        file_chunks = self._chunk_markdown(body, doc_meta)
        self.chunks.extend(file_chunks)

    def _chunk_markdown(self, body: str, meta: DocumentMetadata) -> List[Chunk]:
        """Splits markdown content into chunks based on markdown headings."""
        lines = body.splitlines()
        chunks: List[Chunk] = []

        current_heading = meta.title
        current_lines: List[str] = []
        chunk_idx = 0

        for line in lines:
            # Check for markdown headings (# Heading, ## Subheading, ### Subsubheading)
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
            if heading_match:
                # Flush previous section if it has non-empty content
                content_text = "\n".join(current_lines).strip()
                if content_text:
                    full_text = f"## {current_heading}\n{content_text}" if current_heading else content_text
                    chunks.append(
                        Chunk(
                            chunk_id=f"{meta.document_id}_{chunk_idx}",
                            filename=meta.filename,
                            document_id=meta.document_id,
                            document_title=meta.title,
                            heading=current_heading,
                            content=content_text,
                            full_text=full_text,
                            status=meta.status,
                            policy_authority=meta.policy_authority,
                            customer_answering=meta.customer_answering,
                            audience=meta.audience,
                            effective_date=meta.effective_date,
                            supersedes=meta.supersedes,
                            superseded_by=meta.superseded_by,
                            metadata=meta.raw_frontmatter,
                        )
                    )
                    chunk_idx += 1
                    current_lines = []

                current_heading = heading_match.group(2).strip()
            else:
                current_lines.append(line)

        # Flush final section
        content_text = "\n".join(current_lines).strip()
        if content_text:
            full_text = f"## {current_heading}\n{content_text}" if current_heading else content_text
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.document_id}_{chunk_idx}",
                    filename=meta.filename,
                    document_id=meta.document_id,
                    document_title=meta.title,
                    heading=current_heading,
                    content=content_text,
                    full_text=full_text,
                    status=meta.status,
                    policy_authority=meta.policy_authority,
                    customer_answering=meta.customer_answering,
                    audience=meta.audience,
                    effective_date=meta.effective_date,
                    supersedes=meta.supersedes,
                    superseded_by=meta.superseded_by,
                    metadata=meta.raw_frontmatter,
                )
            )

        return chunks

    def get_document_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        return self.documents.get(filename)
