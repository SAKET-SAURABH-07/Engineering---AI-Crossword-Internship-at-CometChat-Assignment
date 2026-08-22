"""Tests for Knowledge Base parser, frontmatter metadata extraction, and chunking."""

from pathlib import Path
import pytest
from src.knowledge_base import KnowledgeBase, DocumentMetadata, Chunk


@pytest.fixture
def kb():
    return KnowledgeBase("knowledge-base")


def test_kb_loads_all_documents(kb):
    assert len(kb.documents) == 14
    assert len(kb.chunks) > 20


def test_frontmatter_metadata_extraction(kb):
    current_returns = kb.get_document_by_filename("01-returns-policy-current.md")
    assert current_returns is not None
    assert current_returns.document_id == "RET-2026-01"
    assert current_returns.status == "active"
    assert current_returns.policy_authority == "official"
    assert current_returns.supersedes == "RET-2024-01"
    assert current_returns.customer_answering is True


def test_superseded_metadata(kb):
    legacy_returns = kb.get_document_by_filename("02-returns-policy-legacy.md")
    assert legacy_returns is not None
    assert legacy_returns.status == "superseded"
    assert legacy_returns.superseded_by == "RET-2026-01"


def test_draft_internal_metadata(kb):
    migration_notes = kb.get_document_by_filename("14-internal-content-migration-notes.md")
    assert migration_notes is not None
    assert migration_notes.status == "draft"
    assert migration_notes.audience == "internal"
    assert migration_notes.policy_authority == "none"
    assert migration_notes.customer_answering is False


def test_heading_chunking(kb):
    # Verify chunks for returns policy have specific headings
    ret_chunks = [c for c in kb.chunks if c.filename == "01-returns-policy-current.md"]
    headings = {c.heading for c in ret_chunks}
    assert "Standard return window" in headings
    assert "Item condition" in headings
    assert "Return shipping and refunds" in headings
    assert "Exclusions and exceptions" in headings
