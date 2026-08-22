"""Tests for Metadata-Aware RAG Retriever with document precedence and conflict detection."""

import pytest
from src.knowledge_base import KnowledgeBase
from src.retriever import BM25Retriever


@pytest.fixture
def retriever():
    kb = KnowledgeBase("knowledge-base")
    return BM25Retriever(kb)


def test_active_policy_beats_superseded_policy(retriever):
    """Verifies that active returns policy (30 days) ranks above superseded policy (45 days)."""
    res = retriever.retrieve("How long do I have to return an unused item?", top_k=3)
    assert len(res.chunks) > 0
    top_chunk = res.chunks[0]
    assert top_chunk.filename == "01-returns-policy-current.md"
    assert top_chunk.is_authoritative is True
    assert not any(c.filename == "02-returns-policy-legacy.md" and c.score >= top_chunk.score for c in res.chunks)


def test_draft_migration_notes_excluded(retriever):
    """Verifies that unapproved migration notes are excluded from customer answers."""
    res = retriever.retrieve("Do I get 60 days to return everything?", top_k=4)
    sources = [c.filename for c in res.chunks]
    assert "14-internal-content-migration-notes.md" not in sources


def test_breeze_tumbler_source_conflict(retriever):
    """Verifies that conflict is detected between product care and product card on dishwasher safety."""
    res = retriever.retrieve("Can I put the Breeze Tumbler in the dishwasher?", top_k=4)
    assert res.conflict_detected is True
    assert "11-product-care.md" in res.conflicting_sources
    assert "12-breeze-tumbler-product-card.md" in res.conflicting_sources


def test_insufficient_evidence_threshold(retriever):
    """Verifies low retrieval confidence on queries completely absent from knowledge base."""
    res = retriever.retrieve("Are all adhesives and bag linings certified organic vegan?", top_k=4)
    # KB does not mention vegan adhesives, so max score should be low
    assert not res.has_sufficient_evidence or (res.chunks and res.chunks[0].score < 1.5)
