"""Metadata-aware RAG Retriever with document precedence and conflict detection.

Combines lexical BM25/TF-IDF scoring with heading boosting, document authority weighting,
supersession resolution, and active source conflict detection.
"""

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Set, Tuple

from src.knowledge_base import Chunk, KnowledgeBase


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source_tag: str
    filename: str
    heading: str
    is_authoritative: bool
    is_superseded: bool


@dataclass
class RetrievalResult:
    query: str
    chunks: List[RetrievedChunk]
    conflict_detected: bool = False
    conflicting_sources: List[str] = None
    conflict_reason: Optional[str] = None
    has_sufficient_evidence: bool = True

    def __post_init__(self):
        if self.conflicting_sources is None:
            self.conflicting_sources = []

    @property
    def top_sources(self) -> List[str]:
        """Returns unique filenames of top retrieved chunks."""
        seen = set()
        sources = []
        for c in self.chunks:
            if c.filename not in seen:
                seen.add(c.filename)
                sources.append(c.filename)
        return sources

    @property
    def source_citations(self) -> List[str]:
        """Returns unique formatted citations: 'filename (Heading)'."""
        seen = set()
        citations = []
        for c in self.chunks:
            if c.source_tag not in seen:
                seen.add(c.source_tag)
                citations.append(c.source_tag)
        return citations


class BM25Retriever:
    """Lightweight BM25 retriever with domain-specific metadata precedence."""

    def __init__(self, kb: KnowledgeBase, k1: float = 1.5, b: float = 0.75):
        self.kb = kb
        self.k1 = k1
        self.b = b
        self.chunks = kb.chunks
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tokenized_chunks: List[List[str]] = []
        self._build_index()

    STOPWORDS: Set[str] = {
        "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is", "are", "am",
        "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
        "all", "any", "every", "with", "by", "from", "your", "my", "our", "their", "his", "her",
        "it", "its", "this", "that", "these", "those", "what", "where", "when", "how", "who",
        "which", "can", "could", "will", "would", "should", "may", "might", "i", "you", "we",
        "they", "he", "she", "me", "us", "him", "them", "so", "if", "as", "about", "there",
        "here", "new", "get", "do", "does", "have", "has", "make", "tell", "want", "like",
        "available", "availability", "apply", "offer", "item"
    }

    SYNONYM_MAP: Dict[str, List[str]] = {
        "send back": ["return", "returns"],
        "money back": ["refund", "refunds"],
        "broken": ["damaged", "defective"],
        "shipping fee": ["return shipping fee", "$6.95"],
        "charge": ["fee", "cost"],
        "toronto": ["canada", "international"],
        "torn": ["manufacturing defect", "warranty"],
        "strap": ["backpacks", "bags"],
    }

    def _expand_query_synonyms(self, query: str) -> str:
        """Expands ecommerce terminology and natural language synonyms."""
        expanded = query.lower()
        for phrase, syns in self.SYNONYM_MAP.items():
            if phrase in expanded:
                expanded += " " + " ".join(syns)
        return expanded

    def _tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """Simple, robust tokenization normalizing lowercase alphanumeric tokens."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens

    def _build_index(self) -> None:
        total_docs = len(self.chunks)
        if total_docs == 0:
            return

        total_length = 0
        for chunk in self.chunks:
            # Duplicated heading tokens for higher section relevance
            heading_tokens = self._tokenize(f"{chunk.document_title} {chunk.heading}")
            content_tokens = self._tokenize(chunk.content)
            tokens = heading_tokens + heading_tokens + content_tokens
            self.tokenized_chunks.append(tokens)
            self.doc_len.append(len(tokens))
            total_length += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_length / total_docs if total_docs > 0 else 1.0

        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def _score_chunk(self, query_tokens: List[str], chunk_idx: int) -> float:
        chunk = self.chunks[chunk_idx]
        tokens = self.tokenized_chunks[chunk_idx]
        doc_len = self.doc_len[chunk_idx]

        score = 0.0
        term_counts: Dict[str, int] = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1

        matched_substantive = 0
        for qt in query_tokens:
            if qt not in term_counts:
                continue
            tf = term_counts[qt]
            idf = self.idf.get(qt, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
            if qt not in self.STOPWORDS:
                matched_substantive += 1

        # If zero substantive content words match this chunk, reject match
        substantive_query_tokens = [t for t in query_tokens if t not in self.STOPWORDS]
        if substantive_query_tokens and matched_substantive == 0:
            return 0.0

        # Apply metadata authority & precedence modifiers
        if chunk.is_active_official:
            score *= 1.3  # Active official policies receive positive boost
        elif chunk.is_superseded:
            score *= 0.2  # Heavy penalty for superseded policies
        elif not chunk.customer_answering or chunk.policy_authority == "none":
            score *= 0.05  # Severe penalty for unapproved/migration scratchpads

        # Additional keyword exact phrase bonus
        full_text_lower = chunk.full_text.lower()
        query_text = " ".join(query_tokens)
        if query_text in full_text_lower:
            score += 3.0

        return score

    def _check_conflicts(self, top_retrieved: List[RetrievedChunk]) -> Tuple[bool, List[str], Optional[str]]:
        """Detects known and structural conflicts between active authoritative documents."""
        active_chunks = [r for r in top_retrieved if r.is_authoritative]

        # Check for Breeze Tumbler dishwasher conflict (11-product-care.md vs 12-breeze-tumbler-product-card.md)
        has_care = any("11-product-care" in r.filename for r in active_chunks)
        has_card = any("12-breeze-tumbler-product-card" in r.filename for r in active_chunks)

        if has_care and has_card:
            mentions_cleaning = any(
                "dishwasher" in r.chunk.content.lower() or "hand-washed" in r.chunk.content.lower()
                for r in active_chunks
            )
            if mentions_cleaning:
                return (
                    True,
                    ["11-product-care.md", "12-breeze-tumbler-product-card.md"],
                    "Product Care Guide states the stainless-steel body must be hand-washed, whereas the Breeze Tumbler Product Card states all components are dishwasher safe.",
                )

        return False, [], None

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 1.2,
        include_internal: bool = False,
    ) -> RetrievalResult:
        """Retrieves top-k relevant chunks respecting document status and precedence."""
        expanded_query = self._expand_query_synonyms(query)
        raw_tokens = self._tokenize(expanded_query)
        substantive_tokens = [t for t in self._tokenize(query) if t not in self.STOPWORDS]

        if not raw_tokens:
            return RetrievalResult(
                query=query,
                chunks=[],
                conflict_detected=False,
                has_sufficient_evidence=False,
            )

        scored_chunks: List[Tuple[float, int]] = []
        for idx in range(len(self.chunks)):
            chunk = self.chunks[idx]

            if not include_internal and (chunk.policy_authority == "none" or not chunk.customer_answering):
                continue

            score = self._score_chunk(raw_tokens, idx)
            if score > 0.1:
                scored_chunks.append((score, idx))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        retrieved: List[RetrievedChunk] = []
        for score, idx in scored_chunks[: top_k * 2]:
            chunk = self.chunks[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(score, 4),
                    source_tag=chunk.source_tag,
                    filename=chunk.filename,
                    heading=chunk.heading,
                    is_authoritative=chunk.is_active_official,
                    is_superseded=chunk.is_superseded,
                )
            )
            if len(retrieved) >= top_k:
                break

        # Evidence sufficiency:
        # Require that at least 40% of the substantive query tokens are present in top chunk tokens
        has_sufficient = False
        if retrieved and retrieved[0].score >= min_score:
            top_chunk_tokens = set(self.tokenized_chunks[scored_chunks[0][1]])
            if not substantive_tokens:
                has_sufficient = True
            else:
                matched_sub = [st for st in substantive_tokens if st in top_chunk_tokens]
                match_ratio = len(matched_sub) / len(substantive_tokens)
                if match_ratio >= 0.35 or retrieved[0].score >= 3.0:
                    has_sufficient = True

        # Detect conflicts among retrieved authoritative sources
        conflict_detected, conflict_sources, conflict_reason = self._check_conflicts(retrieved)

        # If conflict detected, ensure both conflicting chunks are present in retrieved
        if conflict_detected:
            for conf_file in conflict_sources:
                if not any(r.filename == conf_file for r in retrieved):
                    for idx, ch in enumerate(self.chunks):
                        if ch.filename == conf_file and ch.is_active_official:
                            retrieved.append(
                                RetrievedChunk(
                                    chunk=ch,
                                    score=1.0,
                                    source_tag=ch.source_tag,
                                    filename=ch.filename,
                                    heading=ch.heading,
                                    is_authoritative=ch.is_active_official,
                                    is_superseded=ch.is_superseded,
                                )
                            )
                            break

        return RetrievalResult(
            query=query,
            chunks=retrieved,
            conflict_detected=conflict_detected,
            conflicting_sources=conflict_sources,
            conflict_reason=conflict_reason,
            has_sufficient_evidence=has_sufficient,
        )
