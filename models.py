"""
Database Models
----------------
All tables for TruthLens. pgvector enabled for claim embeddings (RAG dedup).
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List
import uuid

from sqlalchemy import (
    String, Text, Float, Boolean, DateTime, ForeignKey,
    Enum, Integer, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────────────────

class ClaimType(str, PyEnum):
    EMPIRICAL = "empirical"       # Has a verifiable true/false answer
    CONTESTED = "contested"       # Political / opinion — show sources, no verdict


class Verdict(str, PyEnum):
    TRUE = "true"
    FALSE = "false"
    UNVERIFIED = "unverified"
    CONTESTED = "contested"       # Only used for ClaimType.CONTESTED


class PipelineStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Models ───────────────────────────────────────────────────────────────────

class Article(Base):
    """Raw ingested news article."""
    __tablename__ = "articles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    country_of_origin: Mapped[Optional[str]] = mapped_column(String(10))  # ISO 3166-1 alpha-2

    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="article")


class Claim(Base):
    """An extracted, checkable claim from an article."""
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id"), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[Optional[str]] = mapped_column(Text)  # cleaned for comparison
    claim_type: Mapped[Optional[ClaimType]] = mapped_column(Enum(ClaimType))
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768))  # Gemini embedding dim
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    article: Mapped["Article"] = relationship("Article", back_populates="claims")
    fact_checks: Mapped[List["FactCheck"]] = relationship("FactCheck", back_populates="claim")
    verdict: Mapped[Optional["ClaimVerdict"]] = relationship("ClaimVerdict", back_populates="claim", uselist=False)


class FactCheck(Base):
    """Evidence gathered from a single fact-checking source for a claim."""
    __tablename__ = "fact_checks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)   # e.g. "PolitiFact"
    source_url: Mapped[Optional[str]] = mapped_column(String(2048))
    source_verdict: Mapped[Optional[str]] = mapped_column(String(100))      # raw verdict from source
    source_weight: Mapped[float] = mapped_column(Float, default=1.0)        # dynamic weight 0-1
    summary: Mapped[Optional[str]] = mapped_column(Text)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    claim: Mapped["Claim"] = relationship("Claim", back_populates="fact_checks")


class ClaimVerdict(Base):
    """Final synthesized verdict for a claim after all evidence is ranked."""
    __tablename__ = "claim_verdicts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), unique=True, nullable=False)
    verdict: Mapped[Verdict] = mapped_column(Enum(Verdict), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 - 1.0
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    sources_used: Mapped[Optional[dict]] = mapped_column(JSON)              # {source: weight}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    claim: Mapped["Claim"] = relationship("Claim", back_populates="verdict")


class PipelineRun(Base):
    """Audit log of every pipeline execution."""
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    triggered_by: Mapped[str] = mapped_column(String(50), default="scheduler")  # scheduler | api
    status: Mapped[PipelineStatus] = mapped_column(Enum(PipelineStatus), default=PipelineStatus.PENDING)
    articles_ingested: Mapped[int] = mapped_column(Integer, default=0)
    claims_extracted: Mapped[int] = mapped_column(Integer, default=0)
    claims_verified: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error: Mapped[Optional[str]] = mapped_column(Text)


class AgentLog(Base):
    """Async structured logs from every agent — queryable, not in critical path."""
    __tablename__ = "agent_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    level: Mapped[str] = mapped_column(String(20), default="INFO")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[dict]] = mapped_column(JSON)
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
