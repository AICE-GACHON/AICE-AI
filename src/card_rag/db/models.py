"""추천 도메인 테이블. 문서 `Data 모델링` ERD 중 AI가 소유하는 부분만 우선 정의.

- 사용자/인증/CODEF 등 BE 소유 테이블은 여기 두지 않고, 추천 요청 시 입력 DTO로 받는다.
- 전월실적 구간은 `benefit_clauses`의 개별 row로 분리(min_spend로 구분) — 팀 결정.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from card_rag.config import settings
from card_rag.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Card(Base):
    __tablename__ = "cards"

    card_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug (예: shinhan-deep-dream)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    issuer: Mapped[str] = mapped_column(String(50), nullable=False)
    annual_fee: Mapped[int] = mapped_column(Integer, default=0)
    highlight: Mapped[Optional[str]] = mapped_column(String(200))

    clauses: Mapped[list["BenefitClause"]] = relationship(back_populates="card", cascade="all, delete-orphan")


class BenefitClause(Base):
    """혜택절 = RAG 검색 단위(1혜택=1청크). 정형 숫자(규칙 엔진용) + 임베딩(포함/제외 매칭용)."""

    __tablename__ = "benefit_clauses"

    benefit_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.card_id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(30), index=True)          # 내부 업종코드
    benefit_type: Mapped[str] = mapped_column(String(20))                  # 적립 | 청구할인

    # 정형 숫자 — 규칙 엔진이 그대로 사용(사람 검수 필수)
    rate: Mapped[float] = mapped_column(Numeric(5, 2))                     # % (적립률/할인율)
    monthly_cap: Mapped[Optional[int]] = mapped_column(Integer)           # 월 한도(원), 없으면 NULL
    min_spend: Mapped[int] = mapped_column(Integer, default=0)            # 전월실적 요건(원)

    # 비정형 자연어 — 임베딩 대상(포함/제외 조건 중심)
    include_notes: Mapped[Optional[str]] = mapped_column(Text)
    exclude_notes: Mapped[Optional[str]] = mapped_column(Text)

    # 검수·추적용
    source_span: Mapped[Optional[str]] = mapped_column(Text)              # 추출 근거 원문
    embedding_text: Mapped[Optional[str]] = mapped_column(Text)           # 실제 임베딩한 문자열(투명성)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(settings.embed_dim))

    card: Mapped["Card"] = relationship(back_populates="clauses")


class CategoryMapping(Base):
    """Kakao 카테고리코드 → 내부 업종코드 매핑(코드 우선, 실패 시 임베딩 폴백)."""

    __tablename__ = "category_mappings"

    kakao_category_code: Mapped[str] = mapped_column(String(20), primary_key=True)
    internal_category_code: Mapped[str] = mapped_column(String(30), nullable=False)
    mapping_type: Mapped[str] = mapped_column(String(20))  # code | embedding_fallback


class Merchant(Base):
    __tablename__ = "merchants"

    merchant_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    latitude: Mapped[float] = mapped_column(Numeric(10, 7))
    longitude: Mapped[float] = mapped_column(Numeric(10, 7))
    kakao_category_code: Mapped[Optional[str]] = mapped_column(String(20))


class Recommendation(Base):
    """추천 결과 스냅샷(설명출처: precomputed / llm_realtime / rule_only)."""

    __tablename__ = "recommendations"

    recommendation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    merchant_id: Mapped[str] = mapped_column(String(36))
    card_id: Mapped[str] = mapped_column(String(64))
    expected_benefit_won: Mapped[int] = mapped_column(Integer)
    eligible: Mapped[bool] = mapped_column(default=True)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    explanation_source: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "merchant_id", "card_id", name="uq_reco_user_merchant_card"),)
