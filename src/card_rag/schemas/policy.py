"""정책 청크(policy) 추출 계약 — 혜택절 하나에 안 담기는 카드 단위 규칙.

약관 심화 판정(A)을 위해 전역 제외·통합한도·특약을 별도로 뽑는다.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from card_rag.schemas.clause import Confidence, InternalCategory

PolicyType = str  # global_exclude | aggregate_cap | special_term | performance_exclude


class ExtractedPolicy(BaseModel):
    """혜택률에 안 담기는 카드 단위 규칙 1건."""

    policy_type: str = Field(
        description="global_exclude(혜택 제외) | performance_exclude(실적 제외) | "
        "aggregate_cap(통합한도·횟수) | special_term(특약: 결제형태·중복 등)"
    )
    category: Optional[InternalCategory] = Field(
        default=None, description="특정 업종에만 적용되면 업종코드, 카드 전체면 null."
    )
    text: str = Field(description="규칙 내용(원문 표현 유지, 간결히).")
    source_span: str = Field(description="근거가 된 약관 원문 문장.")
    confidence: Confidence = Field(default="medium")


class PolicyExtractionResult(BaseModel):
    card_id: str
    policies: list[ExtractedPolicy]
