"""추출/적재 파이프라인이 주고받는 Pydantic 계약.

LLM은 `ExtractedClause`를 출력하고, 사람이 숫자 필드를 검수해 확정한 뒤 DB로 적재한다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 내부 업종코드(프로토타입용 최소 집합). Kakao 카테고리 → 이 코드로 매핑.
InternalCategory = Literal[
    "카페", "음식점", "편의점", "대형마트", "온라인쇼핑",
    "대중교통", "주유", "통신", "영화문화", "병원약국", "해외", "기타",
]

BenefitType = Literal["적립", "청구할인"]
Confidence = Literal["high", "medium", "low"]


class ExtractedClause(BaseModel):
    """LLM이 약관 원문에서 뽑아낸 혜택절 1건(검수 전 초안)."""

    category: InternalCategory
    benefit_type: BenefitType
    rate: float = Field(description="적립률/할인율 (%). 원문에 명시된 값만.")
    monthly_cap: Optional[int] = Field(default=None, description="월 한도(원). 없으면 null.")
    min_spend: int = Field(default=0, description="전월실적 요건(원). 구간마다 별도 절로 분리.")
    include_notes: str = Field(default="", description="포함되는 가맹점/조건(원문 표현).")
    exclude_notes: str = Field(default="", description="제외되는 가맹점/조건(원문 표현).")
    source_span: str = Field(description="판단 근거가 된 약관 원문 문장(검수용).")
    confidence: Confidence = Field(default="medium", description="추출 확신도. low는 검수 우선.")


class ExtractionResult(BaseModel):
    card_id: str
    clauses: list[ExtractedClause]
