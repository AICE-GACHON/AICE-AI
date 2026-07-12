"""규칙 엔진 입력/출력 DTO(엔진 전용, 표준 라이브러리 dataclass).

LLM/DB 의존 없이 순수 로직으로 테스트 가능하게 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PERCENT = "percent"
FLAT = "flat"


@dataclass(frozen=True)
class ClauseCalc:
    """규칙 엔진이 계산에 쓰는 혜택절 1건(검수·적재된 benefit_clauses에서 파생)."""

    card_id: str
    clause_id: str
    category: str
    benefit_type: str            # 적립 | 청구할인
    value_type: str              # percent | flat
    rate: float = 0.0            # % (percent)
    flat_amount: int = 0         # 원, 건당 (flat)
    flat_min_txn: int = 0        # flat 적용 최소 결제액(원)
    min_spend: int = 0           # 전월실적 요건(원)
    remaining_cap: Optional[int] = None  # 남은 월 한도(원). None=무제한
    value_factor: float = 1.0    # 적립 실질가치 계수(캐시백/할인=1.0)


@dataclass(frozen=True)
class BenefitSegment:
    """결제금액 구간 [a_from, a_to)에서 최적 카드와 그 구간 대표 기대혜택.

    a_to=None 이면 '그 이상 전부'(무한대). card_id=None 이면 혜택 없음(무적용) 구간.
    """

    a_from: int
    a_to: Optional[int]
    card_id: Optional[str]
    clause_id: Optional[str]
    benefit_at_from: int         # a_from(구간 시작)에서의 기대혜택(원, 반올림)
