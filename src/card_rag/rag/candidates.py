"""후보 수집 — DB의 benefit_clauses → 규칙 엔진 입력(ClauseCalc) + 판정용 메타.

(보유카드 × 업종)으로 혜택절을 가져와 규칙 엔진/판정이 쓸 형태로 변환한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select

from card_rag.db.base import SessionLocal
from card_rag.db.models import BenefitClause
from card_rag.rules.types import ClauseCalc


@dataclass
class CandidateClause:
    calc: ClauseCalc
    include_text: Optional[str]
    exclude_text: Optional[str]

    @property
    def ambiguous(self) -> bool:
        """포함(화이트리스트)/제외(블랙리스트) 조건이 있으면 자격이 애매 → 판정 필요."""
        return bool(self.include_text or self.exclude_text)


def fetch_candidates(card_ids: list[str], categories: list[str]) -> list[CandidateClause]:
    with SessionLocal() as session:
        rows = (
            session.execute(
                select(BenefitClause).where(
                    BenefitClause.card_id.in_(card_ids),
                    BenefitClause.category.in_(categories),
                )
            )
            .scalars()
            .all()
        )
    out: list[CandidateClause] = []
    for bc in rows:
        calc = ClauseCalc(
            card_id=bc.card_id,
            clause_id=bc.benefit_id,
            category=bc.category,
            benefit_type=bc.benefit_type,
            value_type=bc.value_type,
            rate=float(bc.rate or 0),
            flat_amount=bc.flat_amount or 0,
            flat_min_txn=bc.flat_min_txn or 0,
            min_spend=bc.min_spend or 0,
            remaining_cap=bc.monthly_cap,   # MVP: 남은 한도 = 월 한도(전액 잔여 가정)
            value_factor=1.0,               # 캐시백/할인=1.0
        )
        out.append(CandidateClause(calc, bc.include_text, bc.exclude_text))
    return out
