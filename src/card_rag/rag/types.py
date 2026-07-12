"""추천 파이프라인 내부 DTO(엔진 전용).

FastAPI 경계용 Pydantic 스키마와 분리해, LLM/DB 의존 없이 순수 로직으로 테스트 가능하게
표준 라이브러리 dataclass로 정의한다. (triage는 데이터/모델 없이 검증할 수 있어야 함)
"""
from __future__ import annotations

from dataclasses import dataclass

RULE_ONLY = "rule_only"     # RAG 불필요 — 숫자 + 사전생성/템플릿 설명
RAG_JUDGE = "rag_judge"     # 조건 애매 + 순위 좌우 → 검색 + LLM 자격 판정


@dataclass(frozen=True)
class MerchantCtx:
    merchant_id: str
    name: str
    category: str                      # 내부 업종코드
    category_uncertain: bool = False   # 업종 매핑이 임베딩 폴백이면 True


@dataclass(frozen=True)
class RuleCandidate:
    """규칙 엔진이 (카드×업종)으로 계산한 후보 1건. 카드당 1개(최적 혜택절)를 가정."""

    card_id: str
    clause_id: str
    category: str
    benefit_type: str          # 적립 | 청구할인
    included_won: int          # 이 혜택절이 '적용될 때'의 기대혜택(원)
    excluded_won: int          # 이 혜택절이 '제외될 때' 이 카드의 대체 기대혜택(원)
    has_condition: bool        # include/exclude 조건 존재 여부

    def realized(self, included: bool) -> int:
        return self.included_won if included else self.excluded_won


@dataclass(frozen=True)
class TriageDecision:
    card_id: str
    clause_id: str
    path: str                  # RULE_ONLY | RAG_JUDGE
    reason: str


@dataclass(frozen=True)
class Judgment:
    """혜택절 자격 판정 결과. 숫자는 없고 자격 여부와 근거만(가드레일)."""

    card_id: str
    clause_id: str
    eligible: bool
    confidence: str            # high | medium | low
    reason: str
    source: str                # rule_only | rule_signal | llm | fallback
