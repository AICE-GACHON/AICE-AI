"""규칙 엔진 스켈레톤.

입력: 보유카드 + 이번달 실적(입력 DTO, BE로부터) + 가맹점 업종
처리: (업종 × 보유카드) 혜택절 중 min_spend 충족分 → 기대혜택 = 결제금액×rate, 단 monthly_cap 캡
출력: 카드별 기대혜택(원) 정렬. '애매한 후보'는 grade로 넘겨 RAG 판정.
"""
from __future__ import annotations


def expected_benefit(*args, **kwargs):  # pragma: no cover - 후속 라운드 구현
    raise NotImplementedError("규칙 엔진은 데이터 파이프라인 검증 후 설계합니다.")
