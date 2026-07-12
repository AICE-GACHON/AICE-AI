"""② retrieve + ③ grade — 가맹점 ↔ 포함/제외 조건 의미 매칭.

혜택절은 규칙 엔진이 (card_id, category)로 이미 특정했다. 따라서 벡터의 역할은
'clause 찾기'가 아니라 '이 가맹점이 clause의 포함/제외 조건에 해당하는가'의 신호 생성이다.

- 가맹점(상호명+맥락)을 search_query로 임베딩
- clause의 include_embedding / exclude_embedding 과 각각 코사인 유사도
- sim_include vs sim_exclude 격차가 크면(decisive) LLM 없이 확정, 애매하면 LLM로 넘김
- 근거 불충분(조건이 이 가맹점 유형을 언급 안 함)이면 '관대' 기본값: 제외 명시 없으면 적용
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# 튜닝 파라미터(데이터로 보정 예정)
DECISIVE_MARGIN = 0.08      # |sim_include - sim_exclude| 가 이 값 이상이면 LLM 없이 확정
EXCLUDE_STRONG = 0.55       # exclude만 있을 때 이 이상이면 '제외' 신호로 간주


@dataclass(frozen=True)
class ConditionSignal:
    sim_include: Optional[float]
    sim_exclude: Optional[float]
    lean_included: bool     # 소프트 판정(관대 기본값 반영)
    decisive: bool          # 격차가 커서 LLM 없이 확정 가능한가
    reason: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def condition_signal(
    query_vec: list[float],
    include_vec: Optional[list[float]],
    exclude_vec: Optional[list[float]],
    *,
    margin: float = DECISIVE_MARGIN,
    exclude_strong: float = EXCLUDE_STRONG,
) -> ConditionSignal:
    """가맹점 임베딩과 포함/제외 임베딩 유사도로 자격 신호 생성.

    '관대(lenient)' 기본값(3라운드 결정): 제외 조건에 명확히 걸리지 않으면 적용으로 본다.
    """
    si = cosine(query_vec, include_vec) if include_vec else None
    se = cosine(query_vec, exclude_vec) if exclude_vec else None

    # 제외 조건 자체가 없음 → 무조건 적용(관대), 확정
    if se is None:
        return ConditionSignal(si, None, lean_included=True, decisive=True,
                               reason="제외 조건 없음 → 적용")

    # 포함/제외 둘 다 있음 → 유사도 비교
    if si is not None:
        diff = si - se
        if diff >= margin:
            return ConditionSignal(si, se, True, True, "포함 유사도 우세 → 적용")
        if diff <= -margin:
            return ConditionSignal(si, se, False, True, "제외 유사도 우세 → 미적용")
        return ConditionSignal(si, se, True, False, "포함/제외 유사도 근소 → LLM 판정")

    # 제외만 있음
    if se >= exclude_strong:
        return ConditionSignal(None, se, False, False, "제외 유사도 높음 → LLM 판정")
    return ConditionSignal(None, se, True, True, "제외에 안 걸림 → 적용(관대)")
