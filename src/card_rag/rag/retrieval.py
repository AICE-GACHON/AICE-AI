"""혜택절 검색(메타데이터 필터 + pgvector 유사도). 3라운드에서 구현.

계획:
- 후보 필터: (card_id ∈ 보유카드) AND (category = 가맹점 업종)  ← SQL WHERE
- 벡터 유사도: 가맹점명/맥락을 embed_query 한 뒤, 위 후보 안에서 exclude/include 조건과 매칭
- pgvector: `embedding <=> :qvec` (cosine) ORDER BY, HNSW 인덱스
반환: 규칙 엔진이 계산한 후보에 '자격 판정 근거 절'을 붙여주는 형태.
"""
from __future__ import annotations


def retrieve_clauses(*args, **kwargs):  # pragma: no cover - 3라운드 설계 대상
    raise NotImplementedError("retrieval은 3라운드(검색·grade 설계)에서 구현합니다.")
