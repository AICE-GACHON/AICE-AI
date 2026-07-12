"""[5] 임베딩: Cohere embed-multilingual-v3.0.

문서/질의를 입력타입으로 구분 인코딩한다(문서 확정 사항).
포함/제외 조건을 **분리 임베딩**해(3라운드 결정) 가맹점 query와 각각 유사도를 비교,
sim_include vs sim_exclude 로 자격 신호를 만든다.
"""
from __future__ import annotations

from typing import Optional

import cohere

from card_rag.config import settings
from card_rag.schemas.clause import ExtractedClause

_client: Optional[cohere.ClientV2] = None


def _client_() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=settings.cohere_api_key)
    return _client


def build_include_text(clause: ExtractedClause) -> Optional[str]:
    """포함 조건 임베딩 문자열. 포함 조건이 없으면 None(임베딩 생략)."""
    if not clause.include_notes:
        return None
    return f"업종:{clause.category} 포함:{clause.include_notes}"


def build_exclude_text(clause: ExtractedClause) -> Optional[str]:
    """제외 조건 임베딩 문자열. 제외 조건이 없으면 None(임베딩 생략)."""
    if not clause.exclude_notes:
        return None
    return f"업종:{clause.category} 제외:{clause.exclude_notes}"


def build_search_text(clause: ExtractedClause, card_name: str) -> str:
    """Q&A/일반 검색용 전체 임베딩 문자열(카드명 + 업종 + 혜택 + 조건 통합)."""
    if clause.value_type == "percent":
        val = f"{clause.rate}%"
    else:
        val = f"건당 {clause.flat_amount}원(결제 {clause.flat_min_txn}원 이상)"
    parts = [card_name, clause.category, clause.benefit_type, val]
    if clause.include_notes:
        parts.append(f"포함 {clause.include_notes}")
    if clause.exclude_notes:
        parts.append(f"제외 {clause.exclude_notes}")
    return " ".join(str(p) for p in parts)


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = _client_().embed(
        texts=texts,
        model=settings.cohere_embed_model,
        input_type="search_document",
        embedding_types=["float"],
    )
    return resp.embeddings.float_


def embed_query(text: str) -> list[float]:
    resp = _client_().embed(
        texts=[text],
        model=settings.cohere_embed_model,
        input_type="search_query",
        embedding_types=["float"],
    )
    return resp.embeddings.float_[0]
