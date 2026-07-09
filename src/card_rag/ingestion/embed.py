"""[5] 임베딩: Cohere embed-multilingual-v3.0.

문서/질의를 입력타입으로 구분 인코딩한다(문서 확정 사항).
임베딩 대상 텍스트는 '포함/제외 조건 중심' — build_embedding_text 참조.
"""
from __future__ import annotations

import cohere

from card_rag.config import settings
from card_rag.schemas.clause import ExtractedClause

_client: cohere.ClientV2 | None = None


def _client_() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=settings.cohere_api_key)
    return _client


def build_embedding_text(clause: ExtractedClause) -> str:
    """포함/제외 조건을 중심으로 임베딩 문자열 구성(가맹점 자격 매칭 최적화)."""
    parts = [f"업종:{clause.category}"]
    if clause.include_notes:
        parts.append(f"포함:{clause.include_notes}")
    if clause.exclude_notes:
        parts.append(f"제외:{clause.exclude_notes}")
    return " | ".join(parts)


def embed_documents(texts: list[str]) -> list[list[float]]:
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
