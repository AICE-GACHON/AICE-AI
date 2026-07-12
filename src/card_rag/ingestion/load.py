"""[6] 적재: 검수 완료된 clauses JSON → DB(benefit_clauses)로 upsert.

검수 게이트: data/clauses/{card_id}.json 은 '사람이 승인한' 단일 진실원본이다.
숫자 필드는 이 파일에서 사람이 확정한 값이 그대로 규칙 엔진으로 흘러간다.
포함/제외 조건은 각각 임베딩해 저장(3라운드 결정).
"""
from __future__ import annotations

from pathlib import Path

from card_rag.db.base import SessionLocal
from card_rag.db.models import BenefitClause, Card, PolicyClause
from card_rag.ingestion.embed import (
    build_exclude_text,
    build_include_text,
    build_search_text,
    embed_documents,
)
from card_rag.schemas.clause import ExtractionResult
from card_rag.schemas.policy import PolicyExtractionResult

CLAUSES_DIR = Path("data/clauses")
POLICIES_DIR = Path("data/policies")


def load_card(card_id: str, *, name: str, issuer: str, annual_fee: int = 0, highlight: str = "") -> int:
    """검수된 혜택절을 임베딩해 DB에 적재. 반환값=적재된 혜택절 수."""
    result = ExtractionResult.model_validate_json((CLAUSES_DIR / f"{card_id}.json").read_text("utf-8"))

    inc_texts = [build_include_text(c) for c in result.clauses]
    exc_texts = [build_exclude_text(c) for c in result.clauses]
    search_texts = [build_search_text(c, name) for c in result.clauses]
    # None(조건 없음)은 임베딩 대상에서 제외하고, 인덱스로 되돌려 매핑한다.
    inc_vecs = _embed_optional(inc_texts)
    exc_vecs = _embed_optional(exc_texts)
    search_vecs = embed_documents(search_texts) if search_texts else []

    with SessionLocal() as session:
        session.merge(Card(card_id=card_id, name=name, issuer=issuer,
                           annual_fee=annual_fee, highlight=highlight or None))
        session.query(BenefitClause).filter_by(card_id=card_id).delete()  # 카드 단위 멱등 재적재
        for i, clause in enumerate(result.clauses):
            session.add(BenefitClause(
                card_id=card_id,
                category=clause.category,
                benefit_type=clause.benefit_type,
                value_type=clause.value_type,
                rate=clause.rate,
                flat_amount=clause.flat_amount,
                flat_min_txn=clause.flat_min_txn,
                monthly_cap=clause.monthly_cap,
                min_spend=clause.min_spend,
                include_notes=clause.include_notes or None,
                exclude_notes=clause.exclude_notes or None,
                source_span=clause.source_span,
                include_text=inc_texts[i],
                include_embedding=inc_vecs[i],
                exclude_text=exc_texts[i],
                exclude_embedding=exc_vecs[i],
                search_text=search_texts[i],
                search_embedding=search_vecs[i],
            ))
        session.commit()
    return len(result.clauses)


def load_policies(card_id: str) -> int:
    """정책 청크를 임베딩해 DB에 적재(카드는 load_card로 먼저 적재돼 있어야 함)."""
    path = POLICIES_DIR / f"{card_id}.json"
    if not path.exists():
        return 0
    result = PolicyExtractionResult.model_validate_json(path.read_text("utf-8"))
    texts = [p.text for p in result.policies]
    vecs = embed_documents(texts) if texts else []
    with SessionLocal() as session:
        session.query(PolicyClause).filter_by(card_id=card_id).delete()  # 멱등 재적재
        for p, v in zip(result.policies, vecs):
            session.add(PolicyClause(
                card_id=card_id,
                policy_type=p.policy_type,
                category=p.category,
                text=p.text,
                source_span=p.source_span,
                embedding=v,
            ))
        session.commit()
    return len(result.policies)


def _embed_optional(texts: list[str | None]) -> list[list[float] | None]:
    """None은 건너뛰고 나머지만 임베딩한 뒤 원래 위치로 되돌린다."""
    idx = [i for i, t in enumerate(texts) if t]
    vecs = embed_documents([texts[i] for i in idx])
    out: list[list[float] | None] = [None] * len(texts)
    for j, i in enumerate(idx):
        out[i] = vecs[j]
    return out
