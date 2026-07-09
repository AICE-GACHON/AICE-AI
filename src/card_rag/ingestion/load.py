"""[6] 적재: 검수 완료된 clauses JSON → DB(benefit_clauses)로 upsert.

검수 게이트: data/clauses/{card_id}.json 은 '사람이 승인한' 단일 진실원본이다.
숫자 필드는 이 파일에서 사람이 확정한 값이 그대로 규칙 엔진으로 흘러간다.
"""
from __future__ import annotations

from pathlib import Path

from card_rag.db.base import SessionLocal
from card_rag.db.models import BenefitClause, Card
from card_rag.ingestion.embed import build_embedding_text, embed_documents
from card_rag.schemas.clause import ExtractionResult

CLAUSES_DIR = Path("data/clauses")


def load_card(card_id: str, *, name: str, issuer: str, annual_fee: int = 0, highlight: str = "") -> int:
    """검수된 혜택절을 임베딩해 DB에 적재. 반환값=적재된 혜택절 수."""
    result = ExtractionResult.model_validate_json((CLAUSES_DIR / f"{card_id}.json").read_text("utf-8"))

    embed_texts = [build_embedding_text(c) for c in result.clauses]
    vectors = embed_documents(embed_texts) if embed_texts else []

    with SessionLocal() as session:
        session.merge(Card(card_id=card_id, name=name, issuer=issuer,
                           annual_fee=annual_fee, highlight=highlight or None))
        # 카드 단위 재적재: 기존 혜택절 삭제 후 재삽입(멱등).
        session.query(BenefitClause).filter_by(card_id=card_id).delete()
        for clause, etext, vec in zip(result.clauses, embed_texts, vectors):
            session.add(BenefitClause(
                card_id=card_id,
                category=clause.category,
                benefit_type=clause.benefit_type,
                rate=clause.rate,
                monthly_cap=clause.monthly_cap,
                min_spend=clause.min_spend,
                include_notes=clause.include_notes or None,
                exclude_notes=clause.exclude_notes or None,
                source_span=clause.source_span,
                embedding_text=etext,
                embedding=vec,
            ))
        session.commit()
    return len(result.clauses)
