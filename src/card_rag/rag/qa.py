"""B. 자연어 Q&A — 사용자의 자유 질문에 혜택절 근거를 인용해 답한다.

검색: 질문을 Cohere search_query로 임베딩 → pgvector 코사인 유사도로 관련 혜택절 top-k 조회.
생성: Haiku가 검색된 혜택절만 근거로, 숫자는 근거값만 써서 출처와 함께 답한다(환각 방지).
"""
from __future__ import annotations

import anthropic
from sqlalchemy import select

from card_rag.config import settings
from card_rag.db.base import SessionLocal
from card_rag.db.models import BenefitClause, Card, PolicyClause
from card_rag.ingestion.embed import embed_query

_SYS = (
    "너는 카드 혜택 상담원이다. 아래 '혜택절'만 근거로 한국어로 간결히 답한다.\n"
    "- 숫자(금액·%·한도·실적)는 근거에 있는 값만 쓴다. 추정 금지.\n"
    "- 각 사실 끝에 (카드명·업종) 형태로 출처를 단다.\n"
    "- 근거에 없으면 '해당 정보는 확인되지 않는다'고 말한다.\n"
    "- 포함/제외 조건이 질문의 가맹점에 걸리는지 반드시 따진다.\n"
    "- '정책·예외규칙'(전역 제외·통합한도·특약)도 반드시 함께 따져 최종 자격을 판단한다."
)


def _clause_text(bc: BenefitClause, card_name: str) -> str:
    if bc.value_type == "percent":
        val = f"{bc.rate}%"
    else:
        val = f"건당 {bc.flat_amount}원(결제 {bc.flat_min_txn}원 이상)"
    parts = [card_name, bc.category, bc.benefit_type, val]
    if bc.monthly_cap:
        parts.append(f"월한도 {bc.monthly_cap}원")
    if bc.min_spend:
        parts.append(f"전월실적 {bc.min_spend}원 이상")
    if bc.include_notes:
        parts.append(f"포함:{bc.include_notes}")
    if bc.exclude_notes:
        parts.append(f"제외:{bc.exclude_notes}")
    return " · ".join(str(p) for p in parts)


def retrieve(question: str, k: int = 12) -> list[tuple[BenefitClause, str]]:
    """질문 임베딩과 혜택절 search_embedding의 pgvector 코사인 유사도로 top-k 조회."""
    qvec = embed_query(question)
    with SessionLocal() as session:
        rows = session.execute(
            select(BenefitClause, Card.name)
            .join(Card, Card.card_id == BenefitClause.card_id)
            .where(BenefitClause.search_embedding.isnot(None))
            .order_by(BenefitClause.search_embedding.cosine_distance(qvec))
            .limit(k)
        ).all()
    return [(bc, name) for bc, name in rows]


def retrieve_policies(question: str, k: int = 6) -> list[tuple[PolicyClause, str]]:
    """질문과 정책 청크(전역 제외·통합한도·특약) 임베딩의 코사인 유사도로 top-k 조회."""
    qvec = embed_query(question)
    with SessionLocal() as session:
        rows = session.execute(
            select(PolicyClause, Card.name)
            .join(Card, Card.card_id == PolicyClause.card_id)
            .where(PolicyClause.embedding.isnot(None))
            .order_by(PolicyClause.embedding.cosine_distance(qvec))
            .limit(k)
        ).all()
    return [(pc, name) for pc, name in rows]


def answer(question: str, *, k: int = 12, k_policy: int = 6) -> str:
    hits = retrieve(question, k=k)
    pol_hits = retrieve_policies(question, k=k_policy)
    benefit_ctx = "\n".join(f"- {_clause_text(bc, name)}" for bc, name in hits)
    policy_ctx = "\n".join(f"- [{pc.policy_type}] {name}: {pc.text}" for pc, name in pol_hits)
    user = f"[혜택절]\n{benefit_ctx}\n\n[정책·예외규칙]\n{policy_ctx}\n\n[질문] {question}"
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=700,
        system=[{"type": "text", "text": _SYS, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
