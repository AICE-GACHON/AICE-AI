"""① triage — '애매한 후보만 LLM' 라우팅 (agentic 라우팅의 핵심).

규칙(3라운드 확정):
    RAG_JUDGE  ⇔  (조건성 OR 카테고리 불확실)  AND  결정 민감

- 조건성: 혜택절에 include/exclude 조건이 존재 (has_condition)
- 카테고리 불확실: 가맹점 업종이 임베딩 폴백으로 매핑됨 (merchant.category_uncertain)
- 결정 민감: 이 후보의 자격이 뒤집혔을 때 '결정 시그니처'(Top-1 정체성 + Top-k 집합)가
  바뀔 수 있는가. 후보 수가 보유카드 수(~2–10)라 조합 완전탐색으로 정확히 판정한다.

즉 LLM은 '조건이 애매하고 그 결과가 답을 바꿀 때'만 호출된다.
"""
from __future__ import annotations

from typing import Iterable

from card_rag.rag.types import RAG_JUDGE, RULE_ONLY, MerchantCtx, RuleCandidate, TriageDecision

# 결정 민감도 완전탐색 상한. 초과 시 애매 후보를 모두 RAG_JUDGE로 처리(안전측).
MAX_BRUTEFORCE = 16


def _condition_ambiguous(c: RuleCandidate, merchant: MerchantCtx) -> bool:
    return c.has_condition or merchant.category_uncertain


def _signature(realized: dict[str, int], top_k: int) -> tuple[str, frozenset[str]]:
    """결정 시그니처 = (Top-1 카드, Top-k 카드 집합). 동점은 card_id 오름차순으로 tie-break."""
    order = sorted(realized.items(), key=lambda kv: (-kv[1], kv[0]))
    top1 = order[0][0] if order else ""
    topk = frozenset(cid for cid, _ in order[:top_k])
    return top1, topk


def triage(
    candidates: Iterable[RuleCandidate],
    merchant: MerchantCtx,
    *,
    top_k: int = 3,
) -> list[TriageDecision]:
    cands = list(candidates)
    ambiguous = [c for c in cands if _condition_ambiguous(c, merchant)]

    # 애매 후보가 없으면 전부 rule_only
    if not ambiguous:
        return [TriageDecision(c.card_id, c.clause_id, RULE_ONLY, "조건 없음/업종 확실") for c in cands]

    k = len(ambiguous)
    if k > MAX_BRUTEFORCE:
        pivotal = set(id(c) for c in ambiguous)  # 안전측: 전부 판정
    else:
        pivotal = _pivotal_set(cands, ambiguous, top_k, k)

    decisions: list[TriageDecision] = []
    for c in cands:
        if id(c) in pivotal:
            decisions.append(TriageDecision(c.card_id, c.clause_id, RAG_JUDGE, "조건 애매 + 순위 좌우"))
        else:
            amb = _condition_ambiguous(c, merchant)
            why = "조건 애매하나 순위 불변" if amb else "조건 없음/업종 확실"
            decisions.append(TriageDecision(c.card_id, c.clause_id, RULE_ONLY, why))
    return decisions


def _pivotal_set(
    cands: list[RuleCandidate],
    ambiguous: list[RuleCandidate],
    top_k: int,
    k: int,
) -> set[int]:
    """애매 후보 i가 pivotal ⇔ 다른 애매 후보들의 어떤 배정에서든 i의 포함/제외 토글이
    결정 시그니처를 바꾸면 True. 2^k 조합을 돌며 (다른 것 고정, i만 토글) 쌍을 비교한다."""
    fixed = {c.card_id: c.included_won for c in cands if c not in ambiguous}
    sig_cache: dict[int, tuple[str, frozenset[str]]] = {}

    def sig_for(combo: int) -> tuple[str, frozenset[str]]:
        if combo not in sig_cache:
            realized = dict(fixed)
            for bit, c in enumerate(ambiguous):
                realized[c.card_id] = c.realized(included=bool(combo & (1 << bit)))
            sig_cache[combo] = _signature(realized, top_k)
        return sig_cache[combo]

    pivotal: set[int] = set()
    for combo in range(1 << k):
        for bit in range(k):
            if combo & (1 << bit):
                continue  # bit=0인 조합에서만 (0→1) 토글을 검사하면 모든 배정을 1회씩 커버
            if sig_for(combo) != sig_for(combo | (1 << bit)):
                pivotal.add(id(ambiguous[bit]))
    return pivotal
