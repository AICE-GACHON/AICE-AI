"""추천 오케스트레이션 — 규칙 엔진 + 포락선 기반 triage + LLM 판정 + 가드레일.

흐름: 후보수집 → 규칙엔진(포락선) → 이긴 애매 혜택절만 판정 → 제외분 빼고 재계산 → 결과.
숫자는 규칙 엔진에서만 나온다(LLM은 자격/근거만).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from card_rag.rag.candidates import CandidateClause, fetch_candidates
from card_rag.rag.judge import judge
from card_rag.rag.types import Judgment
from card_rag.rules.engine import best_by_amount
from card_rag.rules.types import BenefitSegment

MAX_ROUNDS = 5  # 판정으로 포락선이 바뀌면 다시 판정(연쇄) 하는 최대 횟수


@dataclass
class RecoResult:
    segments: list[BenefitSegment]
    judgments: dict[str, Judgment] = field(default_factory=dict)   # clause_id -> Judgment
    excluded: list[Judgment] = field(default_factory=list)         # 판정으로 제외된 것


def recommend(
    merchant_name: str,
    categories: list[str],
    card_ids: list[str],
    prev_month_spend: dict[str, int],
) -> RecoResult:
    cands: list[CandidateClause] = fetch_candidates(card_ids, categories)
    by_id = {c.calc.clause_id: c for c in cands}

    judged: dict[str, Judgment] = {}
    ineligible: set[str] = set()

    for _ in range(MAX_ROUNDS):
        eligible = [c.calc for c in cands if c.calc.clause_id not in ineligible]
        segments = best_by_amount(eligible, prev_month_spend)
        winners = [s.clause_id for s in segments if s.clause_id]

        # ③ 포락선에서 이긴 혜택절 중, 아직 판정 안 한 '애매한' 것만 대상 (triage 일반화)
        todo = [w for w in winners if w not in judged and by_id[w].ambiguous]
        # 조건 없는 승자는 rule_only로 즉시 통과 기록
        for w in winners:
            if w not in judged and not by_id[w].ambiguous:
                c = by_id[w]
                judged[w] = Judgment(c.calc.card_id, w, True, "high", "조건 없음", "rule_only")
        if not todo:
            break

        for w in todo:  # ④ LLM 판정
            j = judge(merchant_name, by_id[w])
            judged[w] = j
            if not j.eligible:
                ineligible.add(w)

    eligible = [c.calc for c in cands if c.calc.clause_id not in ineligible]
    final = best_by_amount(eligible, prev_month_spend)  # ⑤ 최종 재계산
    excluded = [judged[c] for c in ineligible if c in judged]
    return RecoResult(final, judged, excluded)


def render(result: RecoResult, card_names: dict[str, str]) -> str:
    """사람이 읽는 추천 문구."""
    from card_rag.rules.engine import render_advice

    lines = render_advice(result.segments, card_names)
    out = ["[추천]"] + [f"  {ln}" for ln in lines]
    # 이긴 혜택절의 판정 근거
    shown = {s.clause_id for s in result.segments if s.clause_id}
    reasons = [j for cid, j in result.judgments.items() if cid in shown and j.source in ("llm", "rule_signal")]
    if reasons:
        out.append("[근거]")
        out += [f"  - {card_names.get(j.card_id, j.card_id)}: {j.reason}" for j in reasons]
    if result.excluded:
        out.append("[제외됨]")
        out += [f"  - {card_names.get(j.card_id, j.card_id)}: {j.reason}" for j in result.excluded]
    return "\n".join(out)
