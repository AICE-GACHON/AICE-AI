"""LangGraph 배선 — 추천 파이프라인을 상태 그래프로 구성.

  START → gather → compute ──(이긴 애매 혜택절 있음)──▶ judge → compute (루프)
                      └──────(없음)──────▶ END

- compute: 규칙 엔진으로 포락선(구간별 최적 카드) 계산 (숫자는 여기서만)
- judge: 포락선에서 이긴 애매 혜택절만 LLM 자격 판정 (triage 일반화 + 가드레일)
recommend.py의 순수 함수 흐름을 그래프로 옮긴 것. 둘 다 동일 결과.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from card_rag.rag.candidates import fetch_candidates
from card_rag.rag.judge import judge as judge_clause
from card_rag.rag.recommend import RecoResult
from card_rag.rag.types import Judgment
from card_rag.rules.engine import best_by_amount


class RecoState(TypedDict, total=False):
    merchant: str
    categories: list
    card_ids: list
    prev_spend: dict
    cands: list
    judged: dict
    ineligible: list
    segments: list


def _by_id(state: RecoState) -> dict:
    return {c.calc.clause_id: c for c in state["cands"]}


def n_gather(state: RecoState) -> dict:
    return {
        "cands": fetch_candidates(state["card_ids"], state["categories"]),
        "judged": {},
        "ineligible": [],
    }


def n_compute(state: RecoState) -> dict:
    eligible = [c.calc for c in state["cands"] if c.calc.clause_id not in state["ineligible"]]
    return {"segments": best_by_amount(eligible, state["prev_spend"])}


def _winners(state: RecoState) -> list:
    return [s.clause_id for s in state["segments"] if s.clause_id]


def route(state: RecoState) -> str:
    by = _by_id(state)
    todo = [w for w in _winners(state) if w not in state["judged"] and by[w].ambiguous]
    return "judge" if todo else "end"


def n_judge(state: RecoState) -> dict:
    by = _by_id(state)
    judged = dict(state["judged"])
    ineligible = list(state["ineligible"])
    for w in _winners(state):
        if w in judged:
            continue
        c = by[w]
        if not c.ambiguous:
            judged[w] = Judgment(c.calc.card_id, w, True, "high", "조건 없음", "rule_only")
            continue
        j = judge_clause(state["merchant"], c)
        judged[w] = j
        if not j.eligible:
            ineligible.append(w)
    return {"judged": judged, "ineligible": ineligible}


def build_graph():
    g = StateGraph(RecoState)
    g.add_node("gather", n_gather)
    g.add_node("compute", n_compute)
    g.add_node("judge", n_judge)
    g.add_edge(START, "gather")
    g.add_edge("gather", "compute")
    g.add_conditional_edges("compute", route, {"judge": "judge", "end": END})
    g.add_edge("judge", "compute")
    return g.compile()


_GRAPH = None


def recommend_via_graph(merchant, categories, card_ids, prev_spend) -> RecoResult:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke({
        "merchant": merchant, "categories": categories,
        "card_ids": card_ids, "prev_spend": prev_spend,
    })
    judged = final["judged"]
    excluded = [judged[c] for c in final["ineligible"] if c in judged]
    return RecoResult(final["segments"], judged, excluded)
