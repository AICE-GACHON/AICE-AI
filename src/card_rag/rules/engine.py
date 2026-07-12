"""결정적 규칙 엔진 — 기대혜택을 '결제금액 A의 함수'로 보고 구간별 최적 카드를 계산.

핵심 아이디어(4라운드): 추천 시점엔 결제액을 모르므로 A를 고정값으로 가정하지 않는다.
각 혜택절을 A에 대한 '선분'으로 만들고 상단 포락선(upper envelope)을 구해
"N원 이상이면 X카드가 유리" 형태의 구간별 추천을 낸다.

- percent: y = min(rate/100 × A, remaining_cap) × factor   (기울기 선 → 한도에서 포화)
- flat:    y = min(flat_amount, remaining_cap) × factor,  A ≥ flat_min_txn 에서만  (계단)
- 전월실적(min_spend) 미충족 혜택절은 후보에서 제외

LLM은 이 숫자를 절대 못 바꾼다(가드레일). 출력 BenefitSegment는 triage/RAG의 입력이 된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from card_rag.rules.types import FLAT, PERCENT, BenefitSegment, ClauseCalc

AMAX = 1_000_000   # 계산 상한(원). 이 이상은 '그 이상' 구간으로 취급.
_EPS = 1e-9


@dataclass(frozen=True)
class _Seg:
    """선분: x∈[x0,x1] 에서 y = slope*x + intercept. 특정 혜택절에 귀속."""

    x0: float
    x1: float
    slope: float
    intercept: float
    clause: ClauseCalc

    def y(self, x: float) -> float:
        return self.slope * x + self.intercept


def _segments_for(c: ClauseCalc, amax: int) -> list[_Seg]:
    factor = c.value_factor
    cap = c.remaining_cap
    if cap is not None and cap <= 0:
        return []  # 한도 소진 → 혜택 없음

    if c.value_type == PERCENT:
        slope = (c.rate / 100.0) * factor
        if slope <= 0:
            return []
        if cap is None:
            return [_Seg(0.0, float(amax), slope, 0.0, c)]
        a_sat = cap / (c.rate / 100.0)          # 이 결제액에서 한도 포화
        capped = cap * factor
        segs = [_Seg(0.0, min(a_sat, amax), slope, 0.0, c)]
        if a_sat < amax:
            segs.append(_Seg(a_sat, float(amax), 0.0, capped, c))
        return segs

    if c.value_type == FLAT:
        value = c.flat_amount if cap is None else min(c.flat_amount, cap)
        value *= factor
        if value <= 0:
            return []
        return [_Seg(float(c.flat_min_txn), float(amax), 0.0, value, c)]

    return []


def _intersect(a: _Seg, b: _Seg) -> Optional[float]:
    if abs(a.slope - b.slope) < _EPS:
        return None
    return (b.intercept - a.intercept) / (a.slope - b.slope)


def best_by_amount(
    clauses: list[ClauseCalc],
    prev_month_spend: dict[str, int],
    *,
    amax: int = AMAX,
) -> list[BenefitSegment]:
    """전월실적을 충족하는 혜택절만으로 구간별 최적 카드를 계산."""
    eligible = [c for c in clauses if prev_month_spend.get(c.card_id, 0) >= c.min_spend]
    segs: list[_Seg] = []
    for c in eligible:
        segs.extend(_segments_for(c, amax))

    if not segs:
        return [BenefitSegment(0, None, None, None, 0)]

    # 후보 경계점: 선분 끝점 + 쌍별 교차점
    xs = {0.0, float(amax)}
    for s in segs:
        xs.add(s.x0)
        xs.add(min(s.x1, amax))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            x = _intersect(segs[i], segs[j])
            if x is not None and 0 <= x <= amax:
                xs.add(x)
    bounds = sorted(xs)

    raw: list[tuple[float, float, Optional[_Seg]]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < _EPS:
            continue
        mid = (a + b) / 2
        best: Optional[_Seg] = None
        best_y = 0.0
        for s in segs:
            if s.x0 - _EPS <= mid <= s.x1 + _EPS:
                y = s.y(mid)
                if y > best_y + _EPS:
                    best_y, best = y, s
        raw.append((a, b, best))

    return _merge(raw, amax)


def render_advice(segments: list[BenefitSegment], card_names: dict[str, str]) -> list[str]:
    """구간을 '결제 N원 이상이면 X카드가 유리' 형태의 한국어 문구로 렌더링."""

    def won(n: int) -> str:
        if n and n % 10000 == 0:
            return f"{n // 10000}만원"
        return f"{n:,}원"

    lines: list[str] = []
    for s in segments:
        name = card_names.get(s.card_id, s.card_id) if s.card_id else "혜택 없음"
        if s.a_to is None:
            span = f"{won(s.a_from)} 이상" if s.a_from else "전 구간"
        elif s.a_from == 0:
            span = f"{won(s.a_to)} 미만"
        else:
            span = f"{won(s.a_from)}~{won(s.a_to)}"
        tail = "" if s.card_id is None else f" (약 {s.benefit_at_from:,}원~)"
        lines.append(f"{span} 결제 시 → {name}{tail}")
    return lines


def _merge(raw, amax: int) -> list[BenefitSegment]:
    """인접한 동일 카드 구간 병합 후 BenefitSegment로 변환."""
    out: list[BenefitSegment] = []
    for a, b, seg in raw:
        cid = seg.clause.card_id if seg else None
        clid = seg.clause.clause_id if seg else None
        if out and out[-1].card_id == cid and out[-1].clause_id == clid:
            prev = out[-1]
            out[-1] = BenefitSegment(prev.a_from, int(round(b)), cid, clid, prev.benefit_at_from)
        else:
            y = int(round(seg.y(a))) if seg else 0
            out.append(BenefitSegment(int(round(a)), int(round(b)), cid, clid, y))
    if out:
        last = out[-1]
        out[-1] = BenefitSegment(last.a_from, None if last.a_to and last.a_to >= amax else last.a_to,
                                 last.card_id, last.clause_id, last.benefit_at_from)
    return out
