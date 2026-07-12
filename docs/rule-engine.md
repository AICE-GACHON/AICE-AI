# 규칙 엔진 설계 (4라운드)

결정적 계산기. 숫자는 여기서 확정되고 LLM은 절대 못 바꾼다(가드레일).

## 핵심: 결제금액 A를 변수로 (breakeven 추천)
추천 시점엔 결제액을 모름 → A를 고정값으로 가정하지 않고 **기대혜택을 A의 함수**로 본다.
각 혜택절을 A에 대한 선분으로 만들고 **상단 포락선(upper envelope)** 을 구해
"N원 이상이면 X카드가 유리" 형태의 구간별 추천을 낸다.

- percent: `y = min(rate/100 × A, remaining_cap) × factor` (기울기 선 → 한도 포화)
- flat: `y = min(flat_amount, remaining_cap) × factor`, `A ≥ flat_min_txn` (계단)
- 전월실적(min_spend) 미충족 절은 후보 제외
- 구현 `rules/engine.py:best_by_amount` · 렌더 `render_advice` · 검증 `tests/test_rule_engine.py`

예 (카페, K-패스 실적 30만 충족):
```
1만원 미만   → 토스뱅크 (정액 100원)
1만원~5만원  → 토스뱅크 (정액 500원)
5만원 이상   → K-패스 하나 (1%, 500원~)
```

## 스키마 확장 (실제 카드가 드러낸 이슈)
`benefit_clauses`에 추가: `value_type(percent|flat)`, `flat_amount`, `flat_min_txn`.
- 정률/정액을 한 스키마로. '1만원 이상/미만'은 `flat_min_txn`이 다른 **개별 row**(전월실적 구간 분리와 동일 철학).
- `schemas/clause.py`·`ingestion/extract.py`(프롬프트+tool)·`ingestion/load.py` 반영.

## MVP 결정
- 월 한도 소진 **반영**: `remaining_cap`(남은 월 한도)을 입력으로. None=무제한.
- 적립 실질가치 `value_factor` 기본 1.0(캐시백/할인). 카드별 포인트 계수는 나중.

## triage 연동 (다음)
triage는 지금 스칼라(included_won/excluded_won) 기반. 엔진이 A의 함수를 내므로,
'애매한 절을 제외했을 때 **포락선(구간별 최적 카드 map)** 이 바뀌는가'로 pivotal 판정을 일반화 가능.
→ 5라운드에서 triage와 엔진을 연결 + LLM 판정 노드(reason/guardrail) 구현.
