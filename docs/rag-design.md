# RAG 파이프라인 설계 (3라운드)

하이브리드 추천엔진의 RAG 부분. 숫자는 규칙 엔진이 결정, 애매한 자격 판정·설명만 RAG-LLM.

## 실효 흐름

```
규칙 엔진 (후보 카드 + 기대혜택 숫자)
   │
   ▼
① triage ── 이 후보, RAG 필요?  (조건성 OR 카테고리불확실) AND 결정민감
   ├─ 아니오 → rule_only (숫자 + 사전생성/템플릿 설명)
   └─ 예
        ▼
② retrieve ─ 가맹점(query) ↔ clause의 include/exclude 임베딩 유사도
        ▼
③ grade ─── decisive? (격차 큼 → LLM 생략) / 근거 불충분 → 관대 폴백
        ▼
④ reason ── Claude Haiku 구조화 출력: eligible? + 근거문장   [4라운드]
        ▼
⑤ guardrail  규칙 숫자 불변 + 스키마 검증, 위반 시 rule_only('참고용')  [4라운드]
```

## 결정 사항

### 1. triage — '애매함' 판정 (agentic 라우팅 핵심)
**RAG_JUDGE ⇔ (조건성 OR 카테고리 불확실) AND 결정 민감**
- 조건성: 혜택절에 include/exclude 존재
- 카테고리 불확실: 가맹점 업종이 임베딩 폴백으로 매핑됨
- 결정 민감: 후보 자격이 뒤집혔을 때 결정 시그니처 `(Top-1, Top-k 집합)`가 바뀔 수 있는가.
  후보 수(~보유카드 2–10)라 2^k 조합 완전탐색으로 정확 판정.
- 구현: `rag/triage.py` · 검증: `tests/test_triage.py` (순위 안 바꾸는 애매 후보는 rule_only로 걸러짐)

### 2. retrieve — pgvector 역할
혜택절은 규칙 엔진이 (card_id, category)로 이미 특정. 벡터는 **가맹점 ↔ 조건 매칭**.
- 스키마: 혜택절당 **include_embedding / exclude_embedding 분리 저장** (`db/models.py`)
- `sim_include` vs `sim_exclude` 격차가 크면(decisive) LLM 생략, 애매하면 LLM
- 구현: `rag/retrieval.py:condition_signal` · 검증: `tests/test_retrieval.py`

### 3. grade — 근거 불충분 시 기본값
**관대(lenient)**: 제외 조건에 명확히 걸리지 않으면 적용(+낮은 confidence). 추천 누락 최소화.

## 튜닝 파라미터 (데이터로 보정)
- `retrieval.DECISIVE_MARGIN = 0.08` — LLM 생략 임계
- `retrieval.EXCLUDE_STRONG = 0.55` — 제외만 있을 때 제외 신호 임계
- `triage.MAX_BRUTEFORCE = 16` — 초과 시 애매 후보 전부 판정(안전측)

## 다음 (4라운드)
- LLM 구조화 출력 스키마: `{eligible, confidence, reason_kr, caveat}`
- guardrail: 규칙 숫자 정합성 + 스키마 검증 + '참고용' 폴백
- LangGraph StateGraph 배선 (`rag/graph.py`)
- 규칙 엔진 (`rules/engine.py`) — triage 입력 RuleCandidate 생성
