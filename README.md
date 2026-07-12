# card-rag

AICE 위치기반 카드혜택 추천 — **하이브리드 추천엔진(규칙 엔진 + RAG)** 프로토타입.

> 숫자(전월실적·한도·적립/할인 실질가치)는 **결정적 규칙 엔진**이, 애매한 자격 판정(포함/제외 가맹점)과
> 자연어 설명만 **RAG-LLM**이 담당한다. LLM은 규칙 엔진의 숫자를 **절대 뒤집지 못한다(가드레일)**.

`card-poke`에서 먼저 검증하는 **독립 프로토타입**이며, 이후 BE(FastAPI)로 이식한다.
전체 개요·설계는 **[docs/00_전체정리.md](docs/00_전체정리.md)** 참고.

## 현재 상태
카드 3장(토스뱅크·하나 나라사랑·K-패스 하나 체크)으로 **인제스천~추천 전 과정이 실제로 동작**한다.
- 🅰 **인제스천**: 약관 원문 → 정규화 → LLM 추출 → 자동 검증 → 임베딩 → pgvector 적재 (혜택절 25 + 정책 13)
- 🅱 **추천(end-to-end)**: 규칙 엔진(구간별 금액) + triage + LLM 자격판정 + 가드레일, **LangGraph 배선**
- ➕ **자연어 Q&A**: 질문 → 혜택절·정책 벡터검색 → 근거 인용 답변

## 스택
| 구분 | 채택 |
|---|---|
| 임베딩 | Cohere `embed-multilingual-v3.0` (1024d, document/query 구분) |
| 벡터DB | pgvector (Postgres 16) |
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5`) — tool-use 구조화 출력 |
| 오케스트레이션 | LangGraph (StateGraph) |
| 런타임 | Python 3.14 (64bit), SQLAlchemy 2 + psycopg 3, Pydantic v2 |

## 프로젝트 구조
```
src/card_rag/
  config.py            설정(.env)
  db/models.py         테이블: cards, benefit_clauses(+vector), policy_clauses(+vector) 등
  schemas/             Pydantic 계약: clause.py(혜택절), policy.py(정책)
  ingestion/           normalize · extract · extract_policies · validate · embed · load
  rules/               결정적 규칙 엔진: engine.py(구간별 최적카드), types.py
  rag/                 triage · retrieval · candidates · judge · recommend · qa · graph(LangGraph)
  cli.py               CLI
data/                  raw / normalized / clauses / policies / cards.json  (data/README.md)
docs/                  00_전체정리 · rag-design · rule-engine · ingestion-run
docker-compose.yml     pgvector 포함 Postgres 16
tests/                 순수 로직 테스트(triage · retrieval · rule_engine)
```

## 사전 요구사항
- **Python 3.14 (64bit)** — 32비트는 psycopg/cohere wheel 부재로 불가
- **Docker Desktop** (pgvector Postgres) 또는 pgvector 설치된 Postgres 16
- API 키: `ANTHROPIC_API_KEY`, `COHERE_API_KEY`

## 설치
```bash
cp .env.example .env                       # 키 채우기
docker compose up -d db                     # pgvector Postgres 기동
python -m venv .venv                         # 64bit 인터프리터로
.venv\Scripts\python.exe -m pip install -e . # 의존성 설치

.venv\Scripts\python.exe -m card_rag.cli init-db   # 확장 + 테이블
```

## 사용
> 명령은 `.venv\Scripts\python.exe -m card_rag.cli <command>` 로 실행(콘솔 UTF-8 자동 설정됨).

**카드 적재 / 추가**
```bash
# data/raw/<card_id>/benefits.md 원문 + data/cards.json 메타를 넣은 뒤:
card_rag.cli ingest <card_id>     # normalize→extract(+정책)→자동검증→load 완주
card_rag.cli load   <card_id>     # 검증된 혜택절/정책 임베딩+적재만
```

**추천 (end-to-end)**
```bash
card_rag.cli recommend "스타벅스 강남점"
#  → 1만원 미만: 토스뱅크(100원) / 1만~5만: 토스뱅크(500원) / 5만 이상: K-패스(1%)
#  → [제외됨] 블루보틀 등 화이트리스트 미포함 시 해당 카드 자격 미달 표시
```

**자연어 Q&A**
```bash
card_rag.cli ask "K-패스로 지하철 타면 얼마 돌려받아?"
card_rag.cli ask "토스뱅크로 상품권 사면 캐시백 돼?"     # 전역 제외 정책까지 반영
```

## 파이프라인 (인제스천)
```
[1] collect        data/raw/{card_id}/                     (수동, 소규모)
[2] normalize      HTML/PDF/MD → 텍스트                     card-rag normalize
[3] extract        약관 → 혜택절 JSON  (Claude Haiku, tool-use)  card-rag extract
[3'] extract-policies  약관 → 정책 청크(전역제외·통합한도·특약)  card-rag extract-policies
[4] validate       스키마·정합성 자동 검증(사람 검수 대체)       (ingest에 포함)
[5] embed          포함/제외 + 전체 검색용 임베딩 (Cohere)       (load에 포함)
[6] load           benefit_clauses · policy_clauses 적재        card-rag load
```

## 테스트 (설치 불필요, 순수 로직)
```bash
python tests/test_rule_engine.py   # 결제금액 구간별 최적카드
python tests/test_triage.py        # 애매+순위좌우일 때만 LLM
python tests/test_retrieval.py     # 가맹점↔조건 매칭 + 관대 폴백
```

## 로드맵
- [x] 스캐폴드 + 인제스천 파이프라인
- [x] 카드 3장 수집 → 추출·자동검증·적재 (전체 자동화)
- [x] 검색(retrieval) + triage('애매함' 판정)
- [x] 규칙 엔진 (결제금액 함수 → 구간별 추천)
- [x] 자연어 Q&A (혜택절 + 정책 청크 벡터검색)
- [x] 약관 심화 판정 (정책 청크: 전역제외·통합한도·특약)
- [x] 추천 end-to-end (LLM 자격판정 + 가드레일 + LangGraph)
- [ ] 추출 품질 튜닝 (브랜드 누락·경계값·업종 taxonomy)
- [ ] 가맹점→업종 매핑(Kakao Local) · 한도 소진 실데이터
- [ ] BE(FastAPI) 이식 · 카드 14장 확장
