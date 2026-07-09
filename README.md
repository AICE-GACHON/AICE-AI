# card-rag

AICE 위치기반 카드혜택 추천 — **하이브리드 추천엔진(규칙 엔진 + RAG)** 프로토타입.

> 숫자(전월실적·한도·적립/할인 실질가치)는 **결정적 규칙 엔진**이, 애매한 자격 판정(포함/제외 가맹점)과
> 자연어 설명만 **RAG-LLM**이 담당한다. LLM은 규칙 엔진의 숫자를 **절대 뒤집지 못한다(가드레일)**.

이 레포는 `card-poke`에서 먼저 검증하는 **독립 프로토타입**이며, 이후 BE(FastAPI) 레포로 이식한다.

## 스택 (문서 확정)
| 구분 | 채택 |
|---|---|
| 임베딩 | Cohere `embed-multilingual-v3.0` (1024d, document/query 구분) |
| 벡터DB | pgvector (Postgres 16, HNSW) |
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5`) |
| 오케스트레이션 | LangGraph (retrieve→grade→reason→generate→guardrail) |

## 프로젝트 구조
```
src/card_rag/
  config.py            설정(.env)
  db/                  SQLAlchemy 엔진·모델(cards, benefit_clauses[+vector], ...)
  schemas/             Pydantic 계약(ExtractedClause 등)
  ingestion/           수집→정규화→추출→(검수)→임베딩→적재
  rag/                 LangGraph 검색·판정 (3라운드)
  rules/               결정적 규칙 엔진 (후속)
  cli.py               인제스천 CLI
data/                  raw / normalized / clauses  (data/README.md 참고)
docker-compose.yml     pgvector 포함 Postgres 16
```

## 사전 요구사항
- **Python 3.11+** (현재 로컬 3.9 → 업그레이드 필요)
- **Docker Desktop** (pgvector Postgres 실행용) 또는 pgvector가 설치된 Postgres 16
- API 키: `ANTHROPIC_API_KEY`, `COHERE_API_KEY`

## 시작하기
```bash
cp .env.example .env         # 키 채우기
docker compose up -d db      # pgvector Postgres 기동
pip install -e .             # 의존성 설치(가상환경 권장)

card-rag init-db             # 확장 + 테이블 생성

# 카드 1장 인제스천 (약관 원문을 data/raw/<card_id>/ 에 먼저 넣어둔다)
card-rag ingest shinhan-deep-dream   # normalize + extract (검수용 JSON 생성 후 정지)
#   → data/clauses/shinhan-deep-dream.json 의 숫자 필드 검수
card-rag load shinhan-deep-dream     # 임베딩 + DB 적재
```

## 파이프라인 (인제스천)
```
[1] collect   data/raw/{card_id}/  (수동, 3~4장 규모)
[2] normalize HTML/PDF → 텍스트          card-rag normalize
[3] extract   약관 → 혜택절 JSON 초안     card-rag extract   (Claude Haiku, tool-use)
[4] review    숫자 필드 사람 검수 ★
[5] embed     포함/제외 조건 → 벡터        (load에 포함, Cohere)
[6] load      benefit_clauses 적재         card-rag load
```

## 로드맵
- [x] 프로젝트 스캐폴드 + 인제스천 파이프라인 뼈대
- [ ] 프로토타입 카드 3~4장 원문 수집 → 추출·검수·적재로 파이프라인 검증
- [ ] 검색(retrieval) + grade('애매함' 판정) 설계 — 3라운드
- [ ] LangGraph 그래프 + 가드레일
- [ ] 규칙 엔진
- [ ] BE↔AI 인터페이스 계약
