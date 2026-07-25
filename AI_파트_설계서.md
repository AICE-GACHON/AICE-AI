# AI 파트 상세 설계서 — ML/AI 논문 리서치 어시스턴트

> 기획서(`ML_AI_논문_RAG_서비스_기획서.md`)의 AI 파트 구체화 문서.
> 프론트/백엔드는 다른 팀원 담당. AI 파트는 **Python 패키지**로 개발해 나중에 Python 백엔드에서 import하여 통합한다.

---

## 1. 확정된 기술 결정 사항

| 항목 | 결정 | 비고 |
|---|---|---|
| 벡터 DB | **pgvector** (Postgres) | 벡터 + 메타데이터 + 인용 엣지를 Postgres 하나로 통합 |
| 그래프 DB | **사용 안 함** | 인용 관계는 엣지 테이블로 충분. 다중 홉 순회 기능 없음 |
| 임베딩 모델 | **SPECTER2** (allenai, HuggingFace) | 학술 논문 특화. 논문 1편 = title+abstract → 벡터 1개 |
| 오케스트레이션 | **LangGraph** | 고정 DAG + 병렬 노드. LLM supervisor 라우팅 없음 |
| LLM | **Claude API** (예산 제약, §13) | 리뷰 추출은 **$0 휴리스틱 우선**, 쿼리 시점만 Claude |
| 검색 방식 | **하이브리드** | SPECTER2 벡터 + Postgres full-text, RRF로 결합 |
| 데이터 범위 | ICLR + NeurIPS **최근 5년+** (2020~) | **실측 43,515편** / 리뷰 약 15만 건 (§10) |
| 사용자 입력 | 텍스트(제목+초록) + **PDF draft 업로드** | PDF에서 제목/초록 추출 후 동일 파이프라인 |
| 리뷰 지적 항목 추출 | **오프라인 배치** (수집 시) | 쿼리 시점엔 클러스터링+집계만 |
| 유사성 근거 태깅 | **MVP 포함** | 상위 10~20편에 대해 쿼리 시점 LLM 태깅 |
| 재투고 흐름 추적 | **제대로 구현** | arXiv ID + 제목 유사도 + 저자 매칭 |
| 제공 형태 | **Python 패키지** | 백엔드도 Python. FastAPI 데모 서버는 개발용으로만 |
| 패키지 관리 | **pip + requirements.txt** | 팀 통일 |

---

## 2. 전체 아키텍처

### 2.1 두 개의 독립된 파이프라인

```
[A] 수집/인덱싱 파이프라인 (오프라인 배치, 주기적 실행)
    OpenReview API ──┐
    Semantic Scholar ─┼─→ 정규화 → LLM 리뷰 구조화 → 임베딩 → Postgres 적재
    arXiv API ────────┘

[B] 쿼리 파이프라인 (LangGraph, 사용자 요청마다 실행)
    사용자 입력 → 검색 → 병렬 분석 → 종합 리포트
```

### 2.2 쿼리 파이프라인 (LangGraph DAG)

Supervisor 패턴 대신 **고정 DAG**. 워크플로우가 매번 동일하므로 LLM 라우팅은
불필요한 비용/지연/불확실성만 추가한다. LLM은 지능이 필요한 노드 안에서만 사용.

```
        ┌─────────────────────────────┐
        │  input_node                 │  텍스트 or PDF → 제목/초록 정규화
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │  retrieval_node             │  하이브리드 검색 (벡터+FTS, RRF)
        │                             │  → 유사 논문 상위 K편 (기본 20)
        └──────┬──────────────┬───────┘
     ┌─────────▼────┐  ┌──────▼─────────┐     ← 이 3개만 병렬
     │ similarity_  │  │ review_        │  ┌────────────────┐
     │ tagging_node │  │ analysis_node  │  │ venue_trend_   │
     │ (LLM 태깅)   │  │ (클러스터링)    │  │ node (SQL 집계) │
     └─────────┬────┘  └──────┬─────────┘  └──────┬─────────┘
        ┌──────▼──────────────▼────────────────────▼───────┐
        │  synthesis_node (Sonnet)                          │
        │  → 최종 구조화 리포트 (JSON + 마크다운 요약)        │
        └───────────────────────────────────────────────────┘
```

**노드별 역할**

| 노드 | LLM 사용 | 내용 |
|---|---|---|
| `input_node` | PDF일 때만 (Haiku) | PDF → PyMuPDF로 텍스트 추출 → Haiku로 제목/초록 식별. 텍스트 입력이면 통과 |
| `retrieval_node` | 없음 | SPECTER2 임베딩 → pgvector 코사인 검색 + Postgres `tsvector` full-text 검색 → RRF(Reciprocal Rank Fusion) 결합 → 상위 K편 |
| `similarity_tagging_node` | Haiku | 상위 10~20편 각각에 대해 "왜 유사한가" 태깅: `methodology` / `dataset` / `problem_setting` / `citation` + 한 줄 근거. 논문당 1콜, 병렬 호출 |
| `review_analysis_node` | 없음 (임베딩만) | 유사 논문들의 **사전 추출된 지적 항목**을 DB에서 로드 → 임베딩 기반 클러스터링(HDBSCAN 또는 agglomerative) → "10편 중 6편이 실험 규모 지적" 형태로 집계 |
| `venue_trend_node` | 없음 | SQL 집계: 유사 논문들의 최종 decision 분포, 학회별 accept 비율, 재투고 흐름(예: ICLR reject → NeurIPS accept N건) |
| `synthesis_node` | Sonnet | 세 분석 결과를 받아 사람이 읽는 종합 리포트 생성. 구조화 JSON도 함께 반환 (프론트가 컴포넌트별 렌더링 가능하도록) |

### 2.3 수집/인덱싱 파이프라인 (배치)

5년치(5만+ 편)이므로 **재시작 가능(체크포인트)** 설계가 필수.

```
1. fetch_openreview   : venue×연도 단위로 논문+리뷰+메타리뷰+rebuttal+decision 수집
2. fetch_s2           : Semantic Scholar에서 메타데이터, 인용 관계, 저자 ID, venue 보강
3. fetch_arxiv        : arXiv ID 매칭, 초록/카테고리 보강
4. extract_review_points : 리뷰 → Haiku → 구조화된 지적 항목 리스트 (아래 §4)
5. link_submissions   : 재투고 흐름 매칭 (아래 §6)
6. embed              : SPECTER2로 논문/지적항목 임베딩
7. load               : Postgres 적재 (upsert, 단계별 상태 컬럼으로 체크포인트)
```

- 각 단계는 독립 실행 가능한 스크립트 + `ingest_status` 테이블로 진행 상태 추적
- OpenReview API v2는 rate limit 존재 → 지수 백오프 + venue×연도 단위 체크포인트
- LLM 비용: 리뷰 지적 항목 추출이 대부분. 리뷰 ~20만 건 × Haiku ≈ 감당 가능한 수준이지만, **1개 venue×연도로 먼저 파일럿 실행해서 편당 비용 측정 후 전체 실행**

---

## 3. 데이터 스키마 (Postgres + pgvector)

```sql
-- 논문 (검색의 기본 단위)
CREATE TABLE papers (
    id              BIGSERIAL PRIMARY KEY,
    openreview_id   TEXT UNIQUE,
    arxiv_id        TEXT,
    s2_paper_id     TEXT,
    title           TEXT NOT NULL,
    abstract        TEXT,
    venue           TEXT,          -- 'ICLR', 'NeurIPS'
    year            INT,
    decision        TEXT,          -- 'accept-oral', 'accept-poster', 'reject', 'withdrawn'
    final_venue     TEXT,          -- 최종 게재처 (재투고 추적 결과)
    embedding       vector(768),   -- SPECTER2
    tsv             tsvector GENERATED ALWAYS AS
                      (to_tsvector('english', title || ' ' || coalesce(abstract,''))) STORED
);
CREATE INDEX ON papers USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON papers USING gin (tsv);

-- 저자 (재투고 매칭용)
CREATE TABLE authors (
    id           BIGSERIAL PRIMARY KEY,
    s2_author_id TEXT UNIQUE,
    name         TEXT
);
CREATE TABLE paper_authors (
    paper_id  BIGINT REFERENCES papers(id),
    author_id BIGINT REFERENCES authors(id),
    position  INT,
    PRIMARY KEY (paper_id, author_id)
);

-- 리뷰 원문
CREATE TABLE reviews (
    id            BIGSERIAL PRIMARY KEY,
    paper_id      BIGINT REFERENCES papers(id),
    openreview_id TEXT UNIQUE,
    review_type   TEXT,   -- 'review', 'meta_review', 'rebuttal', 'decision'
    rating        TEXT,
    confidence    TEXT,
    content       JSONB   -- OpenReview 원본 필드 보존
);

-- 사전 추출된 리뷰 지적 항목 (클러스터링의 단위)
CREATE TABLE review_points (
    id         BIGSERIAL PRIMARY KEY,
    review_id  BIGINT REFERENCES reviews(id),
    paper_id   BIGINT REFERENCES papers(id),
    aspect     TEXT,        -- 통제된 분류 (§4)
    sentiment  TEXT,        -- 'weakness', 'strength', 'question'
    text       TEXT,        -- 지적 내용 요약 (1~2문장)
    embedding  vector(768)
);
CREATE INDEX ON review_points USING hnsw (embedding vector_cosine_ops);

-- 인용 엣지 (그래프 DB 대체)
CREATE TABLE citations (
    citing_paper_id BIGINT REFERENCES papers(id),
    cited_paper_id  BIGINT REFERENCES papers(id),
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

-- 재투고 연결 (같은 논문의 복수 투고 기록)
CREATE TABLE submission_links (
    earlier_paper_id BIGINT REFERENCES papers(id),
    later_paper_id   BIGINT REFERENCES papers(id),
    match_method     TEXT,      -- 'arxiv_id', 'title_exact', 'title_author_fuzzy'
    confidence       REAL,
    PRIMARY KEY (earlier_paper_id, later_paper_id)
);

-- 수집 체크포인트
CREATE TABLE ingest_status (
    venue TEXT, year INT, stage TEXT, status TEXT, updated_at TIMESTAMPTZ,
    PRIMARY KEY (venue, year, stage)
);
```

---

## 4. 청킹/임베딩 전략

**핵심: 검색 대상별로 단위가 다르다. 논문 유사도 검색에는 청킹이 없다.**

| 대상 | 단위 | 모델 | 이유 |
|---|---|---|---|
| 논문 유사도 검색 | **논문 1편 = 벡터 1개** (title + `[SEP]` + abstract) | SPECTER2 (proximity adapter) | SPECTER2가 정확히 이 용도로 학습됨. 본문 청킹은 노이즈만 추가 |
| 리뷰 지적 패턴 | **지적 항목 1개 = 벡터 1개** | SPECTER2 base (또는 동일 모델 통일) | 리뷰 전체 임베딩은 여러 주제가 섞여 클러스터링 품질 저하. LLM으로 항목 분리 후 임베딩 |
| 본문 full-text | **MVP 제외** | — | Phase 3 "예상 지적 예측" 때 섹션 단위로 추가 |

**리뷰 지적 항목 추출 (오프라인 배치, Haiku)**

리뷰 1건 → 아래 형태의 리스트로 구조화:

```json
[
  {"aspect": "experimental_scale", "sentiment": "weakness",
   "text": "Experiments limited to CIFAR-10/100; no ImageNet-scale validation."},
  {"aspect": "novelty", "sentiment": "weakness",
   "text": "Method is incremental over prior work X."}
]
```

`aspect`는 자유 생성이 아니라 **통제된 분류 체계**를 프롬프트에 명시 (클러스터링·집계 품질을 위해):
`novelty` / `experimental_scale` / `baselines` / `clarity` / `theoretical_soundness` / `reproducibility` / `related_work` / `significance` / `other`

쿼리 시점 클러스터링은 이 aspect 1차 그룹핑 + 임베딩 유사도 2차 병합으로 "유사 논문 10편 중 6편이 실험 규모 지적" 형태 집계 생성.

---

## 5. 하이브리드 검색 상세

```
score = RRF(vector_rank, fts_rank)   # 1/(60+rank) 합산, 표준 RRF
```

1. SPECTER2로 쿼리(제목+초록) 임베딩 → pgvector 코사인 top-50
2. Postgres `ts_rank` full-text top-50 (특정 데이터셋명·기법명 정확 매칭 보완)
3. RRF 결합 → 상위 K=20편 반환
4. 상위 20편만 similarity_tagging_node로 전달

전부 Postgres 쿼리 1~2개로 처리 가능. 별도 검색 엔진 불필요.

### ⚠️ full-text 쿼리는 반드시 OR 결합할 것 (실측으로 발견한 함정)

`plainto_tsquery`는 입력의 **모든 단어를 AND로 결합**한다. 사용자 입력이 초록
전체(수백 단어)이므로 그 단어를 전부 포함하는 문서만 걸린다 —
**실측: 200편 중 1편만 매칭되어 FTS가 사실상 무력화됐다.**

```sql
-- 잘못됨: 초록 전체를 넣으면 AND 조건이 되어 거의 매칭되지 않음
WHERE tsv @@ plainto_tsquery('english', :query)

-- 올바름: lexeme을 OR로 결합해 ts_rank가 겹치는 정도로 순위를 매김
WITH q AS (SELECT to_tsquery('english', string_agg(lexeme, ' | ')) AS query
           FROM unnest(to_tsvector('english', :query)))
SELECT p.id FROM papers p, q WHERE p.tsv @@ q.query
ORDER BY ts_rank(p.tsv, q.query) DESC
```

수정 후 같은 조건에서 200/200편 매칭되며, 벡터 검색에서 9~10위였던 논문이
키워드 매칭 덕에 3~4위로 올라오는 **하이브리드 본래의 동작**이 확인됐다.
회귀 방지 테스트: `tests/test_db_integration.py::test_fulltext_or_matching_beats_and_matching`

---

## 6. 재투고 흐름 매칭 (제대로 구현)

우선순위 폴백 체인:

1. **arXiv ID 일치** — 다른 venue 투고 기록이 같은 arXiv ID를 가리키면 확정 (confidence 1.0)
2. **제목 정확 일치** (정규화 후: 소문자, 공백/특수문자 정리) — confidence 0.95
3. **제목 유사 + 저자 겹침** — 제목 임베딩(또는 문자열 유사도 ≥ 임계값) AND 저자 집합 Jaccard ≥ 0.5 → confidence 산출, 임계값 이하 폐기

결과는 `submission_links`에 저장하고 venue_trend_node에서
"이 유형 논문의 재투고 흐름: ICLR'24 reject → NeurIPS'24 accept 12건" 형태로 집계.
저자 매칭을 위해 Semantic Scholar **author ID를 수집 단계에서 반드시 저장**.

---

## 7. 패키지 구조

```
paper_assistant/               # pip install -e . 로 백엔드에서 import
├── requirements.txt
├── setup.py (or pyproject.toml)
├── paper_assistant/
│   ├── __init__.py            # 공개 API: analyze(query) -> Report
│   ├── config.py              # DB URL, API 키 (환경변수)
│   ├── ingest/                # 파이프라인 A (배치)
│   │   ├── openreview_client.py
│   │   ├── s2_client.py
│   │   ├── arxiv_client.py
│   │   ├── review_extractor.py    # Haiku 지적항목 추출
│   │   ├── submission_linker.py   # 재투고 매칭
│   │   └── run_ingest.py          # CLI 엔트리포인트 (체크포인트 재개)
│   ├── embedding/
│   │   └── specter2.py
│   ├── retrieval/
│   │   └── hybrid_search.py       # pgvector + FTS + RRF
│   ├── graph/                 # 파이프라인 B (LangGraph)
│   │   ├── state.py               # TypedDict 상태 정의
│   │   ├── nodes.py               # 6개 노드
│   │   └── pipeline.py            # DAG 조립
│   ├── pdf/
│   │   └── extract.py             # PyMuPDF + Haiku 제목/초록 추출
│   └── schemas.py             # Pydantic: Report, SimilarPaper, ReviewPattern, VenueTrend
├── scripts/
│   └── init_db.sql
├── demo_server/               # 개발용 FastAPI (통합 전 데모/테스트)
│   └── main.py
└── tests/
```

**백엔드 통합 계약(contract)**: 공개 API는 단 하나 —
`paper_assistant.analyze(title, abstract, pdf_bytes=None) -> Report` (Pydantic 모델).
백엔드 팀은 이 함수 시그니처와 `Report` 스키마만 알면 됨. 스트리밍이 필요해지면
LangGraph의 `astream()`을 그대로 노출하는 `analyze_stream()` 추가.

---

## 8. 구현 순서 (AI 파트 로드맵)

1. ~~**주차 1 — 데이터 파일럿**~~ **(진행 중)**: OpenReview API 탐색 ✅, 정규화 레이어 + 10개 venue 검증 ✅ → 남은 것: 리뷰 추출 프롬프트 튜닝 + 편당 LLM 비용 측정 (Anthropic 키 필요)
2. ~~**주차 2 — 검색 코어**~~ **✅ 완료** (§11, §12): SPECTER2 임베딩 + pgvector 적재 + 하이브리드 검색, 200편 end-to-end 검증 통과
3. ~~**주차 3 — LangGraph 파이프라인**~~ **✅ 완료** (§14): 6개 노드 조립, 병렬 분석, $0 배선 검증. 남은 것: 재투고 매칭·PDF·Haiku 태깅 실측
4. **주차 4 — 전체 수집** *(진행 중)*: 43k 배치 실행(백그라운드, ~9h), ~~재투고 매칭~~ ✅(§15)
5. **주차 5 — 마감**: PDF 입력, demo_server, Report 스키마 문서화 → 백엔드 팀 전달

---

## 9. 리스크 (AI 파트 한정)

- ~~**SPECTER2 차원 확인**~~ → **해소됨**. 실측 768차원 확정 (§11). 스키마의 `vector(768)` 그대로 사용
- ~~**OpenReview 스키마 변동**~~ → **해소됨**. 실측 결과 §10 참고. 정규화 레이어(`ingest/normalize.py`)로 흡수 완료, 10개 venue 검증 통과
- **리뷰 추출 품질**: aspect 분류가 흔들리면 클러스터링 전체가 흔들림 → 파일럿 단계에서 수동 라벨 50건과 비교 검증
- **Claude API 비용**: 수집 단계가 지배적. 파일럿에서 실측 후 전체 실행 여부 판단

---

## 10. OpenReview API 실측 결과 (2026-07-21 조사)

기획 단계의 추정이 아니라 **실제 API를 호출해 확인한 사실**. 수집 파이프라인 구현의 근거.

### 10.1 API 버전이 두 개로 갈린다

2023년 전후로 OpenReview가 API를 교체했고, **구 venue는 v2에서 조회되지 않는다.**

| venue | API | submission invitation | 논문 수 |
|---|---|---|---|
| ICLR 2020 | v1 | `-/Blind_Submission` | 2,213 |
| ICLR 2021 | v1 | `-/Blind_Submission` | 2,594 |
| ICLR 2022 | v1 | `-/Blind_Submission` | 2,617 |
| ICLR 2023 | v1 | `-/Blind_Submission` | 3,792 |
| ICLR 2024 | v2 | `-/Submission` | 7,404 |
| ICLR 2025 | v2 | `-/Submission` | 11,672 |
| NeurIPS 2021 | v1 | `-/Blind_Submission` | 2,768 |
| NeurIPS 2022 | v1 | `-/Blind_Submission` | 2,824 |
| NeurIPS 2023 | v2 | `-/Submission` | 3,395 |
| NeurIPS 2024 | v2 | `-/Submission` | 4,236 |
| **합계** | | | **43,515** |

리뷰 추정 약 **15만 건** (편당 3.5건). v1은 `api.openreview.net`, v2는 `api2.openreview.net`.
v2는 모든 content 필드를 `{"value": x}`로 감싸지만 v1은 raw 값 — 정규화 레이어에서 흡수.

### 10.2 인증·요청 관련 함정 (전부 실측으로 확인)

- **익명 `/notes` 요청은 403** (`ChallengeRequiredError`, 봇 검증) → **로그인 필수**. v1/v2 모두 동일
- **`/login` 자체에 rate limit** 존재 → 토큰을 디스크에 캐시해 재사용 (`data/.token_*.json`, JWT `exp` 검사)
- **v2는 `limit=1`이면 캐시 응답을 주고 `count` 필드를 생략** → 총 개수를 알려면 `limit>=3` + `offset` 명시
- 공식 `openreview-py` 라이브러리는 의존성(`editdistance`)이 **Python 3.14 휠 미제공**으로 설치 실패 → raw REST로 직접 구현 (의존성도 가볍고 체크포인트 제어도 쉬움)

### 10.3 리뷰 필드가 venue×연도마다 전부 다르다

**이것이 최대 함정이었다.** 같은 ICLR인데도 연도마다 필드명·점수 형식이 바뀐다.

| venue | 리뷰 본문 필드 | 점수 필드 | 강점/약점 분리 |
|---|---|---|---|
| ICLR 2020 | `review` | `rating` | ❌ 통짜 |
| ICLR 2021 | `review` | `rating` | ❌ 통짜 |
| ICLR 2022 | `main_review` | `recommendation` | ❌ 통짜 |
| ICLR 2023 | `strength_and_weaknesses` | `recommendation` | △ 합쳐짐 |
| ICLR 2024/2025 | `strengths` + `weaknesses` | `rating` | ✅ 분리 |
| NeurIPS 2021 | `main_review` | `rating` | ❌ 통짜 |
| NeurIPS 2022 | `strengths_and_weaknesses` | `rating` | △ 합쳐짐 |
| NeurIPS 2023/2024 | `strengths` + `weaknesses` | `rating` | ✅ 분리 |

**점수 형식도 제각각**: `"8: Accept"`, `"5"`, `"3: reject, not good enough"`, `"2 fair"` → 선두 숫자 파싱으로 통일.

**설계에 미치는 영향**: 2024년 이후 venue는 `weaknesses` 필드만 LLM에 넘기면 되지만,
그 이전은 리뷰 본문 전체를 넘겨 강점/약점부터 분리해야 한다. `NormalizedReview.needs_llm_split`
플래그로 구분하고 `llm_input` 프로퍼티가 최소 토큰만 반환하도록 설계 → **LLM 비용 절감**.

### 10.4 메타리뷰 위치도 다르다

`Meta_Review` 노트가 **존재하는 venue는 ICLR 2024, NeurIPS 2022뿐**.
나머지는 `Decision` 노트의 `comment` 필드(ICLR 2023은 `metareview:_summary,_strengths_and_weaknesses`)에 들어있다.
→ 정규화 레이어에서 Meta_Review 노트 우선, 없으면 Decision 노트로 폴백.

### 10.5 decision 판별

`venue` 문자열(`"ICLR 2024 poster"`, `"Submitted to ICLR 2024"`)로 판별하는 게 1순위지만,
**ICLR 2020/2021은 submission content에 `venue` 필드 자체가 없다** → `Decision` 노트 값으로 폴백.
정규화 결과: `accept-oral` / `accept-spotlight` / `accept-poster` / `accept-notable` / `accept` /
`reject` / `withdrawn` / `desk-reject` / `unknown`.

### 10.6 검증 상태

`scripts/verify_normalize.py`로 **10개 venue × 3편**을 실제 API에서 받아 정규화 검증 →
title/abstract/decision/rating/리뷰본문/author_ids 전 항목 정상, 문제 0건.
단위 테스트 9건(`tests/test_normalize.py`)이 각 연도 형식을 회귀 방지용으로 고정.

---

## 11. SPECTER2 임베딩 실측 결과 (2026-07-21)

### 11.1 환경·성능

| 항목 | 실측값 |
|---|---|
| 임베딩 차원 | **768** (스키마 `vector(768)` 확정) |
| 처리 속도 | 편당 **69ms** (CPU) |
| 전체 43,515편 예상 | **약 0.8시간** (CPU) |
| **GPU 필요 여부** | **불필요** — CPU로 1시간 내 완료 |
| Python 3.14 호환 | torch 2.13 / transformers 4.57 / adapters 1.3 **정상 설치** |

모델 구성: `allenai/specter2_base` + proximity adapter.
입력 형식은 학습 시와 동일하게 `title + [SEP] + abstract`, 출력은 마지막 레이어 CLS 토큰을 L2 정규화.

### 11.2 유사도 스케일이 좁다 — 절대 임계값을 쓰면 안 된다

ICLR 2024 논문 **300편(무작위 쌍 89,700개)** 으로 측정한 코사인 유사도 분포:

| 통계 | 값 |
|---|---|
| 최소 / 평균 / 최대 | 0.721 / **0.845** / 0.978 |
| 표준편차 | 0.033 |
| 25 / 50 / 75 분위 | 0.823 / 0.845 / 0.867 |
| 95 / 99 / 99.9 분위 | 0.900 / 0.923 / 0.944 |

검증용 논문쌍의 위치:

| 쌍 | 코사인 | 백분위 |
|---|---|---|
| Transformer ↔ 단백질 구조 예측 (무관) | 0.844 | **55.8%** (중앙값) |
| Transformer ↔ BERT (관련) | 0.920 | 상위 1.3% |
| TabR ↔ Revisiting Tabular DL (관련) | 0.956 | 상위 0.1% |

**모델은 정상 작동한다** — 무관한 쌍을 정확히 중앙값에 놓았다.
문제는 **스케일**이다. 무작위 쌍조차 0.845가 나오므로:

1. **`0.85 이상 = 유사` 같은 절대 임계값은 무의미**하다. 검색은 반드시 **top-K 순위 기반**.
   → RRF 하이브리드(§5)가 순위 기반이라 이 특성과 잘 맞는다. 설계 선택이 검증된 셈.
2. **프론트에 원시 코사인 값을 "유사도 84%"로 노출하면 사용자가 반드시 오해한다.**
   → `similarity_percentile()`로 백분위 변환해서 전달한다 (측정된 분위수 기반 선형 보간).
   백엔드/프론트 팀에 넘길 `Report` 스키마에는 **백분위를 담고 원시 코사인은 담지 않는다.**

재측정이 필요하면 `scripts/measure_similarity_dist.py` 실행 (참조 분위수 갱신용).

---

## 12. DB 구축 및 검색 검증 결과 (2026-07-21)

### 12.1 구성

pgvector 공식 이미지(`pgvector/pgvector:pg17`)를 Docker로 기동. **포트 5433**을 쓴다
(로컬에 다른 Postgres가 있어도 충돌하지 않도록).

```bash
docker compose up -d      # 최초 기동 시 scripts/init_db.sql 자동 실행
```

테이블 8개: `papers` / `reviews` / `review_points` / `authors` / `paper_authors` /
`citations` / `submission_links` / `ingest_status`.
확장: `vector`(임베딩), `pg_trgm`(재투고 제목 매칭).

**벡터 인덱스(HNSW)는 스키마에 넣지 않았다.** 빈 테이블에 미리 만들면 적재 내내
인덱스 갱신 비용이 발생하므로, 전체 적재 후 `scripts/build_indexes.sql`로 생성한다.

### 12.2 End-to-end 검증 (ICLR 2024, 200편)

| 단계 | 실측 |
|---|---|
| 수집 + 정규화 | 200편 |
| 임베딩 (CPU) | 58초 |
| DB 적재 | 0.9초 (논문 200 + 저자 902명) |
| 하이브리드 검색 | **16ms** |

쿼리 논문 자신이 1위로 반환되고, 2~5위가 전부 같은 분야(그래프 신경망) 논문으로
채워지는 것을 확인. `tsvector`는 생성 컬럼이라 적재만 하면 자동으로 채워진다.

### 12.3 Python 3.14 관련 이슈

커넥션 풀을 명시적으로 닫지 않으면 인터프리터 종료 시
`PythonFinalizationError: cannot join thread at interpreter shutdown`이 발생한다
(3.14부터 종료 시점 스레드 join이 금지됨). `atexit.register(close_pool)`로 해결.

### 12.4 남은 성능 과제

현재 200편에서 16ms지만 **43,515편 전체에서는 HNSW 인덱스 없이 순차 스캔이 되어
느려진다.** 전체 적재 후 `build_indexes.sql`을 실행하고 재측정할 것.

---

## 13. LLM 비용 전략 (예산 $4.92 기준, 2026-07-22)

### 13.1 전체 추출은 예산 밖

리뷰 15만 건을 Haiku로 지적항목 추출하면 정가 **약 $200** (Batch 50%로도 $100).
가용 예산 $4.92로는 전체의 5%도 못 돌린다. 실측 기반 추정:

| 항목 | 리뷰당 토큰 | 15만 건 |
|---|---|---|
| 입력 (weakness ~350 + 프롬프트 ~250) | ~600 | 90M → $90 |
| 출력 (JSON ~150) | ~150 | 22M → $110 |

가격: Haiku 4.5 = 입력 $1 / 출력 $5 per 1M. Sonnet 5 = $2/$10 (2026-08-31까지 인트로).

### 13.2 결정: $0 휴리스틱 추출을 기본값으로

수집 단계 LLM 비용을 **$0으로** 만들고, 예산은 쿼리 시점 데모 콜(태깅·종합)에만 쓴다.

- **`HeuristicExtractor`** (`ingest/review_extractor.py`): 리뷰를 불릿/번호/문장 단위로
  분리하고 키워드로 aspect를 근사. LLM 불필요.
- **`HaikuExtractor`**: 동일 인터페이스(`PointExtractor`)의 플레이스홀더. 품질이
  부족하면 수집 스크립트에서 extractor만 교체하면 되고 다운스트림은 그대로.

### 13.3 실측 품질 (ICLR 2024, 100편 / 리뷰 384건)

| 지표 | 값 | 평가 |
|---|---|---|
| 리뷰당 지적항목 수 | 평균 **7.4개** | ✅ 분리 우수 |
| aspect `other` 비율 | **68%** | ⚠️ 키워드 분류는 약함 |

**핵심 판단**: `other` 68%는 병목이 아니다. "유사 논문 N편 중 M편이 비슷한 지적"
기능은 **aspect 라벨이 아니라 지적항목 텍스트의 임베딩 클러스터링**으로 생성되며,
클러스터는 대표 문장(medoid)으로 라벨링된다. 즉 aspect가 `other`여도 그 항목은
임베딩되어 정상적으로 클러스터링된다. aspect는 보조 필터일 뿐이다.

→ **$0 경로로 리뷰 패턴 분석 기능이 성립한다.** Haiku가 개선하는 것은 aspect 라벨의
정확도뿐이며(있으면 좋은 정도), 핵심 기능을 막지 않는다.

### 13.4 예산 사용 계획

1. 수집 + 임베딩: 전체 43,515편, **$0** (LLM 미사용)
2. 리뷰 지적항목: 전체 **$0** (휴리스틱)
3. 예산 $4.92: 쿼리 시점 태깅(Haiku) + 종합 리포트(Sonnet) 데모 콜 전용
   (쿼리당 ~$0.05 → 약 90~100회 데모 가능)

풀스케일 Haiku 추출($100)은 예산 확보 시 Batch로 하룻밤 실행하는 향후 과제로 남긴다.

---

## 14. LangGraph 파이프라인 구현 결과 (2026-07-22)

### 14.1 구조

설계 §2.2의 고정 DAG를 LangGraph로 구현. supervisor 없음.

```
input → retrieval → ┬ similarity_tagging (Haiku) ┐
                    ├ review_analysis   (no LLM) ┼→ synthesis (Sonnet) → END
                    └ venue_trend       (no LLM) ┘
```

검색 이후 3개 분석 노드가 **병렬**, synthesis는 fan-in. 200편 파일럿에서
end-to-end 정상 작동 확인.

### 14.2 예산 안전장치: LLM 토글

`get_llm(enabled=False)`(기본)면 태깅·종합 노드가 **결정론적 스텁**을 만든다.
→ 크레딧 0원으로 DAG 배선·스키마를 전부 검증 가능. 데모 때만
`PAPER_ASSISTANT_USE_LLM=1`로 실제 Claude(Haiku/Sonnet) 호출.

### 14.3 ⚠️ SPECTER2는 리뷰 문장 클러스터링에 부적합 (설계 변경)

설계 §4는 지적항목을 임베딩→클러스터링하려 했으나, **SPECTER2로 짧은 리뷰
문장을 임베딩하니 유사도가 평균 0.872에 압축**된다 (논문 title+abstract용
모델이라 짧은 문장을 변별 못 함). 실측: ICLR 2020 지적항목 400개, 무작위 쌍
유사도 50분위 0.873 / 95분위 0.925. **임계값 0.80에서도 17편이 한 클러스터로 뭉침** →
클러스터링 무의미.

**대응**: 임베딩 클러스터링 대신 **키워드 aspect 기반 집계**를 1차 방법으로 채택.
`review_analysis_node`는 지적항목을 aspect별로 묶어 "20편 중 12편이 명확성 지적,
12편 baselines, 10편 신규성 지적" 형태로 집계한다. 장점:

- **쿼리 시점 임베딩 불필요** → 더 빠르고 단순 (§13에서 리뷰 임베딩을 미룬 결정과 부합)
- 해석 가능하고 정직 (임베딩 클러스터의 애매한 medoid 라벨보다 명확)
- `HeuristicExtractor`의 aspect 68% other여도, 나머지 32%가 깔끔한 패턴을 만듦

`clustering.py._greedy_cluster`는 범용 유틸로 남겨둠 (향후 aspect 내부 세분화 등).

### 14.4 백엔드 통합 계약 확정

```python
from paper_assistant import analyze
report = analyze(title, abstract, pdf_bytes=None) -> Report   # Pydantic
```

`Report`(`schemas.py`)는 similar_papers / review_patterns / venue_trends /
resubmission_flows / summary_markdown로 구성. **원시 코사인은 담지 않고
similarity_percentile만** 담는다 (§11.2). JSON 직렬화 왕복 테스트 통과.

### 14.5 남은 것

- ~~**재투고 흐름**~~ **✅ 완료** (§15)
- **PDF 입력**: `pdf/extract.py` (PyMuPDF + Haiku)
- **similarity_tagging 실측**: 크레딧으로 Haiku 태깅 품질 확인 (아직 스텁만 검증)

---

## 15. 재투고 매칭 구현 (2026-07-22)

### 15.1 폴백 체인 (`ingest/submission_linker.py`)

| 순위 | 방법 | confidence | 상태 |
|---|---|---|---|
| 1 | arXiv ID 일치 | 1.00 | arxiv_id가 아직 전부 NULL(S2 보강 전) → **현재 no-op**, 채워지면 자동 활성화 |
| 2 | 정규화 제목 정확 일치 | 0.95 | ✅ 작동 |
| 3 | 제목 유사(pg_trgm ≥ 0.7) + 저자 Jaccard ≥ 0.5 | = 제목 유사도 | ✅ 작동 |

- 제목 정규화: 소문자 + 영숫자 외 제거 + 공백 정리
- 방향: `venue_sort_key`로 정렬 — 같은 해면 ICLR(상반기) < NeurIPS(하반기).
  → "ICLR 2024 reject → NeurIPS 2024 accept" 흐름이 올바르게 생성됨
- 전량 재계산(TRUNCATE + insert) 멱등. 전체 수집 완료 후 재실행하면 커버리지 상승
- 결과는 `venue_trend_node`가 유사 논문 집합에 대해 집계 → `Report.resubmission_flows`

### 15.2 검증

부분 데이터(ICLR 2020/2021)에서 실제 재투고 정확히 포착:
**"Towards Finding Longer Proofs" ICLR 2020 reject → ICLR 2021 reject** (title_exact, 0.95).
NeurIPS venue 적재 후 ICLR↔NeurIPS 흐름이 다수 잡힐 것으로 예상.

### 15.3 psycopg 함정 (실측)

- `set_limit(0.7)` → `set_limit(0.7::real)` 캐스트 필요 (double precision 거부)
- trgm `%` 연산자는 SQL에서 `%%`로 쓰되 **빈 파라미터 `()`를 넘겨야** psycopg가 축약

### 15.4 NUL 바이트 (수집 중 발견)

일부 논문 초록/리뷰에 NUL(0x00) 바이트 → Postgres text 컬럼이 거부(ICLR 2021에서 발생).
`normalize.clean_text()`가 제어 문자를 제거(탭·개행 보존)하도록 수정. 수집 재개.
