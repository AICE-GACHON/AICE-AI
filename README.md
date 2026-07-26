# AICE-AI — ML/AI 논문 리서치 어시스턴트 (AI 파트)

ML/AI 연구자를 위한 RAG 서비스의 **AI 파이프라인 파트**.
유사 논문 검색 → OpenReview 리뷰 히스토리 분석 → 게재 학회 경향 분석을 수행한다.

- 전체 기획: [ML_AI_논문_RAG_서비스_기획서.md](ML_AI_논문_RAG_서비스_기획서.md)
- AI 파트 상세 설계 (아키텍처·스키마·확정 결정): [AI_파트_설계서.md](AI_파트_설계서.md)

## 기술 스택 (확정)

SPECTER2 임베딩 · pgvector 하이브리드 검색(RRF) · LangGraph 고정 DAG · Claude API (Haiku 추출 / Sonnet 종합) · 데이터: OpenReview + Semantic Scholar + arXiv (ICLR/NeurIPS 최근 5년+)

## 시작하기

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # 열어서 자격 증명 입력

docker compose up -d          # pgvector DB 기동 (포트 5433, 스키마 자동 생성)
```

- OpenReview 계정 필수 (익명 API는 봇 검증에 막힘): https://openreview.net/signup
- Semantic Scholar API 키: https://www.semanticscholar.org/product/api#api-key

DB 접속: `postgresql://paper:paper@localhost:5433/paper_assistant`
(로컬 Postgres와 충돌을 피해 **5433 포트**를 쓴다. `DATABASE_URL`로 덮어쓸 수 있다.)

## 수집 대상 (실측)

ICLR 2020–2025 + NeurIPS 2021–2024 = **43,515편**, 리뷰 약 15만 건.
venue별 논문 수와 API 버전은 [AI_파트_설계서.md](AI_파트_설계서.md) §10.1 참고.

**주의**: OpenReview는 2023년 전후로 API가 갈린다 (구 venue는 v1, 신 venue는 v2).
리뷰 필드명도 연도마다 다르다 (`review` / `main_review` / `strengths`+`weaknesses` …).
이 차이는 `ingest/normalize.py`가 흡수하므로 상위 코드는 신경 쓸 필요 없다.

## 실행

```bash
python -m paper_assistant.ingest.run_pilot 20   # ICLR 2024 샘플 20편 + 리뷰 수집
python scripts/count_all.py                     # venue별 논문 수 집계
python scripts/verify_normalize.py              # 10개 venue 정규화 검증
python scripts/verify_embedding.py              # SPECTER2 차원·품질·속도 검증
python scripts/load_pilot.py 200                # 수집→임베딩→적재→검색 end-to-end
python scripts/build_base_rates.py              # 코퍼스 aspect base rate 계산 (수집 후 1회)
pytest tests/                                   # 회귀 테스트 (DB 없으면 통합 테스트만 skip)
```

`scripts/` 실행 시 `PYTHONPATH`에 저장소 루트가 필요하다 (Windows: `$env:PYTHONPATH="."`).

## 임베딩

SPECTER2(`allenai/specter2_base` + proximity adapter), **768차원**, 논문 1편 = 벡터 1개.
**CPU로 충분하다** — 전체 43,515편이 약 0.8시간, GPU 불필요.

```python
from paper_assistant.embedding.specter2 import Specter2Embedder, similarity_percentile

embedder = Specter2Embedder()
vecs = embedder.encode([(title, abstract), ...])   # L2 정규화된 (N, 768)
```

⚠️ **원시 코사인 값을 사용자에게 노출하지 말 것.** SPECTER2는 유사도가 0.72~0.98에
압축되어 있어 **무관한 논문쌍도 0.845가 나온다**. `similarity_percentile()`로 백분위로
변환해서 전달한다. 자세한 측정치는 [AI_파트_설계서.md](AI_파트_설계서.md) §11.2 참고.

## 검색

SPECTER2 벡터 + Postgres full-text를 **RRF**(순위 기반)로 결합한다.
유사도 절대값이 못 쓸 물건이라(§11.2) 순위 기반 결합이 필수다.

```python
from paper_assistant.retrieval.hybrid_search import hybrid_search

results = hybrid_search(query_vector, f"{title} {abstract}", top_k=20)
```

⚠️ full-text 쿼리는 **OR 결합**해야 한다. `plainto_tsquery`는 모든 단어를 AND로
묶어서 긴 초록을 넣으면 거의 매칭되지 않는다 (실측 200편 중 1편). 자세한 내용은
[AI_파트_설계서.md](AI_파트_설계서.md) §5.

## 분석 파이프라인 (LangGraph)

공개 진입점은 함수 하나 — **백엔드 통합 계약**:

```python
from paper_assistant import analyze
report = analyze(title, abstract)          # -> Report (Pydantic)
```

고정 DAG: `input → retrieval → (유사성 태깅 ‖ 리뷰 분석 ‖ 게재 경향) → 종합`.
검색 이후 3개 분석이 병렬. supervisor 없음.

**예산 안전장치**: 기본은 LLM off(`$0`) — 태깅·종합이 스텁으로 동작해 배선을
검증할 수 있다. 데모 때만 실제 Claude 호출:

```bash
python scripts/verify_pipeline.py           # $0, 스텁
PAPER_ASSISTANT_USE_LLM=1 python ...        # Haiku 태깅 + Sonnet 종합
```

리뷰 패턴은 **키워드 aspect 집계**로 만든다. SPECTER2가 짧은 리뷰 문장 클러스터링에
부적합해서다 — [설계서 §14](AI_파트_설계서.md) 참고.

⚠️ **빈도로 줄세우지 말 것.** 코퍼스의 78.8%가 baselines 지적을 받으므로
"20편 중 17편 baselines"는 정보량이 0이다. 두 가지를 함께 낸다 ([§18](AI_파트_설계서.md)):

- **lift** = 관측률 ÷ 코퍼스 base rate, + 이항검정 p값 → `is_distinctive`
- **당락 대조** = 이 지적을 받은 이웃 vs 아닌 이웃의 accept율, + Fisher 정확검정
  → `is_contrast_significant`. **False면 표본 부족이므로 단정 금지** (n=4에서
  "0% 통과"는 노이즈다).

base rate는 사전 계산해 `aspect_base_rates`에 넣어둔다 — 수집 후 1회 실행:

```bash
python scripts/build_base_rates.py
```

이 테이블이 비어 있으면 lift 없이 예전처럼 빈도순으로 폴백한다(경고 로그).

## 구조

```
paper_assistant/
├── __init__.py                # 공개 API: analyze()
├── config.py                  # .env 로드, DATABASE_URL
├── schemas.py                 # Report 등 Pydantic (백엔드 계약)
├── ingest/
│   ├── openreview_client.py   # v1/v2 분기 + 토큰 캐시 + 페이지네이션 + 백오프
│   ├── normalize.py           # venue×연도별 필드 차이 → 단일 스키마
│   ├── review_extractor.py    # 휴리스틱 지적항목 추출($0) / Haiku(플레이스홀더)
│   ├── submission_linker.py   # 재투고 매칭 (arXiv→제목→제목+저자)
│   ├── run_pilot.py           # ICLR 2024 파일럿
│   └── run_ingest.py          # 전체 43k 수집·적재 (멱등 재개)
├── embedding/
│   └── specter2.py            # SPECTER2 임베딩 + 코사인→백분위 변환
├── db/
│   ├── connection.py          # 커넥션 풀
│   └── load.py                # upsert 적재 + review_points + 체크포인트
├── retrieval/
│   └── hybrid_search.py       # 벡터 + full-text, RRF 결합
└── graph/                     # LangGraph 파이프라인
    ├── state.py               # 공유 상태 (TypedDict)
    ├── llm.py                 # Claude 래퍼 (토글 가능)
    ├── base_rates.py          # 코퍼스 aspect base rate 조회 (lift 분모)
    ├── clustering.py          # aspect 집계 + lift/Fisher 검정 + 범용 클러스터 유틸
    ├── nodes.py               # 6개 노드
    └── pipeline.py            # DAG 조립 + analyze()
scripts/                       # 조사·검증용 + init_db.sql / build_indexes.sql
                               #   + build_base_rates.py (lift 분모 사전 계산)
tests/                         # 회귀 테스트 72건
```

**전체 적재를 마친 뒤** 벡터 인덱스를 생성할 것 (빈 테이블에 미리 만들면 적재가 느려진다):

```bash
docker exec -i paper-assistant-db psql -U paper -d paper_assistant < scripts/build_indexes.sql
```

`demo/`는 팀 시연용 임시 웹 화면 (독립 폴더, 나중에 삭제 가능) — [demo/README.md](demo/README.md).

전체 목표 구조와 로드맵은 [AI_파트_설계서.md](AI_파트_설계서.md) §7–8 참고.

## 데모 웹 (팀 시연용)

```bash
pip install -r demo/requirements.txt
python -m uvicorn demo.server:app --port 8000     # http://localhost:8000
```

제목+초록 입력 또는 PDF 업로드 → 유사 논문·리뷰 패턴·게재 경향·재투고 흐름.
`paper_assistant.analyze()` 하나만 호출하므로 **백엔드 통합 계약을 그대로 시연**한다.
실제 프론트가 준비되면 `demo/` 폴더를 통째로 삭제하면 된다.

## 백엔드 통합 계약

AI 파트는 Python 패키지로 제공되며, 공개 API는 다음 하나로 고정:

```python
from paper_assistant import analyze
report = analyze(title, abstract, pdf_bytes=None)  # -> Report (Pydantic)
```
