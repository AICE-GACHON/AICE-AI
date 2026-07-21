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
pytest tests/                                   # 테스트 25건 (DB 없으면 통합 테스트만 skip)
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

## 구조

```
paper_assistant/
├── config.py                  # .env 로드, DATABASE_URL
├── ingest/
│   ├── openreview_client.py   # v1/v2 분기 + 토큰 캐시 + 페이지네이션 + 백오프
│   ├── normalize.py           # venue×연도별 필드 차이 → 단일 스키마
│   └── run_pilot.py           # ICLR 2024 파일럿 수집
├── embedding/
│   └── specter2.py            # SPECTER2 임베딩 + 코사인→백분위 변환
├── db/
│   ├── connection.py          # 커넥션 풀
│   └── load.py                # upsert 적재 + 수집 체크포인트
└── retrieval/
    └── hybrid_search.py       # 벡터 + full-text, RRF 결합
scripts/                       # 조사·검증용 + init_db.sql / build_indexes.sql
tests/                         # 회귀 테스트 25건
```

**전체 적재를 마친 뒤** 벡터 인덱스를 생성할 것 (빈 테이블에 미리 만들면 적재가 느려진다):

```bash
docker exec -i paper-assistant-db psql -U paper -d paper_assistant < scripts/build_indexes.sql
```

전체 목표 구조와 로드맵은 [AI_파트_설계서.md](AI_파트_설계서.md) §7–8 참고.

## 백엔드 통합 계약 (예정)

AI 파트는 Python 패키지로 제공되며, 공개 API는 다음 하나로 고정:

```python
paper_assistant.analyze(title, abstract, pdf_bytes=None) -> Report  # Pydantic 모델
```
