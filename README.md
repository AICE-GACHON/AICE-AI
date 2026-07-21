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
```

- OpenReview 계정 필수 (익명 API는 봇 검증에 막힘): https://openreview.net/signup
- Semantic Scholar API 키: https://www.semanticscholar.org/product/api#api-key

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
pytest tests/                                   # 단위 테스트
```

`scripts/` 실행 시 `PYTHONPATH`에 저장소 루트가 필요하다 (Windows: `$env:PYTHONPATH="."`).

## 구조

```
paper_assistant/
├── config.py                  # .env 로드
└── ingest/
    ├── openreview_client.py   # v1/v2 분기 + 토큰 캐시 + 페이지네이션 + 백오프
    ├── normalize.py           # venue×연도별 필드 차이 → 단일 스키마
    └── run_pilot.py           # ICLR 2024 파일럿 수집
scripts/                       # 조사·검증용 (패키지 아님)
tests/                         # 정규화 회귀 테스트
```

전체 목표 구조와 로드맵은 [AI_파트_설계서.md](AI_파트_설계서.md) §7–8 참고.

## 백엔드 통합 계약 (예정)

AI 파트는 Python 패키지로 제공되며, 공개 API는 다음 하나로 고정:

```python
paper_assistant.analyze(title, abstract, pdf_bytes=None) -> Report  # Pydantic 모델
```
