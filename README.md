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

## 파일럿 수집 (1주차)

```bash
python -m paper_assistant.ingest.run_pilot 20   # ICLR 2024 샘플 20편 + 리뷰
```

결과는 `data/raw/pilot_iclr2024/sample.json`에 저장되고, 리뷰/decision 필드 구조 분석이 출력된다.

## 구조

```
paper_assistant/
├── config.py              # .env 로드
└── ingest/
    ├── openreview_client.py   # 인증 + 페이지네이션 + 백오프
    └── run_pilot.py           # ICLR 2024 파일럿 수집
```

전체 목표 구조와 로드맵은 [AI_파트_설계서.md](AI_파트_설계서.md) §7–8 참고.

## 백엔드 통합 계약 (예정)

AI 파트는 Python 패키지로 제공되며, 공개 API는 다음 하나로 고정:

```python
paper_assistant.analyze(title, abstract, pdf_bytes=None) -> Report  # Pydantic 모델
```
