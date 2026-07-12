# 인제스천 실행 런북 (프로토타입 3장)

> ✅ **현재: 전체 파이프라인 실제 가동 완료.** 3장 모두 extract→검증→load 되어 pgvector에 적재됨(혜택절 25개).
> 환경(64비트 Python 3.14 venv·Docker·키)·자동화(사람 검수 제거 → 자동 검증) 반영됨. 전체 개요는 [00_전체정리.md](00_전체정리.md).
> 아래는 **새 카드를 추가**할 때의 순서(참고). 검수 체크리스트/미해결 이슈는 히스토리로 남겨둠.

## 대상 카드
| card_id | 카드 | 특징(엔진 검증 포인트) |
|---|---|---|
| `tossbank-check` | 토스뱅크 카드 | 정액 캐시백, 전월실적 없음, 영역별 포함 브랜드 |
| `hana-narasarang-check` | 하나 나라사랑카드(체크) | % 할인, 실적 없음, 통합한도, 브랜드 조건 |
| `kpass-hana-check` | K-패스 하나 체크카드 | **전월실적 구간(30/60만)**, 포함/제외 명확 |

## 사전 준비물
- Python 3.11+ (현재 로컬 3.9 → 업그레이드)
- Docker Desktop (pgvector Postgres)
- `ANTHROPIC_API_KEY`, `COHERE_API_KEY`

## 실행 순서
```bash
cp .env.example .env            # 키 채우기
docker compose up -d db         # pgvector Postgres
python -m venv .venv && . .venv/Scripts/activate   # (Windows: .venv\Scripts\activate)
pip install -e .

card-rag init-db

# step 2 normalize는 이미 실행됨(data/normalized/*.md 존재). 재실행도 무방.
# step 3 extract: 약관 → 혜택절 JSON 초안
card-rag extract tossbank-check
card-rag extract hana-narasarang-check
card-rag extract kpass-hana-check

# step 4 검수 ★ : data/clauses/<id>.json 의 숫자 필드를 공식 약관과 대조해 확정
#   - 아래 '검수 체크리스트' 참고

# step 5+6 embed + load
card-rag load tossbank-check
card-rag load hana-narasarang-check
card-rag load kpass-hana-check
```

## 검수 체크리스트 (step 4)
- [ ] **토스뱅크**: 정액 캐시백(500원/100원)이 스키마의 `rate`(%)와 안 맞음 → 아래 '미해결 이슈' 참고. 현행 '스위치' 버전 기준으로 확정.
- [ ] **나라사랑**: 통합한도 영역(스타벅스/패스트푸드/온라인 등) 할인율이 `?` 미확인 → 공식 '서비스 이용안내'에서 확정.
- [ ] **K-패스 하나**: 전월실적 30만/60만 대중교통 캐시백이 **별도 혜택절 2건**으로 분리됐는지 확인.
- [ ] 모든 카드: `confidence: "low"` 항목 우선 검수. `source_span`과 원문 대조.

## 미해결 이슈 → 4라운드/규칙 엔진에서 결정
- **정액 캐시백 vs 정률(%)**: `benefit_clauses.rate`는 백분율 전제인데 토스뱅크는 건당 정액.
  규칙 엔진의 기대혜택 계산이 달라짐. 스키마에 `value_type(percent|flat)` + `flat_amount` 추가 검토 필요.
- **건당/일 횟수 캡**(예: 매일 1회, 월 10회)은 현재 미모델링(월 금액 한도 `monthly_cap`만 있음).
