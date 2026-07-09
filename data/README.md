# data/

인제스천 파이프라인의 파일 흐름. `raw`/`normalized`는 저작권·용량 문제로 git 제외.

```
raw/{card_id}/          # [1] 수집(수동): 카드사 상품안내 HTML, 상품설명서 PDF, txt
normalized/{card_id}.md # [2] normalize: 텍스트로 변환·병합 (자동)
clauses/{card_id}.json  # [3] extract → [4] 사람 검수 → 승인본(단일 진실원본, 선택 커밋)
cards.json              # 카드 메타(card_id → name/issuer/...). cards.example.json 참고해 작성
```

## 검수(Review) 규칙
`clauses/{card_id}.json`의 **숫자 필드(rate, monthly_cap, min_spend)** 는 규칙 엔진이
그대로 사용하므로, load 전에 반드시 사람이 원문(`source_span`)과 대조해 확정한다.
`confidence: "low"` 항목을 우선 검수한다.
