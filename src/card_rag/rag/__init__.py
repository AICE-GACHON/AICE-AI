"""RAG 파이프라인(LangGraph). 3라운드에서 설계·구현 예정.

계획된 그래프: retrieve → grade → reason → generate → guardrail
- grade: '자격이 애매한 후보'만 LLM으로 보내는 라우팅(판정 기준 미확정 → 같이 설계).
- guardrail: 규칙 엔진 숫자와의 정합성 + 스키마 검증, 위반 시 규칙 값 폴백('참고용').
"""
