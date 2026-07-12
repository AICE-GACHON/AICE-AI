"""[3] 추출: 약관 원문 → 혜택절 JSON 초안 (Claude Haiku, tool-use로 스키마 강제).

핵심 원칙:
- 전월실적 구간이 여러 개면 각 구간을 **별도 혜택절**로 분리(min_spend로 구분).
- 숫자는 원문에 명시된 값만. 추정 금지, 애매하면 confidence=low.
- source_span에 근거 문장을 담아 사람 검수를 돕는다.
- system 프롬프트는 프롬프트 캐싱(cache_control)으로 재사용해 카드가 늘어도 저렴.
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic

from card_rag.config import settings
from card_rag.schemas.clause import ExtractedClause, ExtractionResult

NORMALIZED_DIR = Path("data/normalized")
CLAUSES_DIR = Path("data/clauses")

_CATEGORIES = ["카페", "음식점", "편의점", "대형마트", "온라인쇼핑",
               "대중교통", "주유", "통신", "영화문화", "병원약국", "해외", "기타"]

SYSTEM_PROMPT = (
    "너는 한국 신용카드 약관·상품설명서에서 '혜택절'을 구조화 추출하는 전문가다.\n"
    "규칙:\n"
    "1) 하나의 혜택은 (업종 × 혜택유형 × 전월실적구간 × 결제금액구간) 단위로 분리한다. "
    "전월실적 구간이나 결제금액 구간('1만원 이상/미만')별로 값이 다르면 각 구간을 별도 혜택절로 만든다.\n"
    "2) 정률(%) 혜택은 value_type='percent'로 rate에, 건당 정액 캐시백은 value_type='flat'으로 "
    "flat_amount(원)·flat_min_txn(적용 최소 결제액)에 담는다. "
    "숫자(rate/flat_amount/monthly_cap/min_spend)는 원문 명시값만. 추정 금지, 불명확하면 confidence=low.\n"
    "3) include_notes/exclude_notes에는 포함/제외 가맹점·조건을 원문 표현 그대로 담는다.\n"
    "4) source_span에는 판단 근거가 된 원문 문장을 그대로 인용한다.\n"
    f"5) category는 다음 중 하나로만 매핑한다: {', '.join(_CATEGORIES)}."
)

# tool-use 입력 스키마(= ExtractedClause 배열). 모델이 이 형태로만 응답하도록 강제.
_TOOL = {
    "name": "emit_clauses",
    "description": "약관에서 추출한 혜택절 목록을 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clauses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": _CATEGORIES},
                        "benefit_type": {"type": "string", "enum": ["적립", "청구할인"]},
                        "value_type": {"type": "string", "enum": ["percent", "flat"]},
                        "rate": {"type": "number"},
                        "flat_amount": {"type": ["integer", "null"]},
                        "flat_min_txn": {"type": "integer"},
                        "monthly_cap": {"type": ["integer", "null"]},
                        "min_spend": {"type": "integer"},
                        "include_notes": {"type": "string"},
                        "exclude_notes": {"type": "string"},
                        "source_span": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["category", "benefit_type", "value_type", "source_span"],
                },
            }
        },
        "required": ["clauses"],
    },
}


def extract_card(card_id: str) -> ExtractionResult:
    text = (NORMALIZED_DIR / f"{card_id}.md").read_text(encoding="utf-8")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "emit_clauses"},
        messages=[{"role": "user", "content": f"[카드 {card_id} 약관]\n\n{text}"}],
    )

    payload = next(b.input for b in resp.content if b.type == "tool_use")
    clauses = [ExtractedClause.model_validate(c) for c in payload["clauses"]]
    result = ExtractionResult(card_id=card_id, clauses=clauses)

    CLAUSES_DIR.mkdir(parents=True, exist_ok=True)
    out = CLAUSES_DIR / f"{card_id}.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
