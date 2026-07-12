"""[3-정책] 약관 원문 → 정책 청크(전역 제외·통합한도·특약) 추출. Haiku, tool-use.

혜택절 추출(extract.py)과 분리된 2차 패스. 혜택률(rate)에 안 담기는 카드 단위 규칙만 뽑는다.
"""
from __future__ import annotations

from pathlib import Path

import anthropic

from card_rag.config import settings
from card_rag.schemas.clause import InternalCategory
from card_rag.schemas.policy import ExtractedPolicy, PolicyExtractionResult

try:  # InternalCategory는 Literal → 값 목록 추출
    from typing import get_args

    _CATEGORIES = list(get_args(InternalCategory))
except Exception:  # pragma: no cover
    _CATEGORIES = []

NORMALIZED_DIR = Path("data/normalized")
POLICIES_DIR = Path("data/policies")

SYSTEM_PROMPT = (
    "너는 카드 약관에서 '정책 규칙'을 뽑는 전문가다. 적립률/할인율 같은 개별 혜택 수치는 "
    "무시하고, 혜택 하나에 안 담기는 카드 단위 규칙만 추출한다:\n"
    "- global_exclude: 혜택에서 제외되는 결제(상품권·간편결제 충전·국세/지방세·공과금 등)\n"
    "- performance_exclude: 전월실적 산정에서 제외되는 결제\n"
    "- aggregate_cap: 회원 단위 통합 할인한도, 월 이용횟수 상한 등\n"
    "- special_term: 결제형태 제한(오프라인만·후불교통만), 중복적용 불가 등 특약\n"
    "특정 업종에만 적용되면 category에 업종코드를, 카드 전체면 null. "
    "text는 원문 표현을 살려 간결히, source_span에 근거 문장. 없으면 빈 배열."
)

_TOOL = {
    "name": "emit_policies",
    "description": "약관에서 추출한 정책 규칙 목록을 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "policies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "policy_type": {
                            "type": "string",
                            "enum": ["global_exclude", "performance_exclude", "aggregate_cap", "special_term"],
                        },
                        "category": {"type": ["string", "null"], "enum": _CATEGORIES + [None]},
                        "text": {"type": "string"},
                        "source_span": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["policy_type", "text", "source_span"],
                },
            }
        },
        "required": ["policies"],
    },
}


def extract_policies(card_id: str) -> PolicyExtractionResult:
    text = (NORMALIZED_DIR / f"{card_id}.md").read_text(encoding="utf-8")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "emit_policies"},
        messages=[{"role": "user", "content": f"[카드 {card_id} 약관]\n\n{text}"}],
    )
    payload = next(b.input for b in resp.content if b.type == "tool_use")
    policies = [ExtractedPolicy.model_validate(p) for p in payload["policies"]]
    result = PolicyExtractionResult(card_id=card_id, policies=policies)

    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    (POLICIES_DIR / f"{card_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
