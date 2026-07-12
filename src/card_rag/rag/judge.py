"""④ LLM 자격 판정 — 가맹점이 혜택절의 포함/제외 조건에 걸리는지 판정.

가드레일: LLM은 자격(eligible)과 근거(reason)만 낸다. 금액/율 같은 숫자는 절대 안 만진다.
스키마 위반/오류 시 '관대(eligible=True) + 저신뢰 + 주의' 로 폴백한다.
"""
from __future__ import annotations

import anthropic

from card_rag.config import settings
from card_rag.rag.candidates import CandidateClause
from card_rag.rag.types import Judgment

_SYS = (
    "너는 카드 혜택 자격 판정기다. 주어진 가맹점이 이 혜택의 대상인지만 판정한다.\n"
    "규칙:\n"
    "- 포함 목록(화이트리스트)이 있으면, 가맹점이 그 목록/유형에 속할 때만 대상(eligible=true).\n"
    "- 제외 목록에 걸리면 미대상(eligible=false).\n"
    "- 포함 목록이 없고 제외에도 안 걸리면 대상(관대).\n"
    "- 애매하면 confidence를 낮춘다. 금액·율은 판단하지 말 것(자격만)."
)

_TOOL = {
    "name": "judge",
    "description": "가맹점의 혜택 자격을 판정한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "eligible": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string", "description": "한 문장 근거."},
        },
        "required": ["eligible", "confidence", "reason"],
    },
}


def judge(merchant_name: str, cand: CandidateClause) -> Judgment:
    cid, clid = cand.calc.card_id, cand.calc.clause_id
    prompt = (
        f"가맹점: {merchant_name}\n"
        f"업종: {cand.calc.category}\n"
        f"혜택 포함 대상(화이트리스트): {cand.include_text or '제한 없음'}\n"
        f"제외 대상: {cand.exclude_text or '없음'}\n"
        "이 가맹점이 이 혜택의 대상인가?"
    )
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=300,
            system=[{"type": "text", "text": _SYS, "cache_control": {"type": "ephemeral"}}],
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "judge"},
            messages=[{"role": "user", "content": prompt}],
        )
        out = next(b.input for b in resp.content if b.type == "tool_use")
        return Judgment(cid, clid, bool(out["eligible"]), out.get("confidence", "medium"),
                        out.get("reason", ""), "llm")
    except Exception as e:  # 가드레일: 판정 실패 시 관대 폴백 + 주의
        return Judgment(cid, clid, True, "low", f"판정 실패({type(e).__name__}) → 관대 적용(참고용)", "fallback")
