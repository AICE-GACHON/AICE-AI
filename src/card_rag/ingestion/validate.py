"""[4-자동] 추출 결과 자동 검증 — 사람 검수를 대체하는 결정적 게이트.

전체 자동화 결정에 따라 사람 검수를 없앤 대신, 스키마·정합성 검사를 통과해야만 적재한다.
errors(블로킹)와 warnings(비블로킹)를 분리한다. LLM 판단이 애매한 것은 warning으로 남긴다.
"""
from __future__ import annotations

from typing import get_args

from card_rag.schemas.clause import ExtractedClause, ExtractionResult, InternalCategory

_ALLOWED_CATEGORIES = set(get_args(InternalCategory))


def validate_clause(c: ExtractedClause, idx: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    tag = f"[{idx}] {c.category}"

    if c.category not in _ALLOWED_CATEGORIES:
        errors.append(f"{tag}: 허용되지 않은 업종코드")

    if c.value_type == "percent":
        if not (0 < c.rate <= 100):
            errors.append(f"{tag}: percent인데 rate={c.rate} (0~100 벗어남)")
        if c.flat_amount:
            warnings.append(f"{tag}: percent인데 flat_amount가 채워짐 — 무시됨")
    elif c.value_type == "flat":
        if not c.flat_amount or c.flat_amount <= 0:
            errors.append(f"{tag}: flat인데 flat_amount 없음/0")
        if c.flat_min_txn < 0:
            errors.append(f"{tag}: flat_min_txn 음수")
        if c.rate:
            warnings.append(f"{tag}: flat인데 rate가 채워짐 — 무시됨")

    if c.monthly_cap is not None and c.monthly_cap < 0:
        errors.append(f"{tag}: monthly_cap 음수")
    if c.min_spend < 0:
        errors.append(f"{tag}: min_spend 음수")

    if c.confidence == "low":
        warnings.append(f"{tag}: 저신뢰 추출 — 원문 대조 권장")
    if not c.include_notes and not c.exclude_notes:
        warnings.append(f"{tag}: 포함/제외 조건이 비어 있음 — 무조건 적용으로 간주됨")

    return errors, warnings


def validate_clauses(result: ExtractionResult) -> tuple[list[str], list[str]]:
    """반환: (errors, warnings). errors가 있으면 적재를 막는다."""
    errors: list[str] = []
    warnings: list[str] = []
    if not result.clauses:
        errors.append("추출된 혜택절이 0건")
    for i, c in enumerate(result.clauses):
        e, w = validate_clause(c, i)
        errors.extend(e)
        warnings.extend(w)
    return errors, warnings
