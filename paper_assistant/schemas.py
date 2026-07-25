"""공개 API 스키마 (백엔드 통합 계약).

백엔드 팀은 `analyze(...) -> Report` 하나와 이 Pydantic 모델들만 알면 된다.
원시 코사인 값은 절대 노출하지 않는다 — similarity_percentile만 담는다 (설계서 §11.2).
"""
from pydantic import BaseModel, Field


class SimilarityTag(BaseModel):
    """왜 유사한가에 대한 구조화된 근거 (설계서 4.1)."""
    kind: str = Field(description="methodology | dataset | problem_setting | citation")
    reason: str = Field(description="한 줄 근거")


class SimilarPaper(BaseModel):
    paper_id: int
    openreview_id: str
    title: str
    venue: str
    year: int
    decision: str
    similarity_percentile: float = Field(
        description="무작위 논문쌍 대비 백분위(0~100). 원시 코사인 아님.")
    rank: int
    tags: list[SimilarityTag] = Field(default_factory=list)


class ReviewPattern(BaseModel):
    """유사 논문들에서 반복 등장하는 지적 패턴 (지적항목 임베딩 클러스터)."""
    label: str = Field(description="대표 지적 문장 (클러스터 medoid)")
    aspect: str = Field(description="통제된 분류 또는 'other'")
    paper_count: int = Field(description="이 지적을 받은 유사 논문 수")
    total_papers: int = Field(description="분석 대상 유사 논문 총수")
    examples: list[str] = Field(default_factory=list, description="지적 문장 예시")


class VenueTrend(BaseModel):
    """유사 논문들의 게재 학회/결과 경향."""
    venue: str
    year: int | None = None
    paper_count: int
    accept_count: int
    accept_rate: float


class ResubmissionFlow(BaseModel):
    """재투고 흐름 (예: ICLR reject → NeurIPS accept)."""
    from_venue: str
    to_venue: str
    count: int


class Report(BaseModel):
    """analyze()의 최종 반환. 프론트가 섹션별로 렌더링할 수 있도록 구조화."""
    query_title: str
    query_abstract: str
    similar_papers: list[SimilarPaper] = Field(default_factory=list)
    review_patterns: list[ReviewPattern] = Field(default_factory=list)
    venue_trends: list[VenueTrend] = Field(default_factory=list)
    resubmission_flows: list[ResubmissionFlow] = Field(default_factory=list)
    summary_markdown: str = Field(
        default="", description="사람이 읽는 종합 요약 (LLM 생성)")
