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

    # --- 리뷰 점수 (§19). 원점수는 venue별 척도가 달라 단독 해석 금지 ---
    avg_rating: float | None = Field(
        default=None, description="이 논문이 받은 리뷰 점수 평균")
    rating_count: int = Field(default=0, description="리뷰 수")
    rating_spread: float | None = Field(
        default=None, description="최고-최저 점수 차. 크면 리뷰어 의견이 갈렸다는 뜻")
    rating_vs_venue: float | None = Field(
        default=None, description="같은 venue 평균 대비 차이(+면 평균 이상)")
    rating_vs_threshold: float | None = Field(
        default=None, description="당락 경계 대비 차이. 편향 venue는 None")


class ReviewPattern(BaseModel):
    """유사 논문들에서 반복 등장하는 지적 패턴.

    단순 빈도만으로는 정보가 없다 — baselines 지적은 코퍼스 전체 78.8%의 논문이
    받으므로 "20편 중 17편"은 사실상 상수다(설계서 §18). 그래서 코퍼스 base rate
    대비 **lift**와 이항검정 p값을 함께 싣고, 당락 대조까지 붙인다.
    """
    label: str = Field(description="aspect 표시 라벨")
    aspect: str = Field(description="통제된 분류 또는 'other'")
    paper_count: int = Field(description="이 지적을 받은 유사 논문 수")
    total_papers: int = Field(description="분석 대상 유사 논문 총수")
    examples: list[str] = Field(default_factory=list, description="지적 문장 예시")

    # --- base rate 대비 두드러짐 (base_rates 미제공 시 None) ---
    base_rate: float | None = Field(
        default=None, description="코퍼스 전체에서 이 지적을 받는 논문 비율(0~1)")
    lift: float | None = Field(
        default=None, description="이웃 지적률 / base_rate. 1이면 평범, >1이면 이 주제 특유")
    p_value: float | None = Field(
        default=None, description="이항검정 단측 p값 (관측 방향 기준)")
    is_distinctive: bool = Field(
        default=False, description="이 주제군에서 유의하게 두드러진 지적인지")

    # --- 당락 대조: 이 지적을 받은 이웃 vs 받지 않은 이웃 ---
    accept_with: int = Field(default=0, description="이 지적을 받은 이웃 중 accept 수")
    accept_without: int = Field(default=0, description="이 지적이 없는 이웃 중 accept 수")
    decided_with: int = Field(default=0, description="이 지적을 받은 이웃 중 결과 확정 수")
    decided_without: int = Field(default=0, description="이 지적이 없는 이웃 중 결과 확정 수")
    accept_rate_with: float | None = Field(
        default=None, description="이 지적을 받고도 통과한 비율(0~1)")
    accept_rate_without: float | None = Field(
        default=None, description="이 지적이 없을 때 통과 비율(0~1)")
    contrast_p_value: float | None = Field(
        default=None, description="당락 대조의 단측 Fisher 정확검정 p값")
    is_contrast_significant: bool = Field(
        default=False,
        description="당락 격차가 통계적으로 유의한지. False면 표본 부족 — 단정 금지")


class VenueTrend(BaseModel):
    """유사 논문들의 게재 학회/결과 경향.

    ⚠️ accept_rate를 실제 채택률로 해석하면 안 되는 venue가 있다 — NeurIPS는
    코퍼스의 95%가 accept다(OpenReview가 채택 논문 위주로만 공개). `is_coverage_biased`
    가 참이면 절대값이 아니라 코퍼스 대비 상대값(lift)으로만 말한다 (설계서 §19).
    """
    venue: str
    year: int | None = None
    paper_count: int
    accept_count: int
    accept_rate: float
    corpus_accept_rate: float | None = Field(
        default=None, description="이 학회 코퍼스 전체의 accept 비율 (비교 기준선)")
    accept_lift: float | None = Field(
        default=None, description="이웃 accept율 / 코퍼스 accept율")
    is_coverage_biased: bool = Field(
        default=False, description="참이면 accept율 절대값 해석 금지 — 표본이 채택 편향")


class RatingContext(BaseModel):
    """이웃 논문들의 리뷰 점수 분포와 당락 기준선 (§19).

    "비슷한 논문들은 몇 점을 받았고, 붙으려면 몇 점이 필요한가"에 답한다.
    """
    neighbor_mean: float | None = Field(default=None, description="이웃 평균 점수")
    accepted_mean: float | None = Field(default=None, description="통과한 이웃의 평균")
    rejected_mean: float | None = Field(default=None, description="탈락한 이웃의 평균")
    rated_papers: int = Field(default=0, description="점수가 있는 이웃 수")
    threshold: float | None = Field(
        default=None, description="당락 경계 추정치 (편향 없는 venue 기준)")
    threshold_venue: str | None = Field(
        default=None, description="경계를 가져온 venue")
    split_papers: list[str] = Field(
        default_factory=list,
        description="리뷰어 의견이 크게 갈린 논문 제목 (spread 상위)")
    biased_venues: list[str] = Field(
        default_factory=list,
        description="표본이 채택 편향된 venue 목록 — accept율 절대해석 금지")


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
    rating_context: RatingContext = Field(default_factory=lambda: RatingContext())
    resubmission_flows: list[ResubmissionFlow] = Field(default_factory=list)
    summary_markdown: str = Field(
        default="", description="사람이 읽는 종합 요약 (LLM 생성)")
