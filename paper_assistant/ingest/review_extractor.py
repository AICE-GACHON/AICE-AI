"""리뷰 → 지적 항목 추출.

두 가지 구현을 같은 인터페이스(PointExtractor)로 제공한다:

  HeuristicExtractor : LLM 없이 규칙 기반. 비용 $0. (현재 MVP 기본값)
  HaikuExtractor     : Claude Haiku로 추출. 품질은 높지만 유료. (플레이스홀더)

둘 다 ExtractedPoint 리스트를 반환하므로, 품질이 부족하면 수집 스크립트에서
extractor만 바꿔 끼우면 된다 — 다운스트림(임베딩·클러스터링·집계)은 그대로.

Heuristic 방식의 근거:
리뷰어는 약점을 대부분 불릿(-, *, •), 번호(1. 2.), 또는 문장 단위로 나열한다.
이 구조를 이용해 지적 항목을 분리하고, 통제된 aspect 분류(§4)는 키워드 매칭으로
근사한다. LLM만큼 정교하진 않지만 클러스터링의 1차 그룹핑에는 충분하다.
"""
import re
from dataclasses import dataclass
from typing import Protocol

from paper_assistant.ingest.normalize import NormalizedReview

# 통제된 aspect 분류 — normalize.py / init_db.sql 과 동일한 9개 체계.
# 각 aspect의 대표 키워드(소문자, 정규식). 우선순위는 리스트 순서.
ASPECT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("experimental_scale", [
        r"\bimagenet\b", r"\blarger (?:dataset|scale|benchmark)", r"\bsmall(?:er)? (?:dataset|scale)",
        r"\btoy (?:dataset|example)", r"\bonly .{0,20}(?:cifar|mnist)", r"\bscalab", r"\bscale up\b"]),
    ("baselines", [
        r"\bbaseline", r"\bcompar(?:e|ison|ed) (?:to|with|against)", r"\bsota\b",
        r"\bstate[- ]of[- ]the[- ]art", r"\bmissing compar", r"\bstronger baseline"]),
    ("novelty", [
        r"\bnovelt", r"\bincrement", r"\bnot novel", r"\blimited (?:novelty|contribution)",
        r"\bmarginal (?:contribution|improvement)", r"\bsimilar to (?:prior|existing)"]),
    ("theoretical_soundness", [
        r"\btheor", r"\bproof\b", r"\bassumption", r"\bderivation", r"\bmathematic",
        r"\bformal(?:ly)? (?:incorrect|wrong)", r"\brigor"]),
    ("reproducibility", [
        r"\breproduc", r"\bcode (?:is )?(?:not )?(?:available|released)", r"\bimplementation detail",
        r"\bhyperparameter", r"\bunclear how to"]),
    ("clarity", [
        r"\bunclear", r"\bhard to (?:follow|read|understand)", r"\bconfusing", r"\bpoorly written",
        r"\bnotation", r"\btypo", r"\bpresentation (?:is|could)", r"\bwriting"]),
    ("related_work", [
        r"\brelated work", r"\bmissing (?:citation|reference)", r"\bprior work",
        r"\bfails? to cite", r"\bliterature"]),
    ("significance", [
        r"\bsignifican", r"\bimpact", r"\bpractical (?:use|value|relevance)", r"\bmotivat",
        r"\bwhy .{0,30}(?:matter|important)", r"\bnot useful"]),
]

# 지적 항목 분리용 패턴
_BULLET = re.compile(r"^\s*(?:[-*•·▪]|\(?\d+[.)]|\(?[a-z][.)])\s+", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MIN_POINT_CHARS = 25   # 이보다 짧으면 지적 항목으로 보지 않는다
MAX_POINTS_PER_REVIEW = 12


@dataclass
class ExtractedPoint:
    aspect: str
    sentiment: str   # weakness / strength / question
    text: str


class PointExtractor(Protocol):
    def extract(self, review: NormalizedReview) -> list[ExtractedPoint]: ...


def classify_aspect(text: str) -> str:
    """키워드 매칭으로 aspect를 근사. 매칭 없으면 'other'."""
    low = text.lower()
    for aspect, patterns in ASPECT_KEYWORDS:
        if any(re.search(p, low) for p in patterns):
            return aspect
    return "other"


def _split_points(text: str) -> list[str]:
    """텍스트를 지적 항목 단위로 분리.

    불릿/번호 목록이 있으면 그 단위로, 없으면 문장 단위로 쪼갠다.
    """
    text = text.strip()
    if not text:
        return []

    # 불릿/번호가 2개 이상이면 목록으로 간주하고 그 경계로 분리
    if len(_BULLET.findall(text)) >= 2:
        parts = _BULLET.split(text)
    else:
        parts = _SENTENCE_SPLIT.split(text)

    points = [p.strip() for p in parts if len(p.strip()) >= MIN_POINT_CHARS]
    return points[:MAX_POINTS_PER_REVIEW]


class HeuristicExtractor:
    """LLM 없이 규칙 기반으로 지적 항목을 추출. 비용 $0."""

    def extract(self, review: NormalizedReview) -> list[ExtractedPoint]:
        points: list[ExtractedPoint] = []

        # 강/약이 분리된 venue는 weaknesses 필드가 곧 약점 목록.
        # 분리 안 된 venue(needs_llm_split)는 리뷰 본문 전체가 weaknesses에 들어와
        # 강점·약점이 섞여 있지만, 클러스터링 관점에선 '지적 후보'로 묶어도 무방하다.
        for chunk in _split_points(review.weaknesses):
            points.append(ExtractedPoint(
                aspect=classify_aspect(chunk),
                sentiment="weakness",
                text=chunk,
            ))

        # 질문도 사실상 지적인 경우가 많아 포함 (sentiment로 구분)
        for chunk in _split_points(review.questions):
            points.append(ExtractedPoint(
                aspect=classify_aspect(chunk),
                sentiment="question",
                text=chunk,
            ))
        return points[:MAX_POINTS_PER_REVIEW]


class HaikuExtractor:
    """Claude Haiku 기반 추출 (플레이스홀더).

    품질이 부족하다고 판단되면 이 클래스를 구현해 HeuristicExtractor 대신 끼운다.
    수집 단계에서 Batch API로 돌리는 것을 권장 (비용 50% 절감).
    구현 시 통제된 aspect 분류를 프롬프트에 명시하고 ExtractedPoint를 반환할 것.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HaikuExtractor는 아직 미구현. 예산 확보 후 구현하세요 "
            "(AI_파트_설계서.md §4, §13 참고). 현재는 HeuristicExtractor를 쓰세요.")

    def extract(self, review: NormalizedReview) -> list[ExtractedPoint]:
        raise NotImplementedError
