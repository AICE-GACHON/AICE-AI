"""리뷰 지적항목 → 패턴 집계.

**설계 변경 근거 (§14 실측):**
SPECTER2로 지적항목 문장을 임베딩하면 유사도가 평균 0.872에 압축되어(논문
title+abstract용 모델이라 짧은 리뷰 문장을 변별 못 함), 임계값 0.80에서도 대부분
한 클러스터로 뭉친다. 따라서 임베딩 클러스터링 대신 **키워드 aspect 기반 집계**를
1차 방법으로 쓴다: "20편 중 8편이 baselines 지적" 형태. 쿼리 시점 임베딩 불필요.

_greedy_cluster는 범용 유틸로 남겨둔다 (향후 aspect 내부 세분화 등에 사용 가능).
"""
import numpy as np

from paper_assistant.schemas import ReviewPattern

# aspect 표시 라벨 (프론트/요약용)
ASPECT_LABELS = {
    "experimental_scale": "실험 규모·범위",
    "baselines": "베이스라인 비교 부족",
    "novelty": "신규성 부족",
    "theoretical_soundness": "이론적 근거",
    "reproducibility": "재현성",
    "clarity": "명확성·서술",
    "related_work": "관련 연구 누락",
    "significance": "중요성·동기",
    "other": "기타 지적",
}


def aggregate_by_aspect(points: list[dict], total_papers: int,
                        min_papers: int = 2, top_k: int = 8,
                        include_other: bool = False) -> list[ReviewPattern]:
    """지적항목을 aspect별로 묶어 "N편 중 M편" 패턴 생성.

    points: [{"paper_id", "aspect", "text"}, ...] (weakness sentiment 권장)
    total_papers: 분석 대상 유사 논문 총수 (분모)
    include_other: 'other' 버킷도 패턴으로 낼지 (기본 제외 — 일회성 지적이 대부분)
    """
    by_aspect: dict[str, list[dict]] = {}
    for p in points:
        if not p.get("text"):
            continue
        if p["aspect"] == "other" and not include_other:
            continue
        by_aspect.setdefault(p["aspect"], []).append(p)

    patterns = []
    for aspect, items in by_aspect.items():
        paper_ids = {it["paper_id"] for it in items}
        if len(paper_ids) < min_papers:
            continue
        # 대표 문장: 가장 긴(= 구체적인) 지적을 예시 앞에 둔다
        items_sorted = sorted(items, key=lambda it: len(it["text"]), reverse=True)
        # 서로 다른 논문에서 예시를 뽑아 다양성 확보
        examples, seen = [], set()
        for it in items_sorted:
            if it["paper_id"] not in seen:
                examples.append(it["text"][:160])
                seen.add(it["paper_id"])
            if len(examples) >= 3:
                break
        patterns.append(ReviewPattern(
            label=ASPECT_LABELS.get(aspect, aspect),
            aspect=aspect,
            paper_count=len(paper_ids),
            total_papers=total_papers,
            examples=examples,
        ))

    patterns.sort(key=lambda p: p.paper_count, reverse=True)
    return patterns[:top_k]


def _greedy_cluster(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """정규화된 벡터들을 그리디하게 묶는다. 각 클러스터는 인덱스 리스트.

    범용 유틸. 현재 리뷰 패턴 집계에는 쓰지 않지만(§14), 향후 aspect 내부
    세분화 등에 재사용할 수 있다.
    """
    n = len(vectors)
    assigned = [False] * n
    sim = vectors @ vectors.T
    clusters = []
    order = np.argsort(-(sim >= threshold).sum(axis=1))
    for seed in order:
        if assigned[seed]:
            continue
        members = [int(seed)]
        assigned[seed] = True
        for j in range(n):
            if not assigned[j] and sim[seed, j] >= threshold:
                members.append(j)
                assigned[j] = True
        clusters.append(members)
    return clusters
