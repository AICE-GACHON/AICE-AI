"""리뷰 지적항목 집계·클러스터링 로직 테스트 (모델/DB 불필요, numpy만)."""
import numpy as np

from paper_assistant.graph.clustering import _greedy_cluster, aggregate_by_aspect


def _pt(paper_id, aspect, text="a criticism that is long enough to keep"):
    return {"paper_id": paper_id, "aspect": aspect, "text": text}


def test_aggregate_counts_distinct_papers_per_aspect():
    points = [_pt(1, "baselines"), _pt(2, "baselines"),
              _pt(2, "baselines"),   # 같은 논문 중복 → 1편으로
              _pt(3, "clarity")]
    patterns = aggregate_by_aspect(points, total_papers=5, min_papers=1)
    by = {p.aspect: p for p in patterns}
    assert by["baselines"].paper_count == 2   # 논문 1,2
    assert by["clarity"].paper_count == 1
    assert by["baselines"].total_papers == 5


def test_aggregate_excludes_other_by_default():
    points = [_pt(1, "other"), _pt(2, "other"), _pt(3, "baselines"), _pt(4, "baselines")]
    patterns = aggregate_by_aspect(points, total_papers=4)
    assert all(p.aspect != "other" for p in patterns)


def test_aggregate_respects_min_papers():
    points = [_pt(1, "novelty"), _pt(2, "baselines"), _pt(3, "baselines")]
    patterns = aggregate_by_aspect(points, total_papers=3, min_papers=2)
    assert {p.aspect for p in patterns} == {"baselines"}  # novelty는 1편이라 제외


def test_aggregate_sorted_by_paper_count():
    points = ([_pt(i, "baselines") for i in range(5)] +
              [_pt(i, "clarity") for i in range(2)])
    patterns = aggregate_by_aspect(points, total_papers=5, min_papers=2)
    assert patterns[0].aspect == "baselines"
    assert patterns[0].paper_count > patterns[1].paper_count


def test_aggregate_examples_from_distinct_papers():
    points = [_pt(1, "clarity", "paper one is unclear about notation everywhere"),
              _pt(1, "clarity", "paper one also has typos throughout the draft"),
              _pt(2, "clarity", "paper two has confusing figures and captions")]
    patterns = aggregate_by_aspect(points, total_papers=2, min_papers=1)
    examples = patterns[0].examples
    # 서로 다른 논문에서 예시를 뽑아야 함 (논문1 하나 + 논문2 하나)
    assert len(examples) == 2


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_two_tight_groups_separate():
    # 두 방향으로 뚜렷이 갈리는 벡터 → 2개 클러스터
    vecs = _norm([[1, 0, 0], [0.98, 0.02, 0], [0.97, 0.05, 0],   # 그룹 A
                  [0, 1, 0], [0.02, 0.98, 0]])                    # 그룹 B
    clusters = _greedy_cluster(vecs, threshold=0.9)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


def test_all_similar_one_cluster():
    vecs = _norm([[1, 0], [0.99, 0.01], [0.98, 0.02]])
    clusters = _greedy_cluster(vecs, threshold=0.8)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_all_dissimilar_singletons():
    vecs = _norm([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    clusters = _greedy_cluster(vecs, threshold=0.5)
    assert len(clusters) == 3
    assert all(len(c) == 1 for c in clusters)


def test_every_point_assigned_exactly_once():
    rng = np.random.default_rng(0)
    vecs = _norm(rng.normal(size=(30, 8)))
    clusters = _greedy_cluster(vecs, threshold=0.6)
    flat = [i for c in clusters for i in c]
    assert sorted(flat) == list(range(30))   # 빠짐/중복 없음


def test_high_degree_point_seeds_first():
    # 0번이 1,2,3과 모두 유사, 4번은 고립 → 0이 큰 클러스터 시드
    vecs = _norm([[1, 0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [-1, 0]])
    clusters = _greedy_cluster(vecs, threshold=0.9)
    biggest = max(clusters, key=len)
    assert 0 in biggest and len(biggest) == 4
