"""RRF 결합 로직 테스트 (DB 불필요)."""
from paper_assistant.retrieval.hybrid_search import rrf_fuse


def test_both_retrievers_agreeing_beats_either_alone():
    """양쪽 검색기에 모두 잡힌 문서가 한쪽에서만 1위인 문서를 이겨야 한다.

    이것이 하이브리드를 쓰는 이유 그 자체다.
    """
    vector = {100: 1, 200: 2}   # 벡터 1위=100
    fts = {200: 1, 300: 2}      # FTS 1위=200
    scores = rrf_fuse(vector, fts)
    # 200은 양쪽에 잡힘(2위+1위), 100은 벡터에서만 1위
    assert scores[200] > scores[100]
    assert scores[200] > scores[300]


def test_rank_one_scores_higher_than_rank_two():
    scores = rrf_fuse({10: 1, 20: 2, 30: 3})
    assert scores[10] > scores[20] > scores[30]


def test_missing_from_one_retriever_still_scores():
    """한쪽에만 있어도 탈락하지 않는다 (OR 결합이지 AND가 아니다)."""
    scores = rrf_fuse({1: 1}, {2: 1})
    assert 1 in scores and 2 in scores
    assert scores[1] == scores[2]


def test_k_dampens_rank_differences():
    """k가 클수록 상위 순위 간 점수 차이가 완만해진다."""
    tight = rrf_fuse({1: 1, 2: 10}, k=60)
    loose = rrf_fuse({1: 1, 2: 10}, k=1)
    assert (loose[1] / loose[2]) > (tight[1] / tight[2])


def test_empty_input_returns_empty():
    assert rrf_fuse() == {}
    assert rrf_fuse({}, {}) == {}
