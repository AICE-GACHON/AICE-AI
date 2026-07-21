"""임베딩 유틸 테스트 (모델 로드 없이 순수 함수만 — CI에서 빠르게 돌도록)."""
import pytest

from paper_assistant.embedding.specter2 import similarity_percentile


def test_percentile_matches_measured_reference_points():
    """scripts/measure_similarity_dist.py 실측값과 일치해야 한다."""
    assert similarity_percentile(0.8448) == pytest.approx(50.0, abs=0.1)
    assert similarity_percentile(0.9231) == pytest.approx(99.0, abs=0.1)
    assert similarity_percentile(0.7924) == pytest.approx(5.0, abs=0.1)


def test_percentile_is_monotonic():
    scores = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.96, 0.99]
    pcts = [similarity_percentile(s) for s in scores]
    assert pcts == sorted(pcts)


def test_percentile_stays_in_range():
    for s in (0.0, 0.5, 0.72, 0.845, 0.95, 1.0):
        assert 0.0 <= similarity_percentile(s) <= 100.0


def test_unrelated_pair_lands_near_median():
    """무관한 논문쌍의 실측 코사인 0.844는 중앙값 근처여야 한다.

    이 값이 '84% 유사'로 오해되는 것을 막는 것이 이 함수의 존재 이유.
    """
    assert 40 < similarity_percentile(0.8443) < 60


def test_strongly_related_pair_is_top_percentile():
    """Transformer/BERT 쌍(0.9202), tabular 쌍(0.9563)의 실측값."""
    assert similarity_percentile(0.9202) > 95
    assert similarity_percentile(0.9563) > 99
