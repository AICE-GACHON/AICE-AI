"""휴리스틱 지적항목 추출 테스트 (LLM 불필요)."""
from paper_assistant.ingest.normalize import NormalizedReview
from paper_assistant.ingest.review_extractor import (
    HeuristicExtractor, classify_aspect)


def _review(weaknesses="", questions="", needs_split=False):
    return NormalizedReview(
        openreview_id="r", rating=5.0, rating_raw="5", confidence=4.0,
        summary="", strengths="", weaknesses=weaknesses, questions=questions,
        needs_llm_split=needs_split)


def test_classify_aspect_maps_known_phrases():
    assert classify_aspect("Experiments are limited to CIFAR-10, no ImageNet") == "experimental_scale"
    assert classify_aspect("The paper lacks comparison to strong baselines") == "baselines"
    assert classify_aspect("The contribution is incremental over prior work") == "novelty"
    assert classify_aspect("No proof is given for Theorem 2") == "theoretical_soundness"
    assert classify_aspect("Code is not available, hard to reproduce") == "reproducibility"
    assert classify_aspect("The writing is confusing and hard to follow") == "clarity"
    assert classify_aspect("Missing citations to related work on GNNs") == "related_work"


def test_classify_aspect_defaults_to_other():
    assert classify_aspect("The color of the figures is nice.") == "other"


def test_bullet_list_is_split_into_points():
    ext = HeuristicExtractor()
    review = _review(weaknesses=(
        "- The experiments are limited to small datasets like MNIST.\n"
        "- No comparison to recent baselines such as method X.\n"
        "- The theoretical analysis lacks a convergence proof."))
    points = ext.extract(review)
    assert len(points) == 3
    aspects = {p.aspect for p in points}
    assert "experimental_scale" in aspects
    assert "baselines" in aspects
    assert "theoretical_soundness" in aspects
    assert all(p.sentiment == "weakness" for p in points)


def test_numbered_list_is_split():
    ext = HeuristicExtractor()
    review = _review(weaknesses=(
        "1. The method is only evaluated on toy examples. "
        "2. The related work section misses key prior citations."))
    points = ext.extract(review)
    assert len(points) == 2


def test_prose_is_split_into_sentences():
    ext = HeuristicExtractor()
    review = _review(weaknesses=(
        "The proposed approach is vaguely described and not well justified. "
        "The baselines are weak and outdated. "
        "It is unclear how the hyperparameters were chosen."))
    points = ext.extract(review)
    assert len(points) == 3


def test_short_fragments_are_dropped():
    ext = HeuristicExtractor()
    review = _review(weaknesses="- ok\n- fine\n- The experiments only use MNIST which is too small.")
    points = ext.extract(review)
    # 'ok', 'fine'은 MIN_POINT_CHARS 미만이라 제외
    assert len(points) == 1
    assert points[0].aspect == "experimental_scale"


def test_questions_are_included_as_question_sentiment():
    ext = HeuristicExtractor()
    review = _review(
        weaknesses="The evaluation is limited to a single small benchmark dataset.",
        questions="Why was ImageNet-scale evaluation not included in the experiments?")
    points = ext.extract(review)
    sentiments = {p.sentiment for p in points}
    assert "weakness" in sentiments
    assert "question" in sentiments


def test_empty_review_yields_nothing():
    assert HeuristicExtractor().extract(_review()) == []
