"""파이프라인 구조 테스트 — DAG 배선과 노드 로직 (모델/LLM 없이).

무거운 SPECTER2 로드를 피하려고 embedder를 가짜로 주입하고,
DB가 필요한 노드는 별도 통합 테스트(test_db_integration 계열)에 맡긴다.
여기서는 순수 로직(노드 반환 형태, 종합 조립)만 검증한다.
"""
from paper_assistant.graph import nodes
from paper_assistant.graph.state import PipelineState
from paper_assistant.retrieval.hybrid_search import SearchResult
from paper_assistant.schemas import (
    Report, ResubmissionFlow, ReviewPattern, VenueTrend)


def _fake_paper(pid, title, decision="reject", pct=90.0):
    return SearchResult(
        paper_id=pid, openreview_id=f"or{pid}", title=title, abstract="abs",
        venue="ICLR 2024", year=2024, decision=decision, rrf_score=0.03,
        vector_rank=pid, fts_rank=pid, similarity_percentile=pct)


def test_tagging_node_stub_returns_empty_tags_without_llm():
    state: PipelineState = {"query_title": "Q", "query_abstract": "A",
                            "similar_papers": [_fake_paper(1, "P1")]}
    out = nodes.similarity_tagging_node(state, embedder=None, llm=None)
    assert out["similarity_tags"] == {1: []}


def test_synthesis_assembles_report_without_llm():
    state: PipelineState = {
        "query_title": "Graph nets",
        "query_abstract": "abstract",
        "similar_papers": [_fake_paper(1, "P1", "accept-poster"),
                           _fake_paper(2, "P2", "reject")],
        "similarity_tags": {1: [], 2: []},
        "review_patterns": [ReviewPattern(
            label="weak baselines", aspect="baselines",
            paper_count=2, total_papers=2, examples=["x"])],
        "venue_trends": [VenueTrend(venue="ICLR 2024", paper_count=2,
                                    accept_count=1, accept_rate=0.5)],
        "resubmission_flows": [ResubmissionFlow(
            from_venue="ICLR 2024", to_venue="NeurIPS 2024", count=3)],
    }
    out = nodes.synthesis_node(state, embedder=None, llm=None)
    report = out["report"]
    assert isinstance(report, Report)
    assert len(report.similar_papers) == 2
    assert report.similar_papers[0].rank == 1
    assert report.similar_papers[0].similarity_percentile == 90.0
    assert report.review_patterns[0].aspect == "baselines"
    assert report.resubmission_flows[0].from_venue == "ICLR 2024"
    assert report.resubmission_flows[0].count == 3
    assert "유사 논문 2편" in report.summary_markdown


def test_report_is_json_serializable():
    """백엔드 전달용 — Pydantic → JSON 왕복."""
    state: PipelineState = {
        "query_title": "Q", "query_abstract": "A",
        "similar_papers": [_fake_paper(1, "P1")],
        "similarity_tags": {1: []}, "review_patterns": [], "venue_trends": [],
    }
    report = nodes.synthesis_node(state, embedder=None, llm=None)["report"]
    dumped = report.model_dump_json()
    restored = Report.model_validate_json(dumped)
    assert restored.query_title == "Q"


def test_graph_compiles_with_fake_embedder():
    """DAG가 컴파일되고 노드/엣지가 연결되는지 (실행은 안 함)."""
    from paper_assistant.graph.pipeline import build

    class FakeEmbedder:
        dim = 768
    graph, _ = build(embedder=FakeEmbedder(), use_llm=False)
    node_names = set(graph.get_graph().nodes)
    assert {"input", "retrieval", "similarity_tagging",
            "review_analysis", "venue_trend", "synthesis"} <= node_names
