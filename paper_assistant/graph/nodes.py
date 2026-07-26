"""LangGraph 노드 6종.

지능이 필요한 노드(태깅·종합)만 LLM을 쓰고, 나머지는 코드로 처리한다.
LLM이 None이면(예산 off) 결정론적 스텁을 만들어 DAG를 $0으로 검증할 수 있다.

노드는 embedder / llm을 클로저로 주입받는다 (pipeline.build에서 바인딩).
"""
import json
import logging

from paper_assistant.db.connection import cursor
from paper_assistant.graph.clustering import aggregate_by_aspect
from paper_assistant.graph.llm import HAIKU, SONNET
from paper_assistant.graph.state import PipelineState
from paper_assistant.retrieval.hybrid_search import hybrid_search
from paper_assistant.schemas import (
    Report, ResubmissionFlow, ReviewPattern, SimilarityTag, SimilarPaper,
    VenueTrend)

log = logging.getLogger(__name__)

TAG_KINDS = ("methodology", "dataset", "problem_setting", "citation")


# ---------------------------------------------------------------- input
def input_node(state: PipelineState, embedder, llm) -> dict:
    """PDF면 제목/초록 추출, 텍스트면 통과."""
    if state.get("pdf_bytes") and not state.get("query_abstract"):
        from paper_assistant.pdf.extract import extract_title_abstract
        title, abstract = extract_title_abstract(state["pdf_bytes"], llm=llm)
        return {"query_title": title, "query_abstract": abstract}
    return {}


# ------------------------------------------------------------ retrieval
def retrieval_node(state: PipelineState, embedder, llm) -> dict:
    """SPECTER2 임베딩 → 하이브리드 검색 top-K."""
    title = state["query_title"]
    abstract = state.get("query_abstract", "")
    qvec = embedder.encode_one(title, abstract).numpy()
    results = hybrid_search(qvec, f"{title} {abstract}", top_k=20)
    return {"query_embedding": qvec.tolist(), "similar_papers": results}


# ----------------------------------------------- similarity tagging (LLM)
def similarity_tagging_node(state: PipelineState, embedder, llm) -> dict:
    """상위 논문마다 '왜 유사한가' 태깅. LLM off면 스텁."""
    papers = state.get("similar_papers", [])[:10]
    tags: dict[int, list[SimilarityTag]] = {}

    if llm is None:
        # 스텁: 태그 없이 통과 (배선 검증용)
        return {"similarity_tags": {p.paper_id: [] for p in papers}}

    query = f"{state['query_title']}\n{state.get('query_abstract','')}"
    system = (
        "You compare a query paper to a candidate paper and explain why they are "
        "similar. Return JSON: {\"tags\": [{\"kind\": one of "
        "[methodology, dataset, problem_setting, citation], \"reason\": short}]}. "
        "Only include tags that genuinely apply. Reasons under 15 words.")
    for p in papers:
        user = (f"QUERY:\n{query}\n\nCANDIDATE:\n{p.title}\n{p.abstract}")
        out = llm.json(HAIKU, system, user, max_tokens=400)
        parsed = []
        for t in out.get("tags", []):
            if t.get("kind") in TAG_KINDS and t.get("reason"):
                parsed.append(SimilarityTag(kind=t["kind"], reason=t["reason"][:120]))
        tags[p.paper_id] = parsed
    return {"similarity_tags": tags}


# ------------------------------------------------- review analysis (no LLM)
def review_analysis_node(state: PipelineState, embedder, llm) -> dict:
    """유사 논문들의 지적항목을 aspect별로 집계 (쿼리 시점, 임베딩 불필요).

    SPECTER2 임베딩 클러스터링은 리뷰 문장에 부적합해(§14) aspect 기반 집계를 쓴다.
    """
    papers = state.get("similar_papers", [])
    paper_ids = [p.paper_id for p in papers]
    if not paper_ids:
        return {"review_patterns": []}

    with cursor() as cur:
        cur.execute(
            """
            SELECT paper_id, aspect, text FROM review_points
            WHERE paper_id = ANY(%s) AND sentiment = 'weakness'
            """,
            (paper_ids,))
        points = [{"paper_id": r[0], "aspect": r[1], "text": r[2]}
                  for r in cur.fetchall()]

    patterns = aggregate_by_aspect(points, total_papers=len(paper_ids))
    return {"review_patterns": patterns}


# --------------------------------------------------- venue trend (no LLM)
def venue_trend_node(state: PipelineState, embedder, llm) -> dict:
    """유사 논문들의 게재 결과를 SQL 집계."""
    papers = state.get("similar_papers", [])
    paper_ids = [p.paper_id for p in papers]
    if not paper_ids:
        return {"venue_trends": []}

    with cursor() as cur:
        cur.execute(
            """
            SELECT venue,
                   count(*) AS n,
                   count(*) FILTER (WHERE decision LIKE 'accept%%') AS accepts
            FROM papers WHERE id = ANY(%s)
            GROUP BY venue ORDER BY n DESC
            """,
            (paper_ids,))
        trends = [VenueTrend(venue=r[0], paper_count=r[1], accept_count=r[2],
                             accept_rate=round(r[2] / r[1], 3) if r[1] else 0.0)
                  for r in cur.fetchall()]

        # 재투고 흐름: 유사 논문이 한쪽 끝인 링크를 venue쌍으로 집계
        cur.execute(
            """
            SELECT e.venue AS from_v, l.venue AS to_v, count(*) AS n
            FROM submission_links sl
            JOIN papers e ON e.id = sl.earlier_paper_id
            JOIN papers l ON l.id = sl.later_paper_id
            WHERE sl.earlier_paper_id = ANY(%s) OR sl.later_paper_id = ANY(%s)
            GROUP BY e.venue, l.venue ORDER BY n DESC
            """,
            (paper_ids, paper_ids))
        flows = [ResubmissionFlow(from_venue=r[0], to_venue=r[1], count=r[2])
                 for r in cur.fetchall()]
    return {"venue_trends": trends, "resubmission_flows": flows}


# ---------------------------------------------------- synthesis (LLM)
def synthesis_node(state: PipelineState, embedder, llm) -> dict:
    """세 분석을 종합해 Report 조립 + 마크다운 요약 생성."""
    papers = state.get("similar_papers", [])
    tags = state.get("similarity_tags", {})
    patterns = state.get("review_patterns", [])
    trends = state.get("venue_trends", [])

    similar = [
        SimilarPaper(
            paper_id=p.paper_id, openreview_id=p.openreview_id, title=p.title,
            venue=p.venue, year=p.year, decision=p.decision,
            similarity_percentile=round(p.similarity_percentile or 0.0, 1),
            rank=i + 1, tags=tags.get(p.paper_id, []))
        for i, p in enumerate(papers)
    ]

    report = Report(
        query_title=state["query_title"],
        query_abstract=state.get("query_abstract", ""),
        similar_papers=similar,
        review_patterns=patterns,
        venue_trends=trends,
        resubmission_flows=state.get("resubmission_flows", []),
        summary_markdown=_summary(state, similar, patterns, trends, llm),
    )
    return {"report": report}


def _summary(state, similar, patterns, trends, llm) -> str:
    if llm is None:
        # 스텁: 구조화 데이터로 결정론적 요약
        lines = [f"## 유사 논문 {len(similar)}편"]
        if patterns:
            top = patterns[0]
            lines.append(
                f"- 반복 지적: \"{top.label[:60]}\" "
                f"({top.paper_count}/{top.total_papers}편)")
        if trends:
            lines.append("- 게재 경향: " + ", ".join(
                f"{t.venue} {t.accept_count}/{t.paper_count}" for t in trends[:3]))
        return "\n".join(lines)

    facts = {
        "query": state["query_title"],
        "similar_papers": [{"title": s.title, "venue": s.venue,
                            "decision": s.decision} for s in similar[:10]],
        "review_patterns": [{"label": p.label, "count": f"{p.paper_count}/{p.total_papers}"}
                            for p in patterns],
        "venue_trends": [{"venue": t.venue, "accept": f"{t.accept_count}/{t.paper_count}"}
                         for t in trends],
    }
    system = (
        "You are a research assistant. Given structured findings about papers similar "
        "to a query, write a concise Korean markdown briefing (under 250 words) covering: "
        "what similar work exists, what review criticisms recur, and where such papers "
        "get published. Be concrete; cite the counts. Do not invent facts.")
    return llm.text(SONNET, system, json.dumps(facts, ensure_ascii=False),
                    max_tokens=1200)
