"""하이브리드 검색: SPECTER2 벡터 + Postgres full-text, RRF로 결합.

왜 하이브리드인가 (§11.2 실측 근거):
SPECTER2의 코사인 유사도는 0.72~0.98의 좁은 구간에 압축되어 있어 절대 임계값을
쓸 수 없고, 'CIFAR-10', 'LoRA' 같은 고유명사 정확 매칭에도 약하다.
full-text 검색이 후자를 보완하고, RRF가 순위 기반이라 스케일 문제를 우회한다.

RRF(Reciprocal Rank Fusion): score = Σ 1/(k + rank), k=60이 표준 기본값.
점수의 절대 스케일을 무시하고 순위만 쓰기 때문에 이 상황에 정확히 들어맞는다.
"""
import logging
from dataclasses import dataclass

from paper_assistant.db.connection import cursor
from paper_assistant.embedding.specter2 import similarity_percentile

log = logging.getLogger(__name__)

RRF_K = 60
CANDIDATE_POOL = 50   # 각 검색기에서 가져올 후보 수


@dataclass
class SearchResult:
    paper_id: int
    openreview_id: str
    title: str
    abstract: str
    venue: str
    year: int
    decision: str
    rrf_score: float
    vector_rank: int | None
    fts_rank: int | None
    similarity_percentile: float | None
    """무작위 논문쌍 대비 백분위. 원시 코사인은 노출하지 않는다 (§11.2)."""


def _vector_search(cur, embedding, limit: int) -> list[tuple[int, float]]:
    """(paper_id, 코사인 유사도) 순위순. 벡터는 L2 정규화 상태로 저장돼 있다."""
    cur.execute(
        """
        SELECT id, 1 - (embedding <=> %s) AS cosine
        FROM papers
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (embedding, embedding, limit),
    )
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def _fulltext_search(cur, query_text: str, limit: int) -> list[int]:
    """ts_rank 상위 paper_id. 고유명사·기법명 정확 매칭을 보완한다.

    주의: plainto_tsquery는 모든 단어를 AND로 결합하므로 초록 전체(수백 단어)를
    넣으면 그 단어를 전부 가진 문서만 걸린다 — 실측 결과 200편 중 1편만 매칭되어
    FTS가 사실상 무력화됐다. 그래서 lexeme을 OR로 결합해 ts_rank가 겹치는 정도로
    순위를 매기게 한다 (BM25에 가까운 동작). 실측: 같은 조건에서 200/200편 매칭.
    """
    cur.execute(
        """
        WITH q AS (
            SELECT to_tsquery('english', string_agg(lexeme, ' | ')) AS query
            FROM unnest(to_tsvector('english', %s))
        )
        SELECT p.id
        FROM papers p, q
        WHERE q.query IS NOT NULL AND p.tsv @@ q.query
        ORDER BY ts_rank(p.tsv, q.query) DESC
        LIMIT %s
        """,
        (query_text, limit),
    )
    return [row[0] for row in cur.fetchall()]


def rrf_fuse(*rank_maps: dict[int, int], k: int = RRF_K) -> dict[int, float]:
    """여러 검색기의 순위를 RRF로 결합. score = Σ 1/(k + rank).

    점수의 절대 스케일을 쓰지 않고 순위만 쓰기 때문에, 스케일이 다른
    검색기(코사인 0.72~0.98 vs ts_rank 0~1)를 그대로 합칠 수 있다.
    """
    scores: dict[int, float] = {}
    for ranks in rank_maps:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _fetch_metadata(cur, paper_ids: list[int]) -> dict[int, tuple]:
    if not paper_ids:
        return {}
    cur.execute(
        """
        SELECT id, openreview_id, title, abstract, venue, year, decision
        FROM papers WHERE id = ANY(%s)
        """,
        (paper_ids,),
    )
    return {row[0]: row[1:] for row in cur.fetchall()}


def hybrid_search(embedding, query_text: str, top_k: int = 20,
                  pool: int = CANDIDATE_POOL) -> list[SearchResult]:
    """벡터 검색과 full-text 검색을 RRF로 결합해 상위 top_k편 반환.

    embedding: SPECTER2로 인코딩한 쿼리 벡터 (L2 정규화된 numpy/list)
    query_text: full-text 검색에 쓸 원문 (보통 제목 + 초록)
    """
    with cursor() as cur:
        vector_hits = _vector_search(cur, embedding, pool)
        fts_hits = _fulltext_search(cur, query_text, pool)

        vector_ranks = {pid: i + 1 for i, (pid, _) in enumerate(vector_hits)}
        cosines = dict(vector_hits)
        fts_ranks = {pid: i + 1 for i, pid in enumerate(fts_hits)}

        scores = rrf_fuse(vector_ranks, fts_ranks)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        meta = _fetch_metadata(cur, [pid for pid, _ in ranked])

    results = []
    for pid, score in ranked:
        if pid not in meta:
            continue
        openreview_id, title, abstract, venue, year, decision = meta[pid]
        cos = cosines.get(pid)
        results.append(SearchResult(
            paper_id=pid,
            openreview_id=openreview_id,
            title=title,
            abstract=abstract,
            venue=venue,
            year=year,
            decision=decision,
            rrf_score=score,
            vector_rank=vector_ranks.get(pid),
            fts_rank=fts_ranks.get(pid),
            similarity_percentile=similarity_percentile(cos) if cos is not None else None,
        ))
    return results
