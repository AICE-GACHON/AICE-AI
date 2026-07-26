"""코퍼스 전체 aspect base rate를 계산해 `aspect_base_rates`에 적재한다.

리뷰 패턴의 lift 계산에 쓰인다 (설계서 §18). 수집이 끝난 뒤 1회 실행하고,
이후 데이터가 크게 늘면 다시 돌리면 된다. 96만 건 집계라 수 초 걸린다.

    $env:PYTHONPATH="."; python scripts/build_base_rates.py
"""
import logging

from paper_assistant.db.connection import cursor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DDL = """
CREATE TABLE IF NOT EXISTS aspect_base_rates (
    aspect       TEXT NOT NULL,
    sentiment    TEXT NOT NULL,
    paper_count  BIGINT NOT NULL,
    total_papers BIGINT NOT NULL,
    base_rate    REAL   NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (aspect, sentiment)
);
"""

# 분모는 "해당 sentiment의 지적항목이 하나라도 추출된 논문 수".
# 리뷰가 없거나 추출이 안 된 논문을 분모에 넣으면 base rate가 과소평가된다.
COMPUTE = """
INSERT INTO aspect_base_rates (aspect, sentiment, paper_count, total_papers, base_rate)
SELECT rp.aspect,
       %(sentiment)s,
       count(DISTINCT rp.paper_id),
       t.total,
       count(DISTINCT rp.paper_id)::real / NULLIF(t.total, 0)
FROM review_points rp
CROSS JOIN (
    SELECT count(DISTINCT paper_id) AS total
    FROM review_points WHERE sentiment = %(sentiment)s
) t
WHERE rp.sentiment = %(sentiment)s
GROUP BY rp.aspect, t.total
ON CONFLICT (aspect, sentiment) DO UPDATE
SET paper_count  = EXCLUDED.paper_count,
    total_papers = EXCLUDED.total_papers,
    base_rate    = EXCLUDED.base_rate,
    computed_at  = now();
"""


def main(sentiment: str = "weakness") -> None:
    with cursor() as cur:
        cur.execute(DDL)
        cur.execute(COMPUTE, {"sentiment": sentiment})
        cur.execute(
            "SELECT aspect, paper_count, total_papers, base_rate "
            "FROM aspect_base_rates WHERE sentiment = %s ORDER BY base_rate DESC",
            (sentiment,))
        rows = cur.fetchall()

    print(f"\n{'aspect':24} {'논문수':>8} {'분모':>8} {'base rate':>10}")
    print("-" * 54)
    for aspect, cnt, total, rate in rows:
        print(f"{aspect:24} {cnt:8,} {total:8,} {rate*100:9.1f}%")
    print(f"\n✅ {len(rows)}개 aspect 적재 완료 (sentiment={sentiment})")


if __name__ == "__main__":
    main()
