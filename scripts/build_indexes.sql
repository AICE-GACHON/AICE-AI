-- 벡터 인덱스 생성 — **전체 데이터 적재를 마친 뒤** 실행할 것.
-- 빈 테이블에 미리 만들면 적재 내내 인덱스 갱신 비용이 발생한다.
--
--   docker exec -i paper-assistant-db psql -U paper -d paper_assistant \
--       < scripts/build_indexes.sql
--
-- HNSW 파라미터: m=16, ef_construction=64 (pgvector 기본값).
-- 벡터를 L2 정규화해 저장하므로 코사인 연산자(vector_cosine_ops)를 쓴다.

SET maintenance_work_mem = '2GB';   -- 인덱스 생성 속도에 직접 영향

CREATE INDEX IF NOT EXISTS papers_embedding_hnsw
    ON papers USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS review_points_embedding_hnsw
    ON review_points USING hnsw (embedding vector_cosine_ops);

ANALYZE papers;
ANALYZE review_points;
