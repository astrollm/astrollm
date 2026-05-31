-- Migration 001: chunks.embedding vector(1024) -> vector(384)
-- (BGE-large-en-v1.5 -> BGE-small-en-v1.5, see docker/init.sql + docker-compose.yml embedder)
--
-- WHY THIS EXISTS: init.sql only runs on a FRESH Postgres volume. A dev who already has
-- the `pgdata` volume from the old 1024-dim schema will keep vector(1024) while the embedder
-- now emits 384-dim vectors, so indexing/search would fail with a dimension mismatch.
--
-- Run once against an existing dev database, e.g.:
--   docker exec -i <db-container> psql -U astrollm -d astrollm < docker/migrations/001_chunks_embedding_to_384.sql
--
-- For a throwaway dev DB it is simpler to just reset the volume instead of migrating:
--   docker compose -f docker/docker-compose.yml down -v && docker compose -f docker/docker-compose.yml up -d
--
-- Existing 1024-dim embeddings cannot be cast to 384 and must be regenerated with the new
-- model anyway, so this clears `chunks` (paper metadata in `papers` is preserved). Re-run
-- your indexing step afterwards to repopulate embeddings.

BEGIN;
DROP INDEX IF EXISTS idx_chunks_embedding;
TRUNCATE chunks;
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 256);
COMMIT;
