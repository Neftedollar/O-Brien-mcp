-- O-Brien Memory MCP — database schema
-- Run once against your PostgreSQL instance before first use.
--
-- Supported embedding models (set EMBEDDING_DIM to match):
--   nomic-embed-text  → 768  (default, recommended)
--   all-minilm        → 384
--   mxbai-embed-large → 1024
--
-- EMBEDDING_DIM must match the vector() size below.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content          TEXT NOT NULL,
    category         TEXT NOT NULL,
    tags             JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding        vector(768),          -- change to match your EMBEDDING_DIM
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_category    ON memories (category);
CREATE INDEX IF NOT EXISTS idx_memories_tags        ON memories USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_memories_created_at  ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_fts         ON memories USING GIN (to_tsvector('simple', content));
