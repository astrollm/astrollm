-- AstroLLM Database Initialization
-- Creates pgvector extension and core tables for RAG

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Paper metadata
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(20) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT,
    categories TEXT[],
    published_date DATE,
    updated_date DATE,
    citation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Paper chunks for RAG
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    section VARCHAR(50),  -- abstract, introduction, methods, results, conclusion
    content TEXT NOT NULL,
    token_count INTEGER,
    chunk_index INTEGER,  -- Position within paper
    embedding vector(1024),  -- BGE-large dimension
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for efficient retrieval
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX idx_papers_arxiv_id ON papers(arxiv_id);
CREATE INDEX idx_papers_categories ON papers USING GIN(categories);
CREATE INDEX idx_papers_published ON papers(published_date DESC);

-- Full text search
ALTER TABLE chunks ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_chunks_tsv ON chunks USING GIN(tsv);

-- Evaluation results tracking
CREATE TABLE eval_results (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    checkpoint_path TEXT,
    wandb_run_id VARCHAR(50),
    benchmark VARCHAR(50) NOT NULL,  -- astrolab-1, astro-qa, pedagogy, tool-use
    score NUMERIC(6,3),
    details JSONB,
    evaluated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eval_model ON eval_results(model_name, benchmark);
