CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
  id BIGSERIAL PRIMARY KEY,
  chunk_id INTEGER UNIQUE NOT NULL,
  content TEXT NOT NULL,
  source TEXT NOT NULL,
  embedding vector(1024),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION match_document_chunks(
  query_embedding vector(1024),
  match_threshold float DEFAULT 0.0,
  match_count int DEFAULT 5
)
RETURNS TABLE(
  id bigint,
  chunk_id int,
  content text,
  source text,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    dc.id,
    dc.chunk_id,
    dc.content,
    dc.source,
    1 - (dc.embedding <=> query_embedding) AS similarity
  FROM document_chunks dc
  WHERE 1 - (dc.embedding <=> query_embedding) > match_threshold
  ORDER BY dc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Lectura publica" ON document_chunks;
CREATE POLICY "Lectura publica"
  ON document_chunks FOR SELECT
  USING (true);

DROP POLICY IF EXISTS "Insercion service role" ON document_chunks;
CREATE POLICY "Insercion service role"
  ON document_chunks FOR INSERT
  WITH CHECK (true);

DROP POLICY IF EXISTS "Actualizacion service role" ON document_chunks;
CREATE POLICY "Actualizacion service role"
  ON document_chunks FOR UPDATE
  USING (true);
