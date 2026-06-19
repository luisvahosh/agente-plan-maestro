-- ============================================================
-- Crear índice vectorial IVFFLAT
-- Ejecutar en Supabase → SQL Editor DESPUÉS de terminar la ingesta
-- (el índice requiere datos para construirse)
-- ============================================================

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
