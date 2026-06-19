CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  sources JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_messages_conv_idx
  ON chat_messages (conversation_id, created_at);

ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "chat lectura" ON chat_messages;
CREATE POLICY "chat lectura" ON chat_messages FOR SELECT USING (true);

DROP POLICY IF EXISTS "chat insercion" ON chat_messages;
CREATE POLICY "chat insercion" ON chat_messages FOR INSERT WITH CHECK (true);
