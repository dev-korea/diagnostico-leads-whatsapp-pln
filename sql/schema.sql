-- Esquema do banco do TCC (SQLite; compatível com PostgreSQL/Supabase trocando
-- INTEGER PRIMARY KEY AUTOINCREMENT por BIGSERIAL e TEXT de datas por TIMESTAMPTZ).
-- Todos os dados pessoais já chegam pseudonimizados: nunca armazenar telefone/nome reais.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS leads (
    lead_id        TEXT PRIMARY KEY,          -- HMAC-SHA256 do identificador (telefone/título do chat)
    first_seen     TEXT,                      -- ISO 8601
    last_seen      TEXT,
    source         TEXT DEFAULT 'export_txt', -- origem do dado
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Um arquivo exportado = uma carga; permite rastrear/reverter ingestões.
CREATE TABLE IF NOT EXISTS imports (
    import_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_sha256    TEXT UNIQUE NOT NULL,      -- idempotência por arquivo
    lead_id        TEXT NOT NULL REFERENCES leads(lead_id),
    device_format  TEXT,                      -- android | iphone
    n_messages     INTEGER,
    imported_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,         -- lead_id + índice sequencial
    lead_id         TEXT NOT NULL REFERENCES leads(lead_id),
    seq_in_lead     INTEGER NOT NULL,         -- 1 = primeiro contato; >1 = recorrência
    started_at      TEXT NOT NULL,
    ended_at        TEXT NOT NULL,
    n_messages      INTEGER NOT NULL,
    n_lead_messages INTEGER NOT NULL,
    n_agent_messages INTEGER NOT NULL,
    first_response_sec REAL,                  -- tempo até 1ª resposta do atendente (gargalo)
    median_response_sec REAL,
    UNIQUE (lead_id, seq_in_lead)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,         -- SHA256(lead_id|timestamp|sender_role|texto)
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    lead_id         TEXT NOT NULL REFERENCES leads(lead_id),
    ts              TEXT NOT NULL,            -- ISO 8601 local
    sender_role     TEXT NOT NULL CHECK (sender_role IN ('lead','atendente','sistema')),
    text            TEXT NOT NULL,            -- já pseudonimizado
    is_media        INTEGER DEFAULT 0,
    n_redactions    INTEGER DEFAULT 0         -- quantos trechos foram mascarados
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, ts);

-- Rótulos de referência (ground truth). Uma linha por anotador/conversa/dimensão/valor.
CREATE TABLE IF NOT EXISTS labels (
    label_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    annotator       TEXT NOT NULL,            -- 'a1', 'a2', 'adjudicado'
    dimension       TEXT NOT NULL CHECK (dimension IN ('escopo','funil','qualificacao','objecao','gargalo','desfecho','para_terceiro')),
    value           TEXT NOT NULL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (conversation_id, annotator, dimension, value)
);

-- Divisão treino/validação/teste fixa e versionada (sempre por lead, evita vazamento).
CREATE TABLE IF NOT EXISTS splits (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    split_name      TEXT NOT NULL,            -- ex.: 'v1'
    part            TEXT NOT NULL CHECK (part IN ('train','val','test')),
    PRIMARY KEY (conversation_id, split_name)
);

-- Previsões de cada método (reprodutibilidade e testes pareados).
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    method          TEXT NOT NULL,            -- tfidf_lr | llm_fewshot | bertimbau
    model_version   TEXT,
    seed            INTEGER,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    dimension       TEXT NOT NULL,
    value           TEXT NOT NULL,
    score           REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id, method);

-- Léxico de escopo (Etapa 1 do pipeline): termos que indicam assunto do catálogo do cliente.
CREATE TABLE IF NOT EXISTS catalog_terms (
    term_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_item    TEXT NOT NULL,            -- ex.: 'avaliacao_quiropraxica'
    term            TEXT NOT NULL,            -- termo/expressão como o cliente escreve
    term_normalized TEXT NOT NULL,            -- minúsculas, sem acento/pontuação
    UNIQUE (catalog_item, term_normalized)
);
