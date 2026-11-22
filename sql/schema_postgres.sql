-- Esquema PostgreSQL / SUPABASE do TCC. Cole no "SQL Editor" do Supabase e execute uma vez.
-- Só entram dados JÁ PSEUDONIMIZADOS. RLS ligado e sem política pública: apenas a chave de
-- serviço / string de conexão do projeto acessa (a chave "anon" não lê nada).

CREATE TABLE IF NOT EXISTS leads (
    lead_id     TEXT PRIMARY KEY,
    first_seen  TIMESTAMPTZ,
    last_seen   TIMESTAMPTZ,
    source      TEXT DEFAULT 'export_txt',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id     TEXT PRIMARY KEY,
    lead_id             TEXT NOT NULL REFERENCES leads(lead_id),
    seq_in_lead         INTEGER NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ NOT NULL,
    n_messages          INTEGER NOT NULL,
    n_lead_messages     INTEGER NOT NULL,
    n_agent_messages    INTEGER NOT NULL,
    first_response_sec  DOUBLE PRECISION,
    median_response_sec DOUBLE PRECISION,
    UNIQUE (lead_id, seq_in_lead)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    lead_id         TEXT NOT NULL REFERENCES leads(lead_id),
    ts              TIMESTAMPTZ NOT NULL,
    sender_role     TEXT NOT NULL CHECK (sender_role IN ('lead','atendente','sistema')),
    text            TEXT NOT NULL,
    is_media        BOOLEAN DEFAULT FALSE,
    n_redactions    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, ts);

-- ===== TABELA DE ROTULAGEM MANUAL =====
-- Uma linha por conversa e por anotador. Preencha direto no Table Editor do Supabase
-- (parece uma planilha) ou importe CSV. Valores permitidos garantidos por CHECK.
CREATE TABLE IF NOT EXISTS rotulagem (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    annotator       TEXT NOT NULL,                     -- 'a1', 'a2' ou 'adjudicado'
    escopo          TEXT CHECK (escopo IN ('em_escopo','fora_de_escopo')),
    -- taxonomia v1.1: o anotador preenche o FUNIL; o binário é DERIVADO (coluna gerada, não editável)
    funil           TEXT CHECK (funil IN ('sem_dialogo','informativo','intencao','agendado')),
    qualificacao    TEXT GENERATED ALWAYS AS (
        CASE funil WHEN 'intencao' THEN 'qualificado' WHEN 'agendado' THEN 'qualificado'
                   WHEN 'sem_dialogo' THEN 'nao_qualificado' WHEN 'informativo' THEN 'nao_qualificado' END
    ) STORED,
    objecao         TEXT[] DEFAULT '{}'
        CHECK (objecao <@ ARRAY['preco','horario_disponibilidade','localizacao','confianca_resultado',
                                'convenio_reembolso','adiamento','concorrencia','servico_nao_oferecido']::TEXT[]),
    gargalo         TEXT[] DEFAULT '{}'
        CHECK (gargalo <@ ARRAY['demora_resposta','sem_followup','info_incompleta','sem_vaga']::TEXT[]),
    desfecho        TEXT CHECK (desfecho IN ('desmarcou')),          -- metadado opcional
    para_terceiro   TEXT CHECK (para_terceiro IN ('sim')),           -- metadado opcional
    observacao      TEXT,
    status          TEXT NOT NULL DEFAULT 'pendente' CHECK (status IN ('pendente','rotulada','duvida')),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (conversation_id, annotator),
    -- fora de escopo não tem funil (logo, nem qualificação)
    CHECK (escopo IS DISTINCT FROM 'fora_de_escopo' OR funil IS NULL),
    -- 'desmarcou' só faz sentido para quem chegou a agendar
    CHECK (desfecho IS NULL OR funil = 'agendado'),
    -- rotulada em escopo exige funil
    CHECK (status <> 'rotulada' OR escopo IS DISTINCT FROM 'em_escopo' OR funil IS NOT NULL)
);

-- Conversa já montada como texto, para ler e rotular sem abrir mensagem por mensagem.
CREATE OR REPLACE VIEW v_conversas_texto AS
SELECT c.conversation_id, c.lead_id, c.seq_in_lead, c.started_at, c.n_messages,
       round((c.first_response_sec/60.0)::numeric, 1) AS primeira_resposta_min,
       string_agg(CASE m.sender_role WHEN 'lead' THEN 'Cliente: ' WHEN 'atendente' THEN 'Atendente: ' ELSE 'Sistema: ' END
                  || m.text, E'\n' ORDER BY m.ts) AS texto
FROM conversations c JOIN messages m USING (conversation_id)
GROUP BY c.conversation_id;

-- Fila de trabalho: conversas ainda sem rótulo do anotador (troque 'a1' pelo seu código).
CREATE OR REPLACE VIEW v_fila_a1 AS
SELECT v.* FROM v_conversas_texto v
WHERE NOT EXISTS (SELECT 1 FROM rotulagem r WHERE r.conversation_id = v.conversation_id
                  AND r.annotator = 'a1' AND r.status = 'rotulada');

-- Segurança: bloqueia acesso público (chave anon). O pipeline usa a conexão direta do projeto.
ALTER TABLE leads ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE rotulagem ENABLE ROW LEVEL SECURITY;
