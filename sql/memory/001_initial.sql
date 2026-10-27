-- Canonical memory store: current records, version history, per-index
-- acknowledgements, the memory embedding generation and maintenance runs.
-- Names stay unqualified so the connection's search_path picks the schema.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- Orders the repair loop and gives rebuild a cursor. Values do not follow
-- commit order, so delivery relies on memory_index_acks, not on this sequence.
CREATE SEQUENCE memory_change_seq;

CREATE TABLE memories (
    memory_id            uuid PRIMARY KEY,
    user_id              text NOT NULL,
    agent_id             text NOT NULL,
    project              text NOT NULL,
    type                 text NOT NULL,
    status               text NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active', 'superseded')),
    -- An empty topic key is stored as NULL, so it never takes part in uniqueness.
    topic_key            text CHECK (topic_key <> ''),
    content_hash         text NOT NULL,
    text                 text NOT NULL,
    extra                jsonb NOT NULL DEFAULT '{}'::jsonb,
    issue_ids            text[] NOT NULL DEFAULT '{}',
    supersedes           uuid[] NOT NULL DEFAULT '{}',
    source_session_id    text,
    source_provider      text,
    source_project_root  text,
    source_started_at    timestamptz,
    source_ended_at      timestamptz,
    created_at           timestamptz NOT NULL,
    updated_at           timestamptz NOT NULL,
    -- Deletes keep the row as a tombstone so no replay or rebuild revives it.
    deleted_at           timestamptz,
    version              integer NOT NULL DEFAULT 1 CHECK (version > 0),
    change_seq           bigint NOT NULL DEFAULT nextval('memory_change_seq')
);

ALTER SEQUENCE memory_change_seq OWNED BY memories.change_seq;

CREATE UNIQUE INDEX memories_content_hash_uq
    ON memories (user_id, project, content_hash)
    WHERE deleted_at IS NULL;

CREATE UNIQUE INDEX memories_topic_key_uq
    ON memories (user_id, project, topic_key)
    WHERE topic_key IS NOT NULL AND deleted_at IS NULL;

CREATE INDEX memories_listing_idx
    ON memories (user_id, project, type, status, created_at DESC, memory_id DESC)
    WHERE deleted_at IS NULL;

CREATE INDEX memories_source_session_idx
    ON memories (user_id, project, source_session_id);

CREATE INDEX memories_issue_ids_idx ON memories USING gin (issue_ids);

CREATE INDEX memories_mem0_ids_idx ON memories USING gin ((extra -> 'mem0_ids'));

CREATE INDEX memories_change_seq_idx ON memories (change_seq);

-- The full row after every write, tombstones included.
CREATE TABLE memory_versions (
    memory_id            uuid NOT NULL REFERENCES memories (memory_id),
    version              integer NOT NULL,
    user_id              text NOT NULL,
    agent_id             text NOT NULL,
    project              text NOT NULL,
    type                 text NOT NULL,
    status               text NOT NULL,
    topic_key            text,
    content_hash         text NOT NULL,
    text                 text NOT NULL,
    extra                jsonb NOT NULL,
    issue_ids            text[] NOT NULL,
    supersedes           uuid[] NOT NULL,
    source_session_id    text,
    source_provider      text,
    source_project_root  text,
    source_started_at    timestamptz,
    source_ended_at      timestamptz,
    created_at           timestamptz NOT NULL,
    updated_at           timestamptz NOT NULL,
    deleted_at           timestamptz,
    change_seq           bigint NOT NULL,
    written_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, version)
);

-- One row per memory per index consumer ('milvus', 'fts:<host>'). A memory is
-- pending for a consumer while its row is missing or differs from the canonical
-- version or generation.
CREATE TABLE memory_index_acks (
    consumer    text NOT NULL,
    memory_id   uuid NOT NULL REFERENCES memories (memory_id),
    version     integer NOT NULL,
    generation  integer NOT NULL,
    deleted     boolean NOT NULL,
    acked_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (consumer, memory_id)
);

-- Exactly one row. The server fills the model name and dimension; a re-embed
-- bumps the generation, which makes every older Milvus acknowledgement pending.
CREATE TABLE memory_index_meta (
    singleton   boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    model_name  text,
    vector_dim  integer,
    generation  integer NOT NULL DEFAULT 1 CHECK (generation > 0)
);

INSERT INTO memory_index_meta (singleton) VALUES (true);

-- Nightly dump attempts and results. An attempt row is written before the
-- dump and a result row after it, so a run that dies leaves a visible attempt.
CREATE TABLE maintenance_runs (
    run_id         bigserial PRIMARY KEY,
    job            text NOT NULL,
    phase          text NOT NULL CHECK (phase IN ('attempt', 'result')),
    scheduled_for  timestamptz NOT NULL,
    started_at     timestamptz NOT NULL,
    finished_at    timestamptz,
    ok             boolean,
    detail         text
);

CREATE INDEX maintenance_runs_job_idx ON maintenance_runs (job, started_at DESC);
