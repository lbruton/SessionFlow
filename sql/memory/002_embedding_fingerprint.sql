-- The embedding fingerprint the memory vectors were built with: model revision,
-- tokenizer hash, prefixes, pooling, token limit, dimension and implementation
-- version. A difference warns; only a model or dimension change blocks writes.
ALTER TABLE memory_index_meta ADD COLUMN fingerprint jsonb;
