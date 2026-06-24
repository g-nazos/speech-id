-- Docker entrypoint init (runs once on an empty volume).
--
-- Only the pgvector extension is created here. The per-model tables/indexes are
-- created on demand by data_base/populate_db.py from data_base/schema.template.sql,
-- templated by each model's embedding dimension (see configs/models.py). Each
-- model lives in its own schema (e.g. ecapa, wavlm) so vectors of different
-- dimensions never share a column.
CREATE EXTENSION IF NOT EXISTS vector;
