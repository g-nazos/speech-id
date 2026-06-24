-- Per-model schema template.
--
-- Rendered by data_base/populate_db.py with:
--   {schema} -> Postgres schema isolating one embedding model's tables
--   {dim}    -> embedding dimension, i.e. the pgvector VECTOR(...) size
--
-- Each model (see configs/models.py) gets its own schema so that vectors of
-- different dimensions never share a column. Searches run with
-- `SET search_path TO {schema}` so unqualified queries resolve here.
-- Index names live inside their table's schema, so the same names coexist
-- across model schemas.

CREATE SCHEMA IF NOT EXISTS {schema};

-- Speakers (metadata) ------------------------------------------------------
CREATE TABLE IF NOT EXISTS {schema}.speakers (
    speaker_id  VARCHAR(20) PRIMARY KEY,
    gender      VARCHAR(10) NOT NULL,
    nationality VARCHAR(20) NOT NULL
);

-- Supervised metadata centroids (Phase 2) ----------------------------------
CREATE TABLE IF NOT EXISTS {schema}.metadata_centroids (
    category        VARCHAR(20)   NOT NULL,
    category_value  VARCHAR(50)   NOT NULL,
    centroid_vector VECTOR({dim}) NOT NULL,
    PRIMARY KEY (category, category_value)
);

-- Unsupervised cluster centroids (Phase 3) ---------------------------------
CREATE TABLE IF NOT EXISTS {schema}.cluster_centroids (
    cluster_id      INT PRIMARY KEY,
    centroid_vector VECTOR({dim}) NOT NULL
);

-- Audio embeddings (one row per utterance) ---------------------------------
CREATE TABLE IF NOT EXISTS {schema}.audio_embeddings (
    id         INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    speaker_id VARCHAR(20) REFERENCES {schema}.speakers(speaker_id) ON DELETE CASCADE,
    file_path  TEXT NOT NULL,
    cluster_id INT REFERENCES {schema}.cluster_centroids(cluster_id) ON DELETE SET NULL,
    embedding  VECTOR({dim}) NOT NULL
);

-- Relational indexes -------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_speakers_gender_nat
    ON {schema}.speakers(gender, nationality);
CREATE INDEX IF NOT EXISTS idx_audio_embeddings_cluster
    ON {schema}.audio_embeddings(cluster_id);
CREATE INDEX IF NOT EXISTS idx_audio_embeddings_speaker
    ON {schema}.audio_embeddings(speaker_id);

-- Vector index (HNSW, cosine). For an exact brute-force baseline, disable
-- index scans in-session: SET enable_indexscan = off;
CREATE INDEX IF NOT EXISTS idx_global_vector_hnsw
    ON {schema}.audio_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
