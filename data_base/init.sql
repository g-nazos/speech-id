-- 1. Enable pgvector Extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Speakers Table
CREATE TABLE IF NOT EXISTS speakers (
    speaker_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    nationality VARCHAR(20) NOT NULL
);

-- 3. Create Metadata Centroids Table (For Phase 2)
-- Stores the calculated average vectors for Genders and Nationalities
CREATE TABLE IF NOT EXISTS metadata_centroids (
    category VARCHAR(20) NOT NULL, -- e.g., 'gender' or 'nationality'
    category_value VARCHAR(50) NOT NULL, -- e.g., 'M', 'F', 'US', 'UK'
    centroid_vector VECTOR(192) NOT NULL,
    PRIMARY KEY (category, category_value)
);

-- 4. Create Unsupervised Cluster Centroids Table (For Phase 3)
CREATE TABLE IF NOT EXISTS cluster_centroids (
    cluster_id INT PRIMARY KEY,
    centroid_vector VECTOR(192) NOT NULL
);

-- 5. Create Audio Embeddings Table
CREATE TABLE IF NOT EXISTS audio_embeddings (
    -- Best Practice: Use IDENTITY instead of SERIAL
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, 
    speaker_id VARCHAR(20) REFERENCES speakers(speaker_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    cluster_id INT REFERENCES cluster_centroids(cluster_id) ON DELETE SET NULL,
    embedding VECTOR(192) NOT NULL
);

-- 6. Build Relational Indexes
CREATE INDEX IF NOT EXISTS idx_speakers_gender_nat ON speakers(gender, nationality);
CREATE INDEX IF NOT EXISTS idx_audio_embeddings_cluster ON audio_embeddings(cluster_id);
-- Good practice: Index the FK to speed up joins between embeddings and speakers
CREATE INDEX IF NOT EXISTS idx_audio_embeddings_speaker ON audio_embeddings(speaker_id); 

-- 7. Build Vector Indexes (Careful with your baseline!)
-- IMPORTANT: To run your Phase 1 Brute-Force, run 'SET enable_indexscan = off;' in your session first.
-- Best Practice: Define 'm' and 'ef_construction' for better recall tuning
CREATE INDEX IF NOT EXISTS idx_global_vector_hnsw 
ON audio_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Note: Intentionally REMOVED the HNSW index on cluster_centroids. 
-- Exact search (Cosine Similarity) is faster and 100% accurate for small tables (< 10,000 rows).