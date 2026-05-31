# VoxCeleb PostgreSQL Database for Speech-ID

This folder contains the local PostgreSQL database setup used by the Speech-ID pipeline.
It stores VoxCeleb speaker embeddings, speaker metadata, computed metadata centroids, and unsupervised cluster centroids.

## Contents

- `docker-compose.yml` — starts a PostgreSQL container with `pgvector` enabled.
- `init.sql` — creates the database schema, tables, and indexes.
- `populate_db.py` — ingests embeddings, computes metadata centroids, and runs clustering.
- `.env` — database credentials and source file names.

## Database Schema

### `speakers`
Stores speaker metadata.

Columns:
- `speaker_id VARCHAR(20) PRIMARY KEY`
- `gender VARCHAR(10) NOT NULL`
- `nationality VARCHAR(20) NOT NULL`

Indexes:
- `idx_speakers_gender_nat` on `(gender, nationality)`

### `audio_embeddings`
Stores individual audio embedding records.

Columns:
- `id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY`
- `speaker_id VARCHAR(20)` — references `speakers(speaker_id)`
- `file_path TEXT NOT NULL`
- `cluster_id INT` — references `cluster_centroids(cluster_id)`
- `embedding VECTOR(192) NOT NULL`

Indexes:
- `idx_audio_embeddings_speaker` on `speaker_id`
- `idx_audio_embeddings_cluster` on `cluster_id`
- `idx_global_vector_hnsw` on `embedding` using `hnsw (embedding vector_cosine_ops)`

### `metadata_centroids`
Stores averaged embedding centroids derived from speaker metadata.

Columns:
- `category VARCHAR(20) NOT NULL`
- `category_value VARCHAR(50) NOT NULL`
- `centroid_vector VECTOR(192) NOT NULL`

Primary key:
- `(category, category_value)`

Current use:
- Computed from `audio_embeddings` grouped by `gender`
  and by `gender:nationality`.

### `cluster_centroids`
Stores unsupervised cluster centroids.

Columns:
- `cluster_id INT PRIMARY KEY`
- `centroid_vector VECTOR(192) NOT NULL`

Current use:
- Populated by `build_unsupervised_clusters()` in `populate_db.py`.
- Used to assign `audio_embeddings.cluster_id`.

## How `populate_db.py` works

The script has three phases:

1. `load_data_to_postgres()`
   - Loads the `.pt` embeddings file and VoxCeleb metadata CSV.
   - Inserts unique speakers into `speakers`.
   - Writes audio embeddings into `audio_embeddings` using PostgreSQL binary COPY.
   - Calls `compute_and_save_centroids(cur)` to truncate and rebuild `metadata_centroids`.
     This function computes gender-level and gender+nationality centroids from `audio_embeddings`.

2. `fetch_embeddings()`
   - Reads committed embeddings from `audio_embeddings`.
   - Converts them into NumPy arrays for clustering.

3. `build_unsupervised_clusters(num_clusters=256)`
   - Normalizes embeddings for cosine-style geometry.
   - Runs `MiniBatchKMeans`.
   - Deletes old rows from `cluster_centroids`.
   - Inserts new cluster centroids.
   - Updates `audio_embeddings.cluster_id`.

When run as a script, `__main__` executes both ingestion and clustering.

## Configuration

Edit `data_base/.env` for your local environment.

Example:

```ini
DB_USER=user
DB_PASSWORD=password
DB_NAME= db
DB_HOST=localhost
DB_PORT=8023
PT_FILE_NAME=voxceleb_embeddings.pt
METADATA_CSV_NAME=vox1_meta.csv
```

`PT_FILE_NAME` and `METADATA_CSV_NAME` are just file names, not full paths.
The script resolves them against `data_base/data/` relative to `populate_db.py`,
so it works regardless of your current working directory and on Windows, Linux,
and macOS. Place the `.pt` embeddings file and `vox1_meta.csv` in `data_base/data/`.



## How to run

```bash
cd data_base
docker-compose up -d
uv run populate_db.py
```

Use `uv run` (or the project's `.venv` interpreter) so dependencies like
`torch` are available — a bare `python` on your PATH may not have them.

## Notes

- `init.sql` enables the `vector` extension and creates the complete schema used by the script.
- `cluster_centroids` is not just a placeholder; it is populated by the clustering phase.
- `metadata_centroids` is recomputed from the current contents of `audio_embeddings`.
- The current ingest path uses PostgreSQL binary COPY for `audio_embeddings`.

