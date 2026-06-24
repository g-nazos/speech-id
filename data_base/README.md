# VoxCeleb PostgreSQL Database for Speech-ID

This folder contains the local PostgreSQL database setup used by the Speech-ID pipeline.
It stores VoxCeleb speaker embeddings, speaker metadata, computed metadata centroids, and unsupervised cluster centroids.

## Multi-model architecture

The pipeline compares several embedding models, and their vectors have different
dimensions. To keep vectors of different sizes out of the same column, **each
model lives in its own PostgreSQL schema** with an identical set of tables. The
registry in `configs/models.py` defines them:

| model           | schema          | dim | extractor head |
|-----------------|-----------------|-----|----------------|
| `ecapa`         | `public`        | 192 | ecapa          |
| `wavlm`         | `wavlm`         | 768 | meanpool       |
| `wavlm_xvector` | `wavlm_xvector` | 512 | xvector        |

`ecapa` is the original, already-populated model, so it reuses the default
`public` schema untouched. New models get their own schemas. Queries run with
`SET search_path TO <schema>, public`, so the same unqualified SQL serves every
model.

## Contents

- `docker-compose.yml` — starts a PostgreSQL container with `pgvector` enabled.
- `init.sql` — Docker entrypoint init: enables the `pgvector` extension only (runs once on an empty volume).
- `schema.template.sql` — per-model DDL template; `{schema}` and `{dim}` are filled in at populate time to create each model's tables/indexes.
- `configs/models.py` — registry of models (schema, dimension, source, extractor head, `.pt` files).
- `populate_db.py` — ingests one model's embeddings, computes metadata centroids, and runs clustering.
- `schema_overview.py` — reads the live database and generates a Markdown schema/contents overview.
- `schema.dbml` — dbdiagram.io diagram of all model schemas.
- `.env` — database credentials and source file names.

## Database Schema

The tables below are created **per model schema** by `ensure_schema()` from
`schema.template.sql`. `VECTOR(dim)` is templated by the model's dimension
(192 / 768 / 512); the examples show the `ecapa`/`public` case (192).

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

The script populates **one model's schema per run**, selected with `--model`
(see `configs/models.py`). A bare run with no `--model` is rejected on purpose,
so it can never accidentally rebuild the existing `public`/`ecapa` schema. It
first calls `ensure_schema(spec.schema, spec.dim)` to create the schema, tables,
and indexes (idempotent, templated by the model's dimension), then runs three
phases:

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
# Populate a specific model's schema (--model is required):
uv run populate_db.py --model ecapa
uv run populate_db.py --model wavlm
uv run populate_db.py --model wavlm_xvector
```

Generate a read-only Markdown overview of the live database schema and table contents:

```bash
uv run python schema_overview.py --output-file ../results/database_schema_overview.md
```

Omit `--output-file` to print the report only to the terminal.

## Evaluation

The speaker search evaluation code lives in [evaluate/](../evaluate/README.md).

Run it from the project root with:

```bash
python evaluate/evaluate_search.py
```

Use `uv run` (or the project's `.venv` interpreter) so dependencies like
`torch` are available — a bare `python` on your PATH may not have them.

## Notes

- `init.sql` only enables the `vector` extension. The per-model tables/indexes are created on demand by `populate_db.py` from `schema.template.sql` (templated by each model's dimension).
- `cluster_centroids` is not just a placeholder; it is populated by the clustering phase.
- `metadata_centroids` is recomputed from the current contents of `audio_embeddings`.
- The current ingest path uses PostgreSQL binary COPY for `audio_embeddings`.

