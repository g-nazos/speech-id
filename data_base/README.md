# VoxCeleb PostgreSQL Database for Speech-ID

This folder contains the PostgreSQL database setup used to store VoxCeleb speaker embeddings, speaker metadata, and derived centroids.

## Contents

- `docker-compose.yml` - starts a local PostgreSQL instance with `pgvector` enabled.
- `init.sql` - creates the database schema and indexes.
- `populate_db.py` - loads PyTorch-saved embeddings and speaker metadata into the database.
- `.env` - local database and file path configuration.

## Database Components

### 1. `speakers`
Stores speaker-level metadata.

Columns:
- `speaker_id VARCHAR(20) PRIMARY KEY` - unique speaker identifier.
- `gender VARCHAR(10) NOT NULL` - speaker gender label.
- `nationality VARCHAR(20) NOT NULL` - speaker nationality label.

Indexes:
- Composite index `idx_speakers_gender_nat` on `(gender, nationality)` to speed queries by both fields.

### 2. `audio_embeddings`
Stores audio embeddings for individual speech files.

Columns:
- `id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY` - embedding row identifier.
- `speaker_id VARCHAR(20)` - foreign key to `speakers(speaker_id)`.
- `file_path TEXT NOT NULL` - original audio file or processed file path.
- `cluster_id INT` - optional cluster assignment, references `cluster_centroids(cluster_id)`.
- `embedding VECTOR(192) NOT NULL` - 192-dimensional speaker embedding vector.

Indexes:
- `idx_audio_embeddings_speaker` on `speaker_id` for faster joins.
- `idx_audio_embeddings_cluster` on `cluster_id` for cluster-based lookups.
- `idx_global_vector_hnsw` is an HNSW vector index on `embedding` using cosine similarity.

### 3. `metadata_centroids`
Stores centroid vectors aggregated from speaker metadata.

Columns:
- `category VARCHAR(20) NOT NULL` - metadata category, e.g. `gender` or `gender_nationality`.
- `category_value VARCHAR(50) NOT NULL` - category label or composite label, e.g. `male`, `female`, `male:US`.
- `centroid_vector VECTOR(192) NOT NULL` - averaged embedding for the category.

Usage:
- Used to compute metadata-level centroids for downstream analysis or similarity searches.

### 4. `cluster_centroids`
Reserved for unsupervised cluster centroids.

Columns:
- `cluster_id INT PRIMARY KEY`
- `centroid_vector VECTOR(192) NOT NULL`

Notes:
- This table is defined for future clustering phases and is not currently populated by `populate_db.py`.

## How to Run

1. Copy or update `.env` with your local settings.

```ini
DB_USER=postgres
DB_PASSWORD=<your-password>
DB_NAME=voxceleb_speaker_id
DB_HOST=localhost
DB_PORT=5432
PT_FILE_PATH=data_base/data/voxceleb_embeddings.pt
METADATA_CSV_PATH=data_base/data/vox1_meta.csv
```

2. Start the database:

```bash
cd data_base
docker-compose up -d
```

3. Populate the database:

```bash
python populate_db.py
```

## Notes and Recommendations

- `populate_db.py` reads the `.pt` file and the metadata CSV, inserts unique speakers, and loads embeddings into `audio_embeddings`.
- The current `metadata_centroids` population logic builds a hierarchical centroid tree for gender and gender+nationality categories.
- The `cluster_centroids` table exists for future cluster-based queries.
- Paths stored in `file_path` should be reviewed if you want portability across machines.

## File Paths and Configuration

- `PT_FILE_PATH` should point to the serialized embedding file created by `scripts/embeding_extractor.py`.
- `METADATA_CSV_PATH` should point to the VoxCeleb metadata CSV file used to map speaker IDs to gender/nationality.

## Troubleshooting

- Ensure the PostgreSQL container is running before `populate_db.py` connects.
- Verify `.env` values are correct and that `psycopg` and `python-dotenv` are installed.
- If the `.pt` file or metadata CSV cannot be found, `populate_db.py` will emit errors and terminate early.
