# speech-id

Speaker identification on the VoxCeleb dataset. Fixed-size speaker embeddings are
extracted from raw audio with **three models**, stored in PostgreSQL + `pgvector`,
and compared across several search strategies. The models are:

| model           | source                              | dim | extractor head |
|-----------------|-------------------------------------|-----|----------------|
| `ecapa`         | `speechbrain/spkrec-ecapa-voxceleb` | 192 | ECAPA-TDNN     |
| `wavlm`         | `microsoft/wavlm-base-plus`         | 768 | mean-pool      |
| `wavlm_xvector` | `microsoft/wavlm-base-plus-sv`      | 512 | x-vector       |

## How it works

1. **Extraction** — `scripts/embeding_extractor.py` (ECAPA) and
   `scripts/wavlm_extractor.py --model {wavlm,wavlm_xvector}` walk a
   VoxCeleb-structured directory, batch-load `.wav` files, resample to 16 kHz, and
   encode each clip into an embedding. Each model's embeddings + speaker labels are
   saved to its own `.pt` file.
2. **Database** — `data_base/populate_db.py --model <name>` loads a model's
   embeddings into that model's PostgreSQL schema, computes metadata centroids, and
   builds K-means cluster centroids.
3. **Evaluation** — `evaluate/` compares brute-force, metadata-hierarchical, and
   cluster-routed search per model and writes a JSON report.

## Requirements

- Python 3.13+
- CUDA-capable GPU (recommended; falls back to CPU)
- [uv](https://docs.astral.sh/uv/) package manager

## Reproduce with Docker

Requires only Docker (no GPU, no raw audio). The database is shipped as a
pre-populated dump that restores in ~30 s at first startup, so there is no slow
populate step — you just bring up the DB and run the evaluation on demand.

**1. Clone and configure**

```bash
git clone <repo-url> && cd speech-id
cp data_base/.env.example data_base/.env          # demo creds; edit if desired
```

**2. Download the data bundle** from
[Google Drive](https://drive.google.com/drive/folders/1qhWWn4Qvas0FfxecAfIrr74v8sDAZMMG)
and place the files like so:

```
data_base/db_dump/voxceleb_db.dump                  # the pre-populated database
data_base/data/voxceleb_test_embeddings.pt          # ecapa test/probe set
data_base/data/voxceleb_test_embeddings_wavlm.pt    # wavlm test/probe set
data_base/data/voxceleb_test_embeddings_wavlm_xvector.pt
```

**3. Start the database** (auto-restores the dump on first run, ~30 s):

```bash
docker compose up -d --build db
```

**4. Run the evaluation on demand.** Results are written to `./results/`.

```bash
# Quick check — one model, limited queries (~30 s):
docker compose run --rm evaluate \
  uv run --no-sync evaluate/evaluate_search.py --model ecapa --max-queries 500

# One model, full (exact, matches committed results):
docker compose run --rm evaluate \
  uv run --no-sync evaluate/evaluate_search.py --model wavlm

# All three models, full (~45 min — the exact brute-force scan is the cost):
docker compose run --rm evaluate
```

The evaluation deliberately disables index scans for exact results, so the
committed metrics are reproduced exactly. Stop everything with `docker compose
down` (add `-v` to also drop the restored DB volume and re-restore next time).

> If `data_base/db_dump/voxceleb_db.dump` is missing, the DB starts empty and
> evaluation returns no matches. To rebuild from raw embeddings instead, see
> "Rebuild the database from scratch" below.

### Rebuild the database from scratch (optional)

Instead of the dump, you can populate from the raw `.pt` embeddings. This needs
the enrollment files (`voxceleb_embeddings*.pt`) and `vox1_meta.csv` in
`data_base/data/`, and takes ~25 min (it also builds the HNSW index):

```bash
docker compose --profile populate run --rm populate
```

### Optional: regenerate embeddings from raw audio (GPU)

The `extractor` service re-runs `scripts/` to produce the `.pt` files from raw
VoxCeleb audio. This needs an NVIDIA GPU, the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and the VoxCeleb audio under `./data/` (see [Data](#data)).

```bash
docker compose --profile extract run --rm extractor uv run scripts/embeding_extractor.py            # ecapa
docker compose --profile extract run --rm extractor uv run scripts/wavlm_extractor.py --model wavlm
docker compose --profile extract run --rm extractor uv run scripts/wavlm_extractor.py --model wavlm_xvector
```

## Setup (without Docker)

```bash
# Clone the repo
git clone <repo-url>
cd speech-id

# Create virtual environment and install dependencies
uv sync

# Install dev dependencies (black, pre-commit)
uv sync --group dev
pre-commit install
```

### Populate the database, then evaluate

The evaluation queries PostgreSQL, so the database must be populated **before**
running it. Without Docker you need a reachable PostgreSQL with `pgvector`
enabled, with `data_base/.env` pointing at it (`cp data_base/.env.example
data_base/.env` and edit `DB_HOST`/`DB_PORT`/credentials). You can start one with
just `docker compose up -d db`, or use a local install.

```bash
# 1. Put the enrollment .pt files + vox1_meta.csv in data_base/data/, then
#    populate each model's schema (--model is required, deterministic):
uv run python data_base/populate_db.py --model ecapa
uv run python data_base/populate_db.py --model wavlm
uv run python data_base/populate_db.py --model wavlm_xvector

# 2. Run the evaluation against the populated DB:
uv run python evaluate/evaluate_search.py --model ecapa
uv run python evaluate/evaluate_search.py --model wavlm
uv run python evaluate/evaluate_search.py --model wavlm_xvector
```

If you already have the pre-populated `voxceleb_db.dump`, you can skip step 1 by
restoring it instead: `pg_restore -d <db> --no-owner data_base/db_dump/voxceleb_db.dump`.

## Data

Download the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dev or test set and extract it under `data/`:

```
data/
  vox1_dev_wav/
    wav/
      id10032/
        <video_id>/
          00001.wav
          ...
```

## Configuration

Edit `configs/config.ini` to set where embeddings are saved:

```ini
[base]
EMBEDDING_SAVE_DIR = voxceleb_embeddings.pt
```

## Usage

Extract embeddings for each model (run from the project root):

```bash
uv run python scripts/embeding_extractor.py                          # ecapa
uv run python scripts/wavlm_extractor.py --model wavlm               # wavlm (mean-pool)
uv run python scripts/wavlm_extractor.py --model wavlm_xvector       # wavlm x-vector
```

The ECAPA script defaults to `split=vox1_dev_wav`, `batch_size=16`, `target_sample_rate=16000`; edit the `main(...)` call at the bottom to change them. The WavLM extractor takes `--split`, `--batch-size`, `--workers`, and `--max-seconds`.

Pretrained models are downloaded automatically on first run (ECAPA to `pretrained_models/`, the WavLM models to the HuggingFace cache).

Logs are written to `scripts/logs/running_logs.log` and stdout.

## Database

The `data_base/` folder holds the PostgreSQL + `pgvector` setup. The pipeline
compares several embedding models whose vectors have different dimensions, so
**each model lives in its own schema** with an identical set of tables (registry
in `configs/models.py`). Queries run with `SET search_path TO <schema>, public`,
so the same unqualified SQL serves every model.

| model           | schema          | dim | extractor head |
|-----------------|-----------------|-----|----------------|
| `ecapa`         | `public`        | 192 | ecapa          |
| `wavlm`         | `wavlm`         | 768 | meanpool       |
| `wavlm_xvector` | `wavlm_xvector` | 512 | xvector        |

Each schema has four tables (`VECTOR(dim)` templated per model):

- **`speakers`** — `speaker_id` (PK), `gender`, `nationality`.
- **`audio_embeddings`** — `id` (PK), `speaker_id` → speakers, `file_path`, `cluster_id` → cluster_centroids, `embedding VECTOR(dim)`.
- **`metadata_centroids`** — `(category, category_value)` (PK), `centroid_vector` — averaged embeddings grouped by `gender` and `gender:nationality`.
- **`cluster_centroids`** — `cluster_id` (PK), `centroid_vector` — unsupervised K-means centroids.

`populate_db.py` fills one model's schema per run (`--model`, required) in three
phases: load embeddings + metadata and `COPY` into `audio_embeddings`; compute
metadata centroids; run `MiniBatchKMeans(random_state=42)` to build cluster
centroids and assign `cluster_id`. The clustering is seeded, so a populate is
deterministic and reproduces the same results. `schema_overview.py` prints a
Markdown overview of the live DB.

The Docker reproduction ships this database as a pre-restored dump (see above);
populate is only needed to rebuild from raw embeddings.

## Evaluation

The `evaluate/` package evaluates speaker identification over the held-out test
embeddings and writes a JSON summary to
`results/speaker_search_evaluation_<model>.json` (plus a Markdown summary to the
terminal). Run it on demand (see the Docker section) or directly:

```bash
uv run python evaluate/evaluate_search.py --model ecapa
```

It compares three search **architectures**:

- `brute_force` — search the full embedding database.
- `metadata_hierarchy` — route by `gender`, then `gender:nationality`, then search that subset.
- `kmeans_clusters` — route to the nearest cluster centroid(s), then search within.

and three **aggregation** modes: `speaker_centroid` (rank per-speaker centroids),
`closest` (closest single utterance), and `vote` (top hits by voting). Index
scans are deliberately disabled so results are exact (the HNSW index is unused).

Useful options:

- `--model {ecapa,wavlm,wavlm_xvector}` — selects schema, test file, and output name.
- `--max-queries N` — limit test samples (fast smoke runs).
- `--metrics cosine euclidean` — distance metric(s); default cosine.
- `--cluster-probes 1 4 8` — cluster routing depth(s).
- `--utterance-mode all|closest|vote` — which direct-utterance variants to run.
- `--output-file PATH` — override the JSON output location.

## Output

Each extractor writes a `.pt` file containing:

```python
{
    "embeddings": torch.Tensor,  # shape (N, dim) — 192 ecapa / 768 wavlm / 512 wavlm_xvector
    "speakers":   list[str],     # length N, e.g. ["id10032", ...]
}
```
