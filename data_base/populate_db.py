import os
import sys
import csv
import json
import struct
import argparse
import torch
import psycopg
import numpy as np
from psycopg import sql
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
SCHEMA_TEMPLATE_PATH = os.path.join(BASE_DIR, "schema.template.sql")

sys.path.insert(0, PROJECT_ROOT)
from configs.models import get_model, MODELS, ModelSpec  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DB_CONN_STRING = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

METADATA_CSV_PATH = os.path.join(DATA_DIR, os.getenv("METADATA_CSV_NAME"))


def connect(schema):
    """Open a connection scoped to a model's schema via search_path.

    With search_path set, all the unqualified table names below (speakers,
    audio_embeddings, ...) resolve to <schema>.<table>, so the same SQL serves
    every model without modification.
    """
    conn = psycopg.connect(DB_CONN_STRING)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema))
        )
    return conn


def ensure_schema(schema, dim):
    """Create the model's schema/tables/indexes (idempotent), templated by dim."""
    with open(SCHEMA_TEMPLATE_PATH, encoding="utf-8") as f:
        ddl = f.read().format(schema=schema, dim=dim)

    with psycopg.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(ddl)
        conn.commit()
    print(f"Ensured schema '{schema}' with VECTOR({dim}) columns.")


def load_metadata_lookup(csv_path):
    """Parses VoxCeleb metadata into a dictionary mapped by Speaker ID."""
    metadata_lookup = {}

    if not csv_path or not os.path.exists(csv_path):
        print(
            f"Warning: Metadata file not found at '{csv_path}'. Defaulting to 'unknown'."
        )
        return metadata_lookup

    with open(csv_path, mode="r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            spk_id = row.get("VoxCeleb1 ID", "").strip()
            gender = row.get("Gender", "unknown").strip().lower()
            nationality = row.get("Nationality", "unknown").strip()

            gender_full = (
                "male"
                if gender in ["m", "male"]
                else "female" if gender in ["f", "female"] else "unknown"
            )

            if spk_id:
                metadata_lookup[spk_id] = {
                    "gender": gender_full,
                    "nationality": nationality,
                }

    print(f"Indexed metadata for {len(metadata_lookup)} speakers.")
    return metadata_lookup


def compute_and_save_centroids(cur, dim):
    """Computes a two-layer hierarchy into metadata_centroids (Phase 2)."""
    print("Generating hierarchical tree centroids...")

    cur.execute("TRUNCATE TABLE metadata_centroids;")

    # Layer 1: Global Gender Centroids
    gender_query = f"""
        INSERT INTO metadata_centroids (category, category_value, centroid_vector)
        SELECT
            'gender' as category,
            s.gender as category_value,
            avg(ae.embedding)::vector({dim}) as centroid_vector
        FROM audio_embeddings ae
        JOIN speakers s ON ae.speaker_id = s.speaker_id
        WHERE s.gender IS NOT NULL AND s.gender != 'unknown'
        GROUP BY s.gender;
    """
    cur.execute(gender_query)

    # Layer 2: Dependent Gender-Nationality Centroids (e.g., 'male:US')
    gender_nat_query = f"""
        INSERT INTO metadata_centroids (category, category_value, centroid_vector)
        SELECT
            'gender_nationality' as category,
            CONCAT(s.gender, ':', s.nationality) as category_value,
            avg(ae.embedding)::vector({dim}) as centroid_vector
        FROM audio_embeddings ae
        JOIN speakers s ON ae.speaker_id = s.speaker_id
        WHERE s.gender != 'unknown' AND s.nationality != 'unknown'
        GROUP BY s.gender, s.nationality;
    """
    cur.execute(gender_nat_query)
    print("Centroid tree generation completed.")


def load_data_to_postgres(spec: ModelSpec, pt_path: str):
    """Populates speakers and audio_embeddings from raw source files (Phase 1)."""
    if not pt_path or not os.path.exists(pt_path):
        print(f"Error: Embeddings file not found at: '{pt_path}'.")
        return

    metadata_map = load_metadata_lookup(METADATA_CSV_PATH)

    print("Loading tensor data into memory...")
    data = torch.load(pt_path, map_location=torch.device("cpu"))

    embeddings = data["embeddings"].tolist()
    speakers = data["speakers"]
    processed_files = data["processed_files"]

    total_records = len(speakers)
    if total_records and len(embeddings[0]) != spec.dim:
        print(
            f"Error: embeddings have dim {len(embeddings[0])} but model '{spec.name}' "
            f"expects {spec.dim}. Check the registry / .pt file."
        )
        return
    print(f"Processing {total_records} records into schema '{spec.schema}'...")

    try:
        with connect(spec.schema) as conn:
            with conn.cursor() as cur:

                # Clean rebuild: COPY appends with no dedup, so wipe first.
                print("Truncating existing audio_embeddings for a clean rebuild...")
                cur.execute("TRUNCATE audio_embeddings RESTART IDENTITY;")

                unique_speakers = set(speakers)
                speaker_rows = []

                for spk_id in unique_speakers:
                    spk_meta = metadata_map.get(
                        spk_id, {"gender": "unknown", "nationality": "unknown"}
                    )
                    speaker_rows.append(
                        (spk_id, spk_meta["gender"], spk_meta["nationality"])
                    )

                print("Inserting speakers...")
                cur.executemany(
                    """
                    INSERT INTO speakers (speaker_id, gender, nationality)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (speaker_id) DO NOTHING;
                """,
                    speaker_rows,
                )

                print("Executing binary COPY for audio embeddings...")
                copy_query = "COPY audio_embeddings (speaker_id, file_path, embedding) FROM STDIN WITH (FORMAT BINARY)"

                with cur.copy(copy_query) as copy:
                    for i in range(total_records):
                        spk_id = speakers[i]
                        file_path = processed_files[i]
                        vector = embeddings[i]

                        spk_bytes = spk_id.encode("utf-8")
                        path_bytes = file_path.encode("utf-8")

                        dim = len(vector)
                        unused = 0
                        vector_binary_data = struct.pack(
                            f">HH{dim}f", dim, unused, *vector
                        )

                        copy.write_row([spk_bytes, path_bytes, vector_binary_data])

                compute_and_save_centroids(cur, spec.dim)
                conn.commit()

        print("Database population pipeline finished successfully.")

    except psycopg.OperationalError as e:
        print(f"Database Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def fetch_embeddings(cur):
    """Helper: Fetches all committed embeddings and their generated IDs."""
    print("Fetching embeddings from PostgreSQL for clustering...")
    cur.execute("SELECT id, embedding::text FROM audio_embeddings;")
    rows = cur.fetchall()

    ids = np.array([row[0] for row in rows])
    raw_vectors = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)

    print(f"Successfully loaded {len(ids)} vectors into Python memory.")
    return ids, raw_vectors


def build_unsupervised_clusters(spec: ModelSpec, num_clusters=256):
    """Phase 3: Unsupervised Spherical K-Means Clustering on existing entries."""
    try:
        with connect(spec.schema) as conn:
            with conn.cursor() as cur:

                # Step 1: Pull database records into NumPy space
                ids, raw_vectors = fetch_embeddings(cur)

                if len(ids) == 0:
                    print(
                        "Error: No embeddings found to cluster! Ensure Phase 1 ran correctly."
                    )
                    return

                # Step 2: Apply the Spherical K-Means Normalization Trick
                print("Applying L2 normalization to match database Cosine geometry...")
                normalized_vectors = normalize(raw_vectors, norm="l2", axis=1)

                # Step 3: Run K-Means
                print(f"Training MiniBatchKMeans with K={num_clusters}...")
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    batch_size=2048,
                    random_state=42,
                    n_init="auto",
                )
                cluster_labels = kmeans.fit_predict(normalized_vectors)

                # Re-normalize resulting cluster centers to guarantee perfect spherical bounds
                raw_centroids = kmeans.cluster_centers_
                normalized_centroids = normalize(raw_centroids, norm="l2", axis=1)

                # Step 4: Safely wipe cluster_centroids using DML DELETE
                print("Clearing old cluster centers safely...")
                cur.execute("DELETE FROM cluster_centroids;")

                centroid_insert_data = [
                    (i, normalized_centroids[i].tolist()) for i in range(num_clusters)
                ]
                cur.executemany(
                    """
                    INSERT INTO cluster_centroids (cluster_id, centroid_vector)
                    VALUES (%s, %s::vector)
                """,
                    centroid_insert_data,
                )

                # Step 5: Fast temporary-table batch update to assign rows to clusters
                print("Executing high-speed bulk update for cluster assignments...")
                cur.execute("""
                    CREATE TEMP TABLE temp_cluster_assignments (
                        embedding_id INT,
                        cluster_id INT
                    ) ON COMMIT DROP;
                """)

                copy_query = "COPY temp_cluster_assignments (embedding_id, cluster_id) FROM STDIN"
                with cur.copy(copy_query) as copy:
                    for db_id, cluster_id in zip(ids, cluster_labels):
                        copy.write_row([int(db_id), int(cluster_id)])

                cur.execute("""
                    UPDATE audio_embeddings ae
                    SET cluster_id = tc.cluster_id
                    FROM temp_cluster_assignments tc
                    WHERE ae.id = tc.embedding_id;
                """)

                conn.commit()
                print("Phase 3 unsupervised clustering completed successfully.")

    except psycopg.OperationalError as e:
        print(f"Database Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected clustering error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Populate a model's isolated schema with embeddings, metadata "
        "centroids, and K-means clusters."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS),
        help="Which embedding model's schema to populate (see configs/models.py). "
        "Required, so a bare run can never rebuild the existing public/ecapa schema.",
    )
    parser.add_argument(
        "--pt-file",
        default=None,
        help="Override the enrollment .pt filename (under data_base/data/).",
    )
    parser.add_argument("--clusters", type=int, default=256)
    args = parser.parse_args()

    spec = get_model(args.model)
    pt_file = args.pt_file or spec.pt_file
    pt_path = os.path.join(DATA_DIR, pt_file)
    print(
        f"Model: {spec.name} | schema: {spec.schema} | dim: {spec.dim} | pt: {pt_file}"
    )

    # 0. Ensure the per-model schema/tables exist (templated by dimension).
    ensure_schema(spec.schema, spec.dim)

    # 1. Phase 1 (Ingestion) & Phase 2 (Hierarchical Metadata Mapping)
    load_data_to_postgres(spec, pt_path)

    # 2. Phase 3 (Unsupervised Clustering Partitioning via K-Means)
    build_unsupervised_clusters(spec, num_clusters=args.clusters)


if __name__ == "__main__":
    main()
