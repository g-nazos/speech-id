from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    import psycopg


@dataclass(slots=True)
class SpeakerMetadata:
    speaker_id: str
    gender: str
    nationality: str


def load_test_queries(pt_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    if not pt_path.exists():
        raise FileNotFoundError(f"Test embeddings file not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(
            "Loaded .pt file must be a dict containing embeddings and speaker labels."
        )

    embeddings = data.get("embeddings", data.get("x"))
    if embeddings is None:
        raise ValueError(
            "Loaded .pt file is missing an embeddings tensor under 'embeddings' or 'x'."
        )

    speakers = data.get("speakers")
    if speakers is None:
        raise ValueError(
            "Loaded .pt file must include a 'speakers' list for ground truth speaker labels."
        )

    file_paths = data.get("processed_files") or data.get("file_paths") or []

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D test embeddings, got shape {embeddings.shape}.")

    if len(speakers) != embeddings.shape[0]:
        raise ValueError(
            f"Speaker labels length {len(speakers)} does not match embeddings rows {embeddings.shape[0]}"
        )

    normalized = normalize(embeddings, norm="l2", axis=1)
    return (
        normalized.astype(np.float32),
        [str(value) for value in speakers],
        [str(value) for value in file_paths],
    )


def parse_pgvector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
    raise ValueError(f"Unsupported pgvector value type: {type(value)}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def vector_to_pg_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in vector) + "]"


def load_speaker_metadata(conn) -> dict[str, SpeakerMetadata]:
    query = "SELECT speaker_id, gender, nationality FROM speakers ORDER BY speaker_id;"
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    metadata: dict[str, SpeakerMetadata] = {}
    for speaker_id, gender, nationality in rows:
        metadata[str(speaker_id)] = SpeakerMetadata(
            speaker_id=str(speaker_id),
            gender=str(gender).lower(),
            nationality=str(nationality),
        )
    return metadata


def load_speaker_centroids(conn) -> tuple[list[str], np.ndarray]:
    query = """
        SELECT speaker_id, AVG(embedding)::vector AS centroid_vector
        FROM audio_embeddings
        GROUP BY speaker_id
        ORDER BY speaker_id;
    """
    speaker_ids: list[str] = []
    vectors: list[np.ndarray] = []
    with conn.cursor() as cur:
        cur.execute(query)
        for speaker_id, vector_value in cur.fetchall():
            if vector_value is None:
                continue
            speaker_ids.append(str(speaker_id))
            vectors.append(normalize_vector(parse_pgvector(vector_value)))

    if not vectors:
        return [], np.empty((0, 192), dtype=np.float32)
    return speaker_ids, np.asarray(vectors, dtype=np.float32)


def load_metadata_centroids(conn) -> dict[str, tuple[list[str], np.ndarray]]:
    query = """
        SELECT category, category_value, centroid_vector
        FROM metadata_centroids
        ORDER BY category, category_value;
    """
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {
        "gender": [],
        "gender_nationality": [],
    }
    with conn.cursor() as cur:
        cur.execute(query)
        for category, category_value, vector_value in cur.fetchall():
            if vector_value is None:
                continue
            grouped.setdefault(str(category), []).append(
                (str(category_value), normalize_vector(parse_pgvector(vector_value)))
            )

    output: dict[str, tuple[list[str], np.ndarray]] = {}
    for category, entries in grouped.items():
        if not entries:
            output[category] = ([], np.empty((0, 192), dtype=np.float32))
            continue
        labels = [label for label, _ in entries]
        vectors = np.asarray([vector for _, vector in entries], dtype=np.float32)
        output[category] = (labels, vectors)
    return output


def load_cluster_centroids(conn) -> tuple[list[int], np.ndarray]:
    query = """
        SELECT cluster_id, centroid_vector
        FROM cluster_centroids
        ORDER BY cluster_id;
    """
    cluster_ids: list[int] = []
    vectors: list[np.ndarray] = []
    with conn.cursor() as cur:
        cur.execute(query)
        for cluster_id, vector_value in cur.fetchall():
            if vector_value is None:
                continue
            cluster_ids.append(int(cluster_id))
            vectors.append(normalize_vector(parse_pgvector(vector_value)))

    if not vectors:
        return [], np.empty((0, 192), dtype=np.float32)
    return cluster_ids, np.asarray(vectors, dtype=np.float32)


def load_cluster_memberships(conn) -> dict[int, set[str]]:
    query = """
        SELECT DISTINCT cluster_id, speaker_id
        FROM audio_embeddings
        WHERE cluster_id IS NOT NULL
        ORDER BY cluster_id, speaker_id;
    """
    memberships: dict[int, set[str]] = {}
    with conn.cursor() as cur:
        cur.execute(query)
        for cluster_id, speaker_id in cur.fetchall():
            memberships.setdefault(int(cluster_id), set()).add(str(speaker_id))
    return memberships


def build_metadata_groups(
    metadata: dict[str, SpeakerMetadata],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_gender: dict[str, list[str]] = {}
    by_gender_nat: dict[str, list[str]] = {}

    for speaker_id, info in metadata.items():
        by_gender.setdefault(info.gender, []).append(speaker_id)
        by_gender_nat.setdefault(f"{info.gender}:{info.nationality}", []).append(
            speaker_id
        )

    for values in by_gender.values():
        values.sort()
    for values in by_gender_nat.values():
        values.sort()

    return by_gender, by_gender_nat
