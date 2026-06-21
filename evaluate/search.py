from __future__ import annotations

from typing import Any

import numpy as np

from .data import vector_to_pg_literal


def metric_operator(metric: str) -> str:
    if metric == "cosine":
        return "<=>"
    if metric == "euclidean":
        return "<->"
    raise ValueError(f"Unsupported metric: {metric}")


def route_best_label(
    query: np.ndarray, labels: list[str], vectors: np.ndarray, metric: str
) -> str | None:
    if not labels or vectors.size == 0:
        return None

    if metric == "cosine":
        scores = vectors @ query
        return labels[int(np.argmax(scores))]

    distances = np.linalg.norm(vectors - query, axis=1)
    return labels[int(np.argmin(distances))]


def route_best_labels(
    query: np.ndarray, labels: list[str], vectors: np.ndarray, metric: str, top_n: int
) -> list[str]:
    if not labels or vectors.size == 0:
        return []

    if metric == "cosine":
        order = np.argsort(-(vectors @ query))
    else:
        order = np.argsort(np.linalg.norm(vectors - query, axis=1))

    limit = max(1, min(top_n, len(labels)))
    return [labels[index] for index in order[:limit]]


def route_metadata_query(
    query: np.ndarray,
    metric: str,
    metadata_centroids: dict[str, tuple[list[str], np.ndarray]],
    probe_count: int = 1,
) -> tuple[str | None, list[str]]:
    gender_labels, gender_vectors = metadata_centroids.get(
        "gender", ([], np.empty((0, 192), dtype=np.float32))
    )
    best_gender = route_best_label(query, gender_labels, gender_vectors, metric)
    if best_gender is None:
        return None, []

    nat_labels, nat_vectors = metadata_centroids.get(
        "gender_nationality", ([], np.empty((0, 192), dtype=np.float32))
    )
    filtered_labels: list[str] = []
    filtered_vectors: list[np.ndarray] = []
    prefix = f"{best_gender}:"
    for label, vector in zip(nat_labels, nat_vectors):
        if label.startswith(prefix):
            filtered_labels.append(label)
            filtered_vectors.append(vector)

    if not filtered_labels:
        return best_gender, []

    best_labels = route_best_labels(
        query,
        filtered_labels,
        np.asarray(filtered_vectors, dtype=np.float32),
        metric,
        probe_count,
    )
    nationalities = [label.split(":", 1)[1] for label in best_labels if ":" in label]
    return best_gender, nationalities


def route_cluster_query(
    query: np.ndarray,
    metric: str,
    cluster_centroids: tuple[list[int], np.ndarray],
    probe_count: int,
) -> list[int]:
    cluster_ids, vectors = cluster_centroids
    if not cluster_ids or vectors.size == 0:
        return []

    if metric == "cosine":
        scores = vectors @ query
        order = np.argsort(-scores)
    else:
        distances = np.linalg.norm(vectors - query, axis=1)
        order = np.argsort(distances)

    limit = max(1, min(probe_count, len(cluster_ids)))
    return [cluster_ids[index] for index in order[:limit]]


def rank_centroids(
    candidate_ids: list[str],
    candidate_vectors: np.ndarray,
    query: np.ndarray,
    metric: str,
    top_k: int,
) -> tuple[list[str], float]:
    if not candidate_ids or candidate_vectors.size == 0:
        return [], -1.0

    if metric == "cosine":
        scores = candidate_vectors @ query
        order = np.argsort(-scores)
        ranked_ids = [candidate_ids[index] for index in order[:top_k]]
        top_score = float(scores[order[0]]) if len(order) else -1.0
        return ranked_ids, top_score

    distances = np.linalg.norm(candidate_vectors - query, axis=1)
    order = np.argsort(distances)
    ranked_ids = [candidate_ids[index] for index in order[:top_k]]
    top_score = float(-distances[order[0]]) if len(order) else -1.0
    return ranked_ids, top_score


def collapse_utterance_hits(
    rows: list[tuple[int, str, float]],
    metric: str,
    mode: str,
    top_k: int,
) -> tuple[list[str], float]:
    if not rows:
        return [], -1.0

    if mode == "closest":
        ranked: list[str] = []
        seen: set[str] = set()
        for _, speaker_id, _ in rows:
            if speaker_id in seen:
                continue
            seen.add(speaker_id)
            ranked.append(speaker_id)
            if len(ranked) >= top_k:
                break
        top_score = (
            float(1.0 - rows[0][2]) if metric == "cosine" else float(-rows[0][2])
        )
        return ranked, top_score

    if mode != "vote":
        raise ValueError(f"Unsupported utterance aggregation mode: {mode}")

    vote_counts: dict[str, int] = {}
    best_distance: dict[str, float] = {}
    first_position: dict[str, int] = {}
    for position, (_, speaker_id, distance) in enumerate(rows):
        vote_counts[speaker_id] = vote_counts.get(speaker_id, 0) + 1
        best_distance[speaker_id] = min(
            best_distance.get(speaker_id, float("inf")), float(distance)
        )
        first_position.setdefault(speaker_id, position)

    ordered = sorted(
        vote_counts,
        key=lambda speaker_id: (
            -vote_counts[speaker_id],
            best_distance[speaker_id],
            first_position[speaker_id],
            speaker_id,
        ),
    )
    ranked = ordered[:top_k]
    top_speaker = ranked[0]
    top_score = (
        float(1.0 - best_distance[top_speaker])
        if metric == "cosine"
        else float(-best_distance[top_speaker])
    )
    return ranked, top_score


def query_utterances(
    cur: Any,
    query_vector: np.ndarray,
    metric: str,
    limit: int,
    gender: str | None = None,
    nationalities: list[str] | None = None,
    cluster_ids: list[int] | None = None,
) -> list[tuple[int, str, float]]:
    operator = metric_operator(metric)
    vector_text = vector_to_pg_literal(query_vector)

    conditions: list[str] = []
    params: list[object] = [vector_text]

    if gender is not None:
        conditions.append("s.gender = %s")
        params.append(gender)
    if nationalities:
        conditions.append("s.nationality = ANY(%s)")
        params.append(nationalities)
    if cluster_ids is not None:
        conditions.append("ae.cluster_id = ANY(%s)")
        params.append(cluster_ids)

    where_sql = ""
    if conditions:
        where_sql = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT ae.id, ae.speaker_id, ae.embedding {operator} %s::vector AS distance
        FROM audio_embeddings ae
        JOIN speakers s ON ae.speaker_id = s.speaker_id
        {where_sql}
        ORDER BY ae.embedding {operator} %s::vector ASC
        LIMIT %s;
    """
    params.append(vector_text)
    params.append(limit)

    cur.execute(query, params)
    return [(int(row[0]), str(row[1]), float(row[2])) for row in cur.fetchall()]


def disable_indexed_search(cur: Any) -> None:
    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_indexonlyscan = off;")
    cur.execute("SET enable_bitmapscan = off;")


def subset_centroids(
    speaker_ids: list[str],
    speaker_vectors: np.ndarray,
    allowed_ids: list[str],
    speaker_index: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    if not allowed_ids:
        return speaker_ids, speaker_vectors

    indices = [
        speaker_index[speaker_id]
        for speaker_id in allowed_ids
        if speaker_id in speaker_index
    ]
    if not indices:
        return speaker_ids, speaker_vectors

    candidate_ids = [speaker_ids[index] for index in indices]
    candidate_vectors = speaker_vectors[indices]
    return candidate_ids, candidate_vectors
