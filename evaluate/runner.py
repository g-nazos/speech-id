from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    build_connection_string,
    default_output_file,
    get_model,
    load_environment,
    test_embeddings_path,
)
from .data import (
    build_metadata_groups,
    load_cluster_centroids,
    load_cluster_memberships,
    load_metadata_centroids,
    load_speaker_centroids,
    load_speaker_metadata,
    load_test_queries,
)
from .metrics import compute_speaker_metrics
from .reporting import collect_database_summary, render_report, save_results
from .search import (
    collapse_utterance_hits,
    disable_indexed_search,
    query_utterances,
    rank_centroids,
    route_cluster_query,
    route_metadata_query,
    subset_centroids,
)


def evaluate_configuration(
    cur: Any,
    query_vectors: np.ndarray,
    query_speakers: list[str],
    metric: str,
    architecture: str,
    aggregation: str,
    top_k: int,
    candidate_limit: int,
    cluster_probes: int,
    speaker_index: dict[str, int],
    speaker_ids: list[str],
    speaker_vectors: np.ndarray,
    metadata_centroids: dict[str, tuple[list[str], np.ndarray]],
    by_gender: dict[str, list[str]],
    by_gender_nat: dict[str, list[str]],
    cluster_centroids: tuple[list[int], np.ndarray],
    cluster_memberships: dict[int, set[str]],
) -> dict[str, object]:
    disable_indexed_search(cur)

    predictions: list[list[str]] = []
    top_scores: list[float] = []
    latencies_ms: list[float] = []

    for query_vector in query_vectors:
        start_ns = __import__("time").perf_counter_ns()

        routed_gender: str | None = None
        routed_nationalities: list[str] = []
        routed_clusters: list[int] | None = None
        allowed_speakers: list[str] = []

        if architecture == "metadata_hierarchy":
            routed_gender, routed_nationalities = route_metadata_query(
                query_vector, metric, metadata_centroids, cluster_probes
            )
            if routed_gender is not None:
                if routed_nationalities:
                    speakers_in_buckets: set[str] = set()
                    for nationality in routed_nationalities:
                        speakers_in_buckets.update(
                            by_gender_nat.get(f"{routed_gender}:{nationality}", [])
                        )
                    allowed_speakers = sorted(speakers_in_buckets)
                else:
                    allowed_speakers = by_gender.get(routed_gender, [])
        elif architecture == "kmeans_clusters":
            routed_clusters = route_cluster_query(
                query_vector, metric, cluster_centroids, cluster_probes
            )
            if routed_clusters:
                speakers_in_clusters: set[str] = set()
                for cluster_id in routed_clusters:
                    speakers_in_clusters.update(
                        cluster_memberships.get(cluster_id, set())
                    )
                allowed_speakers = sorted(speakers_in_clusters)
        elif architecture != "brute_force":
            raise ValueError(f"Unsupported architecture: {architecture}")

        if aggregation == "speaker_centroid":
            candidate_ids = speaker_ids
            candidate_vectors = speaker_vectors
            if allowed_speakers:
                candidate_ids, candidate_vectors = subset_centroids(
                    speaker_ids, speaker_vectors, allowed_speakers, speaker_index
                )
            ranked_speakers, top_score = rank_centroids(
                candidate_ids, candidate_vectors, query_vector, metric, top_k
            )
        else:
            if architecture == "brute_force":
                utterance_rows = query_utterances(
                    cur, query_vector, metric, candidate_limit
                )
            elif architecture == "metadata_hierarchy":
                utterance_rows = query_utterances(
                    cur,
                    query_vector,
                    metric,
                    candidate_limit,
                    gender=routed_gender,
                    nationalities=routed_nationalities,
                )
            else:
                utterance_rows = query_utterances(
                    cur,
                    query_vector,
                    metric,
                    candidate_limit,
                    cluster_ids=routed_clusters,
                )

            collapse_mode = "closest" if aggregation == "closest" else "vote"
            ranked_speakers, top_score = collapse_utterance_hits(
                utterance_rows, metric, collapse_mode, top_k
            )

        elapsed_ms = (__import__("time").perf_counter_ns() - start_ns) / 1_000_000
        predictions.append(ranked_speakers)
        top_scores.append(top_score)
        latencies_ms.append(elapsed_ms)

    metrics = compute_speaker_metrics(query_speakers, predictions, top_scores)
    return {
        "architecture": architecture,
        "aggregation": aggregation,
        "metric": metric,
        "cluster_probes": cluster_probes,
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker identification search methods using brute force, metadata hierarchy, and unsupervised clusters."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Which embedding model to evaluate (see configs/models.py). Selects the "
        "DB schema (via search_path), the test .pt file, and the output filename.",
    )
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["cosine", "euclidean"],
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument("--cluster-probes", type=int, nargs="+", default=[1])
    parser.add_argument("--metadata-probes", type=int, nargs="+", default=[1])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument(
        "--utterance-mode", choices=["all", "closest", "vote"], default="all"
    )
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    spec = get_model(args.model)
    test_file = args.test_file or test_embeddings_path(spec)
    output_file = args.output_file or default_output_file(spec)
    print(f"Model: {spec.name} | schema: {spec.schema} | dim: {spec.dim}")

    env = load_environment()

    import psycopg
    from psycopg import sql

    conn_str = build_connection_string(env)

    test_vectors, query_speakers, _ = load_test_queries(test_file)
    if args.max_queries is not None:
        test_vectors = test_vectors[: args.max_queries]
        query_speakers = query_speakers[: args.max_queries]

    print(f"Loaded {len(test_vectors)} test vectors from {test_file}")

    with psycopg.connect(conn_str) as conn:
        # Scope every unqualified query to this model's schema (ecapa -> public).
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public;").format(
                    sql.Identifier(spec.schema)
                )
            )
        speaker_metadata = load_speaker_metadata(conn)
        speaker_ids, speaker_vectors = load_speaker_centroids(conn)
        metadata_centroids = load_metadata_centroids(conn)
        cluster_centroids = load_cluster_centroids(conn)
        cluster_memberships = load_cluster_memberships(conn)
        by_gender, by_gender_nat = build_metadata_groups(speaker_metadata)
        database_summary = collect_database_summary(conn)
        speaker_index = {
            speaker_id: index for index, speaker_id in enumerate(speaker_ids)
        }

        results_by_metric: dict[str, list[dict[str, object]]] = {}
        with conn.cursor() as cur:
            for metric in args.metrics:
                metric_rows: list[dict[str, object]] = []
                for architecture in [
                    "brute_force",
                    "metadata_hierarchy",
                    "kmeans_clusters",
                ]:
                    if architecture == "kmeans_clusters":
                        probe_values = sorted(
                            {max(1, value) for value in args.cluster_probes}
                        )
                    elif architecture == "metadata_hierarchy":
                        probe_values = sorted(
                            {max(1, value) for value in args.metadata_probes}
                        )
                    else:
                        probe_values = [1]
                    for cluster_probes in probe_values:
                        for aggregation in ["speaker_centroid", "closest", "vote"]:
                            if (
                                aggregation != "speaker_centroid"
                                and args.utterance_mode not in {"all", aggregation}
                            ):
                                continue

                            print(
                                f"Running architecture={architecture}, aggregation={aggregation}, metric={metric}, cluster_probes={cluster_probes}"
                            )
                            evaluation = evaluate_configuration(
                                cur=cur,
                                query_vectors=test_vectors,
                                query_speakers=query_speakers,
                                metric=metric,
                                architecture=architecture,
                                aggregation=aggregation,
                                top_k=args.top_k,
                                candidate_limit=args.candidate_limit,
                                cluster_probes=cluster_probes,
                                speaker_index=speaker_index,
                                speaker_ids=speaker_ids,
                                speaker_vectors=speaker_vectors,
                                metadata_centroids=metadata_centroids,
                                by_gender=by_gender,
                                by_gender_nat=by_gender_nat,
                                cluster_centroids=cluster_centroids,
                                cluster_memberships=cluster_memberships,
                            )
                            metric_rows.append(evaluation)
                results_by_metric[metric] = metric_rows

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": spec.name,
        "schema": spec.schema,
        "embedding_dim": spec.dim,
        "test_file": str(test_file),
        "query_count": len(test_vectors),
        "metrics": args.metrics,
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "utterance_mode": args.utterance_mode,
        "database_summary": database_summary,
        "results_by_metric": results_by_metric,
    }

    print("\nEvaluation complete.\n")
    print(render_report(report))

    if output_file:
        save_results(output_file, report)
        print(f"Saved evaluation results to {output_file}")


if __name__ == "__main__":
    main()
