from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg


def collect_database_summary(conn) -> dict[str, int]:
    query = """
        SELECT
            (SELECT COUNT(*) FROM speakers) AS speakers,
            (SELECT COUNT(*) FROM audio_embeddings) AS audio_embeddings,
            (SELECT COUNT(*) FROM metadata_centroids) AS metadata_centroids,
            (SELECT COUNT(*) FROM cluster_centroids) AS cluster_centroids;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        speakers, audio_embeddings, metadata_centroids, cluster_centroids = (
            cur.fetchone()
        )
    return {
        "speakers": int(speakers),
        "audio_embeddings": int(audio_embeddings),
        "metadata_centroids": int(metadata_centroids),
        "cluster_centroids": int(cluster_centroids),
    }


def render_report(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Speaker Identification Search Evaluation")
    lines.append("")
    lines.append(f"Generated: {report['timestamp']}")
    if report.get("model"):
        lines.append(
            f"Model: {report['model']} (schema={report.get('schema')}, "
            f"dim={report.get('embedding_dim')})"
        )
    lines.append(f"Test file: {report['test_file']}")
    lines.append(f"Queries evaluated: {report['query_count']}")
    lines.append(f"Distance metrics: {', '.join(report['metrics'])}")
    lines.append(f"Utterance candidate limit: {report['candidate_limit']}")
    lines.append("")
    lines.append("## Database Summary")
    db = report["database_summary"]
    lines.append(f"- Speakers: {db['speakers']}")
    lines.append(f"- Audio embeddings: {db['audio_embeddings']}")
    lines.append(f"- Metadata centroids: {db['metadata_centroids']}")
    lines.append(f"- Cluster centroids: {db['cluster_centroids']}")
    lines.append("")

    for metric_name, rows in report["results_by_metric"].items():
        lines.append(f"## {metric_name.title()} Distance")
        lines.append("")
        lines.append(
            "| Architecture | Aggregation | Probes | Accuracy (%) | Hit@K (%) | Precision (%) | Recall (%) | F1 (%) | Avg Search Time (ms) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                "| "
                f"{row['architecture']} | "
                f"{row['aggregation']} | "
                f"{row['cluster_probes']} | "
                f"{row['metrics']['top1_accuracy']:.2f} | "
                f"{row['metrics']['hit_k']:.2f} | "
                f"{row['metrics']['precision']:.2f} | "
                f"{row['metrics']['recall']:.2f} | "
                f"{row['metrics']['f1']:.2f} | "
                f"{row['avg_latency_ms']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_results(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
