from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0

    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.0

    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    far: list[float] = []
    frr: list[float] = []
    for threshold in thresholds:
        accepted = scores >= threshold
        false_accept = np.logical_and(accepted, negatives).sum()
        miss = np.logical_and(~accepted, positives).sum()
        far.append(false_accept / n_neg)
        frr.append(miss / n_pos)

    far_array = np.asarray(far)
    frr_array = np.asarray(frr)
    index = int(np.argmin(np.abs(far_array - frr_array)))
    return float((far_array[index] + frr_array[index]) / 2.0 * 100.0)


def compute_speaker_metrics(
    query_speakers: list[str],
    predictions: list[list[str]],
    top1_scores: list[float],
) -> dict[str, float]:
    if len(predictions) != len(top1_scores):
        raise ValueError("Prediction count and score count must match.")

    if query_speakers and len(query_speakers) != len(predictions):
        raise ValueError("Query count and prediction count must match when query labels are supplied.")

    if not query_speakers:
        return {
            "top1_accuracy": 0.0,
            "hit_k": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "eer": 0.0,
        }

    y_true = [str(label) for label in query_speakers]
    y_pred = [preds[0] if preds else "__NO_PREDICTION__" for preds in predictions]
    hit_k = [1 if label in preds else 0 for label, preds in zip(y_true, predictions)]

    top1_accuracy = float(sum(1 for true, pred in zip(y_true, y_pred) if true == pred)) / len(y_true) * 100.0
    hit_k_rate = float(sum(hit_k)) / len(hit_k) * 100.0
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    labels = np.array([1 if true == pred else 0 for true, pred in zip(y_true, y_pred)], dtype=int)
    scores = np.asarray(top1_scores, dtype=np.float32)
    eer = compute_eer(labels, scores)

    return {
        "top1_accuracy": top1_accuracy,
        "hit_k": hit_k_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
    }
