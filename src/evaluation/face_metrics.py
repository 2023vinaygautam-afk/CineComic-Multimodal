
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    """Calculate precision, recall and F1 with safe zero-denominator handling."""
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over Union for [x1, y1, x2, y2] bounding boxes."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def detection_metrics(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Evaluate face detections against ground truth.

    Each record must contain a bbox [x1, y1, x2, y2].
    Greedy one-to-one matching is used per supplied frame.
    """
    matched_gt: set[int] = set()
    true_positives = 0

    candidates: list[tuple[float, int, int]] = []

    for pred_idx, pred in enumerate(predicted):
        for gt_idx, gt in enumerate(ground_truth):
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(reverse=True)

    matched_predictions: set[int] = set()

    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_predictions or gt_idx in matched_gt:
            continue
        matched_predictions.add(pred_idx)
        matched_gt.add(gt_idx)
        true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives

    result: dict[str, float | int] = precision_recall_f1(
        true_positives,
        false_positives,
        false_negatives,
    )
    result.update(
        {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
    )
    return result


def pairwise_identity_metrics(
    ground_truth_ids: list[str],
    predicted_ids: list[str],
) -> dict[str, float | int]:
    """Pairwise identity precision, recall and F1 for aligned observations."""
    if len(ground_truth_ids) != len(predicted_ids):
        raise ValueError(
            "Ground-truth and predicted identity lists must have equal lengths."
        )

    tp = fp = fn = 0
    count = len(ground_truth_ids)

    for i in range(count):
        for j in range(i + 1, count):
            gt_same = (
                ground_truth_ids[i] == ground_truth_ids[j]
                and ground_truth_ids[i] != "UNKNOWN"
            )
            pred_same = (
                predicted_ids[i] == predicted_ids[j]
                and predicted_ids[i] != "UNKNOWN"
            )

            if gt_same and pred_same:
                tp += 1
            elif pred_same and not gt_same:
                fp += 1
            elif gt_same and not pred_same:
                fn += 1

    result: dict[str, float | int] = precision_recall_f1(tp, fp, fn)
    result["pairwise_tp"] = tp
    result["pairwise_fp"] = fp
    result["pairwise_fn"] = fn
    return result


def b_cubed_metrics(
    ground_truth_ids: list[str],
    predicted_ids: list[str],
) -> dict[str, float]:
    """Calculate B-cubed precision, recall and F1 for aligned identity labels."""
    if len(ground_truth_ids) != len(predicted_ids):
        raise ValueError(
            "Ground-truth and predicted identity lists must have equal lengths."
        )

    n = len(ground_truth_ids)
    if n == 0:
        return {"b3_precision": 0.0, "b3_recall": 0.0, "b3_f1": 0.0}

    gt_clusters: dict[str, set[int]] = defaultdict(set)
    pred_clusters: dict[str, set[int]] = defaultdict(set)

    for i, (gt_id, pred_id) in enumerate(
        zip(ground_truth_ids, predicted_ids)
    ):
        gt_clusters[str(gt_id)].add(i)
        pred_clusters[str(pred_id)].add(i)

    precision_values: list[float] = []
    recall_values: list[float] = []

    for i, (gt_id, pred_id) in enumerate(
        zip(ground_truth_ids, predicted_ids)
    ):
        gt_members = gt_clusters[str(gt_id)]
        pred_members = pred_clusters[str(pred_id)]
        intersection = len(gt_members & pred_members)

        precision_values.append(intersection / len(pred_members))
        recall_values.append(intersection / len(gt_members))

    precision = float(np.mean(precision_values))
    recall = float(np.mean(recall_values))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "b3_precision": precision,
        "b3_recall": recall,
        "b3_f1": float(f1),
    }


def count_id_switches(
    observations: list[dict[str, Any]],
) -> int:
    """
    Count predicted identity changes for each ground-truth person.

    Observations should contain:
      shot_id, timestamp, ground_truth_id, predicted_id

    Only changes within the same shot are counted.
    """
    by_person: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for observation in observations:
        shot_id = str(observation["shot_id"])
        gt_id = str(observation["ground_truth_id"])
        by_person[(shot_id, gt_id)].append(observation)

    switches = 0

    for person_observations in by_person.values():
        person_observations.sort(key=lambda x: float(x["timestamp"]))

        previous_id: str | None = None

        for observation in person_observations:
            current_id = str(observation["predicted_id"])

            if current_id == "UNKNOWN":
                continue

            if previous_id is not None and current_id != previous_id:
                switches += 1

            previous_id = current_id

    return switches
