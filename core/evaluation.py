from __future__ import annotations

import numpy as np

from core.perception_fusion import Detection, compute_iou


def evaluate_detections(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float = 0.5,
) -> dict:
    """Simple one-to-one class-aware detection evaluation."""

    matched_gt: set[int] = set()
    tp = 0
    ious: list[float] = []

    for pred in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt or gt.class_id != pred.class_id:
                continue
            iou = compute_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
            ious.append(best_iou)

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }
