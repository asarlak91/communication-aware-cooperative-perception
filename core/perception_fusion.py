from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray
    confidence: float
    class_id: int


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    b1 = np.asarray(box1, dtype=float)
    b2 = np.asarray(box2, dtype=float)
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def run_yolo(
    image_path: str | Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    class_ids: Iterable[int] | None = None,
) -> list[Detection]:
    """Run actual Ultralytics YOLO inference on one image."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics to use YOLO inference") from exc

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    model = YOLO(model_name)
    result = model(str(image_path), conf=conf, verbose=False)[0]
    detections: list[Detection] = []
    allowed = None if class_ids is None else set(int(c) for c in class_ids)

    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    for box, score, cls in zip(xyxy, scores, classes):
        if allowed is not None and int(cls) not in allowed:
            continue
        detections.append(
            Detection(xyxy=np.asarray(box, dtype=float), confidence=float(score), class_id=int(cls))
        )
    return detections


def classwise_nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    for class_id in sorted(set(d.class_id for d in detections)):
        pending = sorted(
            [d for d in detections if d.class_id == class_id],
            key=lambda d: d.confidence,
            reverse=True,
        )
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [d for d in pending if compute_iou(best.xyxy, d.xyxy) < iou_threshold]
    return kept


def late_fusion(
    ego_detections: list[Detection],
    helper_detections: dict[int, list[Detection]],
    surviving_helpers: Iterable[int],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Late fusion after channel masking.

    This implementation assumes detections are already expressed in a common
    image/reference coordinate system. It merges all surviving decisions and
    applies class-wise NMS.
    """

    merged = list(ego_detections)
    for helper_id in surviving_helpers:
        merged.extend(helper_detections.get(int(helper_id), []))
    return classwise_nms(merged, iou_threshold=iou_threshold)
