from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from core.dataset import discover_paired_scenes
from core.packet_mask import apply_packet_mask


@dataclass
class Metrics:
    mean_iou: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def load_yolo_ground_truth(label_path: Path, image_shape, label_class_id: int = 0) -> np.ndarray:
    h, w = image_shape[:2]
    boxes = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=float)
    text = label_path.read_text().strip()
    if not text:
        return np.empty((0, 4), dtype=float)

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls != label_class_id:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        boxes.append([
            (xc - bw / 2.0) * w,
            (yc - bh / 2.0) * h,
            (xc + bw / 2.0) * w,
            (yc + bh / 2.0) * h,
        ])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred_boxes, pred_conf, gt_boxes, iou_threshold: float = 0.5) -> Metrics:
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0, 0, n_gt)

    order = np.argsort(-pred_conf)
    matched_gt = set()
    matched_ious = []
    tp = 0

    for pi in order:
        best_j = -1
        best_iou = 0.0
        for gj, gt in enumerate(gt_boxes):
            if gj in matched_gt:
                continue
            iou = box_iou(pred_boxes[pi], gt)
            if iou > best_iou:
                best_iou = iou
                best_j = gj
        if best_j >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_j)
            matched_ious.append(best_iou)
            tp += 1

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return Metrics(mean_iou, precision, recall, f1, tp, fp, fn)


def predict_person(model: YOLO, image: np.ndarray, conf: float):
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=float)
    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(float)
    scores = result.boxes.conf.detach().cpu().numpy().astype(float)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    keep = classes == 0
    return xyxy[keep], scores[keep]


def choose_helper_by_alpha(alphas: dict[str, float]) -> str:
    return max(("h1", "h2", "h3"), key=lambda h: (alphas[h], -int(h[1])))


def metric_dict(prefix: str, m: Metrics) -> dict:
    return {
        f"{prefix}_iou": m.mean_iou,
        f"{prefix}_precision": m.precision,
        f"{prefix}_recall": m.recall,
        f"{prefix}_f1": m.f1,
        f"{prefix}_tp": m.tp,
        f"{prefix}_fp": m.fp,
        f"{prefix}_fn": m.fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", default="train")
    parser.add_argument("--weights", default="yolov8x-worldv2.pt")
    parser.add_argument("--alpha-h1", type=float, default=0.90)
    parser.add_argument("--alpha-h2", type=float, default=0.75)
    parser.add_argument("--alpha-h3", type=float, default=0.55)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--label-class-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--save-masked", type=int, default=5)
    parser.add_argument("--output-dir", default="results/yolo_experiment")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = repo_root / weights
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    alphas = {"h1": args.alpha_h1, "h2": args.alpha_h2, "h3": args.alpha_h3}
    for name, value in alphas.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} alpha must be in [0, 1]")

    selected_helper = choose_helper_by_alpha(alphas)
    print(f"Selected helper by highest alpha: {selected_helper.upper()} (alpha={alphas[selected_helper]:.3f})")

    scenes = discover_paired_scenes(repo_root, args.split)
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]
    print(f"Paired scenes used: {len(scenes)}")

    # Inference only: no training.
    model = YOLO(str(weights))

    out_dir = repo_root / args.output_dir
    masked_dir = out_dir / "masked_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []

    for scene_index, scene in enumerate(scenes):
        ego = cv2.imread(str(scene.ego.image))
        helper_files = scene.helper(selected_helper)
        helper = cv2.imread(str(helper_files.image))
        if ego is None or helper is None:
            raise RuntimeError(f"Could not read scene {scene.scene_id}")

        alpha = alphas[selected_helper]
        corrupted_helper, dropped = apply_packet_mask(helper, alpha, args.grid, rng, 0)

        if scene_index < args.save_masked:
            cv2.imwrite(str(masked_dir / f"{scene.scene_id}_{selected_helper}_masked.jpg"), corrupted_helper)

        ego_gt = load_yolo_ground_truth(scene.ego.label, ego.shape, args.label_class_id)
        helper_gt = load_yolo_ground_truth(helper_files.label, helper.shape, args.label_class_id)

        ego_boxes, ego_conf = predict_person(model, ego, args.conf)
        clean_h_boxes, clean_h_conf = predict_person(model, helper, args.conf)
        corrupt_h_boxes, corrupt_h_conf = predict_person(model, corrupted_helper, args.conf)

        ego_m = evaluate(ego_boxes, ego_conf, ego_gt, args.iou_threshold)
        helper_clean_m = evaluate(clean_h_boxes, clean_h_conf, helper_gt, args.iou_threshold)
        helper_corrupt_m = evaluate(corrupt_h_boxes, corrupt_h_conf, helper_gt, args.iou_threshold)

        # Paper-style evaluation: IoU_s = max(IoU_e, IoU_h).
        helper_wins = helper_corrupt_m.mean_iou > ego_m.mean_iou
        best_m = helper_corrupt_m if helper_wins else ego_m

        row = {
            "scene_id": scene.scene_id,
            "selected_helper": selected_helper,
            "alpha": alpha,
            "drop_probability": 1.0 - alpha,
            "dropped_blocks": int(dropped.sum()),
            "total_blocks": int(args.grid * args.grid),
            "fused_source": selected_helper if helper_wins else "ego",
            "fused_iou": max(ego_m.mean_iou, helper_corrupt_m.mean_iou),
            "fused_precision": best_m.precision,
            "fused_recall": best_m.recall,
            "fused_f1": best_m.f1,
        }
        row.update(metric_dict("ego", ego_m))
        row.update(metric_dict("helper_clean", helper_clean_m))
        row.update(metric_dict("helper_corrupt", helper_corrupt_m))
        rows.append(row)

        if (scene_index + 1) % 25 == 0 or scene_index == len(scenes) - 1:
            print(f"Processed {scene_index + 1}/{len(scenes)} scenes")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scene_metrics.csv", index=False)

    summary_cols = [
        "ego_iou", "ego_recall", "ego_f1",
        "helper_clean_iou", "helper_clean_recall", "helper_clean_f1",
        "helper_corrupt_iou", "helper_corrupt_recall", "helper_corrupt_f1",
        "fused_iou", "fused_recall", "fused_f1", "dropped_blocks",
    ]
    summary = pd.DataFrame({"mean": df[summary_cols].mean(), "std": df[summary_cols].std(ddof=1)})
    summary.to_csv(out_dir / "summary.csv")

    config = {
        "weights": str(weights),
        "split": args.split,
        "num_scenes": len(df),
        "selected_helper": selected_helper,
        "alphas": alphas,
        "grid": args.grid,
        "conf": args.conf,
        "iou_threshold": args.iou_threshold,
        "seed": args.seed,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("\n=== Summary ===")
    print(summary.round(4).to_string())
    print(f"\nSaved: {out_dir / 'scene_metrics.csv'}")
    print(f"Saved: {out_dir / 'summary.csv'}")
    print(f"Masked examples: {masked_dir}")


if __name__ == "__main__":
    main()
