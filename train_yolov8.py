from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

VIEWS = {
    "ego": "Data_Ego_Vehicle",
    "h1": "Data_Helper_1",
    "h2": "Data_Helper_2",
    "h3": "Data_Helper_3",
}
SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCENE_RE = re.compile(r"^(\d+)_")


def scene_id(path: Path) -> int | None:
    m = SCENE_RE.match(path.name)
    return int(m.group(1)) if m else None


def safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def collect_view(repo_root: Path, folder: str) -> dict[int, tuple[Path, Path]]:
    """Collect image/label pairs from all existing train/valid/test folders."""
    root = repo_root / "data" / folder
    found: dict[int, tuple[Path, Path]] = {}

    for split in SOURCE_SPLITS:
        images = root / split / "images"
        labels = root / split / "labels"
        if not images.exists() or not labels.exists():
            continue

        for img in sorted(images.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            sid = scene_id(img)
            if sid is None:
                continue

            lab = labels / f"{img.stem}.txt"
            if not lab.exists():
                continue

            # One image per scene per view is expected.
            if sid not in found:
                found[sid] = (img, lab)

    return found


def build_combined_dataset(
    repo_root: Path,
    rebuild: bool = False,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Build one scene-wise YOLO dataset from Ego/H1/H2/H3.

    A scene id is assigned to exactly one of train/valid/test, so the same
    physical scene cannot leak across splits through another vehicle view.
    """
    out_root = repo_root / "data" / "combined_yolo"
    if rebuild and out_root.exists():
        shutil.rmtree(out_root)

    by_view = {name: collect_view(repo_root, folder) for name, folder in VIEWS.items()}

    common = set(by_view["ego"])
    for name in ("h1", "h2", "h3"):
        common &= set(by_view[name])

    scene_ids = sorted(common)
    if not scene_ids:
        raise RuntimeError("No complete Ego/H1/H2/H3 scenes were found.")

    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    n = len(scene_ids)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    split_ids = {
        "train": scene_ids[:n_train],
        "valid": scene_ids[n_train:n_train + n_val],
        "test": scene_ids[n_train + n_val:],
    }

    counts = {}
    for split, ids in split_ids.items():
        out_images = out_root / split / "images"
        out_labels = out_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        count = 0
        for sid in ids:
            for view in ("ego", "h1", "h2", "h3"):
                img, lab = by_view[view][sid]
                new_stem = f"{view}__{img.stem}"
                safe_link(img, out_images / f"{new_stem}{img.suffix.lower()}")
                safe_link(lab, out_labels / f"{new_stem}.txt")
                count += 1
        counts[split] = count

    yaml_path = out_root / "data.yaml"
    config = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "pedestrians"},
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\nComplete paired scenes:", len(scene_ids))
    print("Scene-wise split:")
    print(f"  train scenes: {len(split_ids['train'])} -> {counts['train']} images")
    print(f"  valid scenes: {len(split_ids['valid'])} -> {counts['valid']} images")
    print(f"  test scenes : {len(split_ids['test'])} -> {counts['test']} images")
    print(f"Dataset YAML: {yaml_path}")

    return yaml_path


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the paired Ego/H1/H2/H3 pedestrian dataset."
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild-dataset", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()

    data_yaml = build_combined_dataset(
        repo_root=repo_root,
        rebuild=args.rebuild_dataset,
        seed=args.seed,
    )

    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nTraining device: {device}")
    print(f"Pretrained model: {args.model}")
    print(f"Epochs: {args.epochs}\n")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=10,
        pretrained=True,
        seed=args.seed,
        plots=True,
        project=str(repo_root / "results" / "training"),
        name="yolov8_pedestrian",
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print("\nTraining complete.")
    print(f"Training results: {save_dir}")
    print(f"Best weights: {best_weights}")

    if best_weights.exists():
        print("\nEvaluating BEST checkpoint on the held-out TEST scenes...")
        best = YOLO(str(best_weights))
        metrics = best.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            device=device,
            plots=True,
            project=str(repo_root / "results" / "training"),
            name="yolov8_pedestrian_test",
            exist_ok=True,
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        print("\n=== TEST DETECTOR RESULTS ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"mAP50:     {float(metrics.box.map50):.4f}")
        print(f"mAP50-95:  {float(metrics.box.map):.4f}")

        print("\nUse this checkpoint in the cooperative-perception experiment:")
        print(best_weights)


if __name__ == "__main__":
    main()
