from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict

SCENE_RE = re.compile(r"^(\d+)_")
VIEW_FOLDERS = {
    "ego": "Data_Ego_Vehicle",
    "h1": "Data_Helper_1",
    "h2": "Data_Helper_2",
    "h3": "Data_Helper_3",
}
SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def scene_id_from_name(path: Path) -> int:
    name = path.name
    if "__" in name:
        name = name.split("__", 1)[1]
    m = SCENE_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse scene id from filename: {path.name}")
    return int(m.group(1))

def _collect_all_original_files(repo_root: Path, view: str):
    root = repo_root / "data" / VIEW_FOLDERS[view]
    images, labels = {}, {}
    for split in SOURCE_SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for img in sorted(image_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                sid = scene_id_from_name(img)
            except ValueError:
                continue
            lab = label_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            images.setdefault(sid, img)
            labels.setdefault(sid, lab)
    return images, labels

def _scene_ids_from_combined_split(repo_root: Path, split: str) -> set[int]:
    split_dir = repo_root / "data" / "combined_yolo" / split / "images"
    if not split_dir.exists():
        return set()
    ids = set()
    for p in split_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            ids.add(scene_id_from_name(p))
        except ValueError:
            pass
    return ids

@dataclass(frozen=True)
class ViewFiles:
    image: Path
    label: Path

@dataclass(frozen=True)
class SceneFiles:
    scene_id: int
    ego: ViewFiles
    h1: ViewFiles
    h2: ViewFiles
    h3: ViewFiles

    def helper(self, name: str) -> ViewFiles:
        key = name.lower()
        if key not in {"h1", "h2", "h3"}:
            raise ValueError(f"Unknown helper {name!r}")
        return getattr(self, key)

def discover_paired_scenes(repo_root: str | Path, split: str = "test") -> list[SceneFiles]:
    repo_root = Path(repo_root).resolve()
    image_index, label_index = {}, {}
    for view in ("ego", "h1", "h2", "h3"):
        image_index[view], label_index[view] = _collect_all_original_files(repo_root, view)

    common = set(image_index["ego"])
    for view in ("ego", "h1", "h2", "h3"):
        common &= set(image_index[view])
        common &= set(label_index[view])

    if not common:
        raise RuntimeError("No complete Ego/H1/H2/H3 scenes were found.")

    requested = _scene_ids_from_combined_split(repo_root, split)
    if requested:
        common &= requested
    else:
        print(f"WARNING: combined_yolo/{split} not found; using all complete scenes.")

    scenes = []
    for sid in sorted(common):
        scenes.append(SceneFiles(
            scene_id=sid,
            ego=ViewFiles(image_index["ego"][sid], label_index["ego"][sid]),
            h1=ViewFiles(image_index["h1"][sid], label_index["h1"][sid]),
            h2=ViewFiles(image_index["h2"][sid], label_index["h2"][sid]),
            h3=ViewFiles(image_index["h3"][sid], label_index["h3"][sid]),
        ))
    if not scenes:
        raise RuntimeError(f"No complete paired scenes found for split={split!r}")
    return scenes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", default="test")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()
    scenes = discover_paired_scenes(args.repo_root, args.split)
    print(f"Complete paired scenes for split={args.split!r}: {len(scenes)}")
    for s in scenes[:args.show]:
        print(f"{s.scene_id}: Ego={s.ego.image.name} | H1={s.h1.image.name} | H2={s.h2.image.name} | H3={s.h3.image.name}")
