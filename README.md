# Communication-Aware Cooperative Perception

This repository is a clean Python reimplementation of selected experiments from our paper:

**Extended Visibility of Autonomous Vehicles via Optimized Cooperative Perception under Imperfect Communication**  
*Transportation Research Part C: Emerging Technologies*, 2025  
DOI: https://doi.org/10.1016/j.trc.2025.105350

The repository focuses on two parts:

1. **Helper selection** using vehicle location, visual range, and motion blur.
2. **Cooperative perception** using a fine-tuned YOLOv8 pedestrian detector with simulated packet loss on helper images.

This is a reconstructed Python implementation for reproducibility and demonstration. It is not the original MATLAB/CARLA codebase used for every result in the paper.

## Installation

```bash
git clone https://github.com/asarlak91/communication-aware-cooperative-perception.git
cd communication-aware-cooperative-perception

pip install -r requirements.txt
```

## Dataset

The dataset used for the YOLOv8 experiment is available from the GitHub Release:

https://github.com/asarlak91/communication-aware-cooperative-perception/releases/tag/v1.0-dataset

Download:

cooperative_perception_dataset.zip

Extract it inside the repository so the data directory has this structure:

data/
├── Data_Ego_Vehicle/
├── Data_Helper_1/
├── Data_Helper_2/
└── Data_Helper_3/

Each vehicle-view directory contains YOLO-format images and labels organized into train, valid, and test folders.

The training script creates data/combined_yolo/ automatically. This generated directory should not be included in the downloaded dataset.

## 1. Helper-selection experiment

The helper-selection objective uses three quantities:

- distance/location cost,
- effective visual range,
- motion blur.

This implementation applies per-scenario normalization before optimization. The raw quantities are still used when reporting physical metrics.

Run:

```bash
python run_benchmarks.py --seeds 100 --N 50 --M-min 3 --M-max 3
```

Results are saved in:

```text
results/generated/
├── generated_objective_comparison.png
├── helper_selection_benchmark.csv
└── helper_selection_summary.csv
```

Example result included in this repository:

| Method | Mean objective ↓ |
|---|---:|
| Proposed | **3.023** |
| Greedy | 3.105 |
| Velocity | 3.158 |
| Random | 4.273 |
| Proximity | 6.786 |

The objective is minimized, so lower is better.

## 2. Fine-tune YOLOv8

The training script combines the ego and three helper views and fine-tunes a YOLOv8 pedestrian detector.

```bash
python train_yolov8.py \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 8 \
  --rebuild-dataset
```

The best checkpoint is saved under:

```text
results/training/yolov8_pedestrian/weights/best.pt
```

Training outputs and model weights are not committed to the repository.

## 3. Cooperative-perception experiment

The helper image is divided into a 4 × 4 grid. Each block is removed independently with probability:

```text
1 - alpha
```

where `alpha` is the assumed link success probability.

Example:

```bash
python run_yolo_experiment.py \
  --weights results/training/yolov8_pedestrian/weights/best.pt \
  --split test \
  --alpha-h1 0.90 \
  --alpha-h2 0.75 \
  --alpha-h3 0.55
```

For this experiment, the helper with the highest supplied `alpha` is used for the perception test. The helper-selection optimization is evaluated separately in `run_benchmarks.py`.

The experiment reports ego, clean-helper, corrupted-helper, and fused metrics. The paper-style fused IoU is evaluated as:

```text
IoU_s = max(IoU_ego, IoU_helper)
```

The included held-out test result uses 25 paired scenes:

| Metric | Ego | Helper clean | Helper after masking | Fused |
|---|---:|---:|---:|---:|
| IoU | 0.393 | 0.814 | 0.801 | **0.840** |
| Recall | 0.362 | 0.907 | 0.913 | 0.883 |
| F1 | 0.336 | 0.903 | 0.908 | 0.884 |

The average number of dropped blocks was 1.36 out of 16 for the selected helper with `alpha = 0.90`.

Results are saved under:

```text
results/yolo_experiment/
```

## Tests

```bash
python -m pytest -q
```

## Notes

- The helper-selection and YOLO experiments are intentionally kept as separate experiments.
- The generated results in this repository come from the Python reimplementation and should not be treated as exact reproductions of every numerical value in the paper.
- Large datasets, trained weights, caches, and training runs are excluded from Git.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{sarlak2025extended,
  title={Extended visibility of autonomous vehicles via optimized cooperative perception under imperfect communication},
  author={Sarlak, Ahmad and Amin, Rahul and Razi, Abolfazl},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={180},
  pages={105350},
  year={2025},
  publisher={Elsevier}
}
```
