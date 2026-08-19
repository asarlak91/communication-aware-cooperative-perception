import numpy as np

from core.evaluation import evaluate_detections
from core.helper_selection import dinkelbach_select, exact_select
from core.packet_mask import apply_packet_mask, expected_dropped_blocks
from core.perception_fusion import Detection, compute_iou
from core.perception_metrics import build_perception_features
from core.scenario import generate_scenario


def test_dinkelbach_exact_matches_direct_exact():
    scenario = generate_scenario(N=8, horizon_s=3.0, seed=7)
    features = build_perception_features(
        scenario.positions,
        scenario.ego_positions,
        scenario.velocities,
        sensor_max=np.inf,
    )
    exact = exact_select(features, M=3)
    dink, _ = dinkelbach_select(features, M=3, subproblem="exact")
    assert set(exact.tolist()) == set(dink.tolist())


def test_packet_mask_alpha_one_keeps_image():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    corrupted, dropped = apply_packet_mask(
        image,
        alpha=1.0,
        grid=4,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(corrupted, image)
    assert not dropped.any()
    assert expected_dropped_blocks(alpha=1.0, grid=4) == 0.0


def test_compute_iou_identical_boxes():
    box = np.array([10.0, 20.0, 30.0, 40.0])
    assert compute_iou(box, box) == 1.0


def test_evaluate_detections_perfect_match():
    gt = [Detection(np.array([0.0, 0.0, 10.0, 10.0]), 1.0, 0)]
    pred = [Detection(np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 0)]
    result = evaluate_detections(pred, gt, iou_threshold=0.5)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["mean_iou"] == 1.0
