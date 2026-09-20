import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
test_smoke_suppression.py
Tests the cross-module smoke suppression and skeleton integrity gating.
"""

import numpy as np
import cv2
from pose_fall_detector import PoseFallDetector
from detector import PPEDetector, _is_in_smoke_or_fire


def test_skeleton_integrity_gate():
    print("=== TEST 1: SKELETON INTEGRITY GATE ===")
    pfd = PoseFallDetector()

    # Case 1: Phantom keypoints (only 2 random points in smoke, conf=0.35)
    phantom_kpts = np.zeros((17, 3), dtype=np.float32)
    phantom_kpts[0] = [100, 100, 0.35]  # Nose
    phantom_kpts[1] = [102, 101, 0.38]  # Eye
    bbox = [50, 50, 250, 250]

    valid, reason = pfd._is_valid_human_skeleton(phantom_kpts, bbox, is_in_smoke=False)
    assert not valid, f"Expected invalid phantom skeleton, got valid! ({reason})"
    print(f"Phantom keypoints correctly rejected: {reason}")

    # Case 2: Clustered keypoints inside smoke without torso core
    smoke_cluster = np.zeros((17, 3), dtype=np.float32)
    for i in range(5):
        smoke_cluster[i] = [100 + i, 100 + i, 0.45]
    valid_smoke, reason_smoke = pfd._is_valid_human_skeleton(smoke_cluster, bbox, is_in_smoke=True)
    assert not valid_smoke, f"Expected invalid smoke cluster, got valid! ({reason_smoke})"
    print(f"Smoke cluster correctly rejected: {reason_smoke}")

    # Case 3: Real human skeleton (shoulders, hips, head, knees)
    real_kpts = np.zeros((17, 3), dtype=np.float32)
    real_kpts[0] = [150, 70, 0.85]   # Nose
    real_kpts[5] = [120, 100, 0.80]  # Left shoulder
    real_kpts[6] = [180, 100, 0.82]  # Right shoulder
    real_kpts[11] = [130, 170, 0.78] # Left hip
    real_kpts[12] = [170, 170, 0.79] # Right hip
    real_kpts[13] = [125, 220, 0.75] # Left knee
    real_kpts[14] = [175, 220, 0.76] # Right knee

    valid_real, reason_real = pfd._is_valid_human_skeleton(real_kpts, bbox, is_in_smoke=False)
    assert valid_real, f"Expected valid real human skeleton, rejected: {reason_real}"
    print("Real human skeleton correctly accepted!")
    print(">>> Test 1 Passed Successfully!")


def test_hazard_spatial_overlap():
    print("\n=== TEST 2: HAZARD OVERLAP CHECK ===")
    smoke_boxes = [[100, 100, 300, 300]]
    fire_boxes = []

    # Box inside smoke (overlap ~ 100%)
    box_inside = [120, 120, 280, 280]
    assert _is_in_smoke_or_fire(box_inside, smoke_boxes, threshold=0.28)
    print("Box inside smoke correctly flagged as in hazard!")

    # Box far away from smoke (overlap = 0%)
    box_outside = [500, 500, 600, 700]
    assert not _is_in_smoke_or_fire(box_outside, smoke_boxes, threshold=0.28)
    print("Box outside smoke correctly passed without flag!")
    print(">>> Test 2 Passed Successfully!")


if __name__ == "__main__":
    test_skeleton_integrity_gate()
    test_hazard_spatial_overlap()
    print("\nALL SMOKE SUPPRESSION & SKELETON INTEGRITY TESTS PASSED!")
