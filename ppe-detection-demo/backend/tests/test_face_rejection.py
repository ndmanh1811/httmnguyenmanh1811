"""
test_face_rejection.py - Test person head suppression and static object penalty
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fire_detector import FireScorer, FireSmokeStreamAnalyzer


def test_person_head_suppression():
    print("=== Test: Person Head Suppression ===")
    analyzer = FireSmokeStreamAnalyzer(fire_conf=0.55)

    # Person detected by PPE detector from (100, 100) to (300, 600)
    person_boxes = [[100, 100, 300, 600]]

    # Simulated candidate: YOLO detects a "fire" on the person's face (150, 120, 250, 220) with conf=0.58
    # Mocking predict_candidates to return this candidate
    analyzer.model.predict_candidates = lambda frame, f_conf, s_conf: [
        {"type": "fire", "confidence": 0.58, "bbox": [150, 120, 250, 220], "cls_id": 1}
    ]

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hazards = analyzer.detect(dummy_frame, timestamp=1.0, person_boxes=person_boxes)

    print(f"Hazards detected when face overlaps person head: {len(hazards)}")
    assert len(hazards) == 0, "Expected fire on person face to be suppressed!"
    print(">>> Person Head Suppression Passed!\n")


def test_static_object_penalty():
    print("=== Test: Static Object Rejection ===")
    scorer = FireScorer()

    # Track 1: Completely static warm object (e.g. face or lamp) across 6 frames
    # History with identical bounding box and no area variation
    fixed_bbox = [200, 200, 300, 300]
    static_track = {
        "id": 1,
        "type": "fire",
        "bbox": fixed_bbox,
        "conf": 0.58,
        "first_seen": 0.0,
        "last_seen": 1.0,
        "hits": 6,
        "last_iou": 0.98,
        "history": [(t * 0.2, fixed_bbox, 0.58) for t in range(6)],
    }

    score_static, comp_static = scorer.calculate(static_track, current_ts=1.0, persistence_window=0.8)
    print(f"Static Object Score: {score_static:.2f} (Trigger Threshold is 0.65)")
    assert score_static < 0.65, f"Expected static object score < 0.65, got {score_static}"

    # Track 2: Real flickering flame with dynamic area fluctuations
    flame_hist = [
        (0.0, [200, 200, 300, 300], 0.75),
        (0.2, [195, 198, 308, 305], 0.78),
        (0.4, [202, 192, 296, 315], 0.82),
        (0.6, [198, 195, 312, 310], 0.80),
        (0.8, [201, 190, 305, 320], 0.85),
    ]
    flicker_track = {
        "id": 2,
        "type": "fire",
        "bbox": [201, 190, 305, 320],
        "conf": 0.85,
        "first_seen": 0.0,
        "last_seen": 0.8,
        "hits": 5,
        "last_iou": 0.82,
        "history": flame_hist,
    }

    score_flame, comp_flame = scorer.calculate(flicker_track, current_ts=0.8, persistence_window=0.8)
    print(f"Real Flame Score: {score_flame:.2f}")
    assert score_flame >= 0.70, f"Expected flame score >= 0.70, got {score_flame}"

    print(">>> Static Object Rejection Passed!\n")


if __name__ == "__main__":
    test_person_head_suppression()
    test_static_object_penalty()
    print("ALL FACE REJECTION & STATIC VETO TESTS PASSED!")
