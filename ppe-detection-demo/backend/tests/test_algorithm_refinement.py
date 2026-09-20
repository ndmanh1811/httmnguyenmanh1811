"""
test_algorithm_refinement.py - Verification script for Option 1 Computer Vision Algorithms
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fire_detector import (
    extract_fire_smoke_contour,
    compute_plume_optical_flow,
    check_exclusion_breakout,
    FireSmokeStreamAnalyzer,
    FireSmokeModel,
)


def test_contour_extraction():
    print("=== Test 1: Contour & Mask Extraction ===")
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # 1. Fire simulation: bright yellow/orange circle at center
    cv2.circle(frame, (200, 200), 50, (0, 165, 255), -1)  # BGR orange
    bbox_fire = [140, 140, 260, 260]
    cnt_fire, area_fire = extract_fire_smoke_contour(frame, bbox_fire, "fire")
    print(f"Fire contour points: {len(cnt_fire) if cnt_fire is not None else 0}, area: {area_fire}")
    assert cnt_fire is not None and len(cnt_fire) >= 3, "Expected valid fire contour"
    assert area_fire > 1000, "Expected significant fire area"

    # 2. Smoke simulation: gray/dark cloud
    cv2.circle(frame, (100, 100), 40, (110, 110, 110), -1)  # Neutral gray
    bbox_smoke = [50, 50, 150, 150]
    cnt_smoke, area_smoke = extract_fire_smoke_contour(frame, bbox_smoke, "smoke")
    print(f"Smoke contour points: {len(cnt_smoke) if cnt_smoke is not None else 0}, area: {area_smoke}")
    assert cnt_smoke is not None and len(cnt_smoke) >= 3, "Expected valid smoke contour"
    assert area_smoke > 1000, "Expected significant smoke area"

    print(">>> Test 1 Passed Successfully!\n")


def test_optical_flow():
    print("=== Test 2: Upward Plume Optical Flow ===")
    f1 = np.zeros((300, 300), dtype=np.uint8)
    f2 = np.zeros((300, 300), dtype=np.uint8)

    np.random.seed(42)
    texture = (np.random.rand(60, 60) * 200).astype(np.uint8)
    f1[120:180, 120:180] = texture
    f2[112:172, 120:180] = texture  # shifted up by 8 pixels (dy = -8)

    bbox = [110, 110, 190, 190]
    upi_upward = compute_plume_optical_flow(f1, f2, bbox)
    print(f"Upward Motion UPI: {upi_upward:.2f}")
    assert upi_upward >= 0.70, f"Expected strong upward UPI >= 0.70, got {upi_upward}"

    # Stationary test: f2 identical to f1
    upi_static = compute_plume_optical_flow(f1, f1, bbox)
    print(f"Stationary Motion UPI: {upi_static:.2f}")
    assert upi_static <= 0.40, f"Expected stationary UPI <= 0.40, got {upi_static}"

    print(">>> Test 2 Passed Successfully!\n")


def test_contour_exclusion():
    print("=== Test 3: Contour-based Exclusion Breakout ===")
    img_w, img_h = 1000, 1000
    polygon = [
        [0.2, 0.2],
        [0.5, 0.2],
        [0.5, 0.5],
        [0.2, 0.5],
    ]
    contour_inside = np.array([
        [250, 250],
        [400, 250],
        [325, 400],
    ])
    ignored, out_ratio, is_bk = check_exclusion_breakout(
        bbox=[250, 250, 400, 400],
        polygon_normalized=polygon,
        img_w=img_w,
        img_h=img_h,
        contour=contour_inside,
    )
    print(f"Contour Inside: ignored={ignored}, out_ratio={out_ratio:.2f}, is_bk={is_bk}")
    assert ignored is True and is_bk is False, "Expected contour inside to be ignored"

    contour_burst = np.array([
        [250, 250],
        [850, 250],
        [550, 700],
    ])
    ignored_b, out_ratio_b, is_bk_b = check_exclusion_breakout(
        bbox=[250, 250, 850, 700],
        polygon_normalized=polygon,
        img_w=img_w,
        img_h=img_h,
        contour=contour_burst,
    )
    print(f"Contour Burst: ignored={ignored_b}, out_ratio={out_ratio_b:.2f}, is_bk={is_bk_b}")
    assert ignored_b is False and is_bk_b is True, "Expected bursting contour to trigger breakout"

    print(">>> Test 3 Passed Successfully!\n")


if __name__ == "__main__":
    test_contour_extraction()
    test_optical_flow()
    test_contour_exclusion()
    print("ALL OPTION 1 ALGORITHMIC REFINEMENT TESTS PASSED!")
