"""
test_exclusion.py - Verification script for Exclusion Zones & Breakout Guard
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fire_detector import check_exclusion_breakout, FireSmokeStreamAnalyzer
from app import create_app
from models import db, ExclusionZone


def test_breakout_geometry():
    print("=== Test 1: Breakout Guard Geometric Evaluation ===")
    img_w, img_h = 1000, 1000

    # Polygon: square from (100, 100) to (500, 500) -> normalized: [0.1, 0.1] to [0.5, 0.5]
    polygon = [
        [0.1, 0.1],
        [0.5, 0.1],
        [0.5, 0.5],
        [0.1, 0.5],
    ]

    # Case A: Box fully inside the exclusion polygon
    box_inside = [200, 200, 300, 300]
    ignored_a, ratio_a, is_bk_a = check_exclusion_breakout(box_inside, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case A (Fully inside): ignored={ignored_a}, outside_ratio={ratio_a:.2f}, is_breakout={is_bk_a}")
    assert ignored_a is True and is_bk_a is False, "Expected fully inside box to be ignored and not breakout"
    assert ratio_a <= 0.05, "Expected outside ratio ~0"

    # Case B1: Box partially inside (25% outside, 75% inside -> safely tolerated inside)
    box_b1 = [200, 200, 600, 400]
    ignored_b1, ratio_b1, is_bk_b1 = check_exclusion_breakout(box_b1, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case B1 (25% outside): ignored={ignored_b1}, outside_ratio={ratio_b1:.2f}, is_breakout={is_bk_b1}")
    assert ignored_b1 is True and is_bk_b1 is False, "Expected 25% outside to be safely tolerated inside zone"

    # Case B2: Box 75% outside, 25% inside (>40% outside and >=15% inside -> Genuine BREAKOUT!)
    box_b2 = [400, 200, 800, 400]
    ignored_b2, ratio_b2, is_bk_b2 = check_exclusion_breakout(box_b2, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case B2 (75% outside, Breakout): ignored={ignored_b2}, outside_ratio={ratio_b2:.2f}, is_breakout={is_bk_b2}")
    assert ignored_b2 is False and is_bk_b2 is True, "Expected 75% outside to trigger true breakout"
    assert ratio_b2 >= 0.70, "Expected outside ratio >= 0.70"

    # Case C: Box completely outside polygon (0% inside)
    box_outside = [600, 600, 800, 800]
    ignored_c, ratio_c, is_bk_c = check_exclusion_breakout(box_outside, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case C (Completely outside): ignored={ignored_c}, outside_ratio={ratio_c:.2f}, is_breakout={is_bk_c}")
    assert ignored_c is False and is_bk_c is False, "Expected outside box not to be ignored and not breakout"
    assert ratio_c == 1.0, "Expected 100% outside"

    # Case D: Box merely grazing the edge (e.g. 5% inside, 95% outside -> External object, NOT breakout!)
    # Box from x: 480 to 900 (width 420), y: 200 to 400 (height 200).
    # Inside: x from 480 to 500 (width 20), y 200 to 400 (height 200). Inside area = 4,000 / 84,000 = 4.7% inside.
    box_d = [480, 200, 900, 400]
    ignored_d, ratio_d, is_bk_d = check_exclusion_breakout(box_d, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case D (Edge grazing 4.7% inside): ignored={ignored_d}, outside_ratio={ratio_d:.2f}, is_breakout={is_bk_d}")
    assert ignored_d is False and is_bk_d is False, "Expected edge grazing not to trigger false breakout"

    # Case E: Massive Smoke/Fire Cloud engulfing entire exclusion zone (e.g. 100% of zone engulfed, but box is 10x larger)
    # Box from 0 to 1000 (entire frame). Zone is [100, 100] to [500, 500].
    # inter_area = 160,000. bbox_area = 1,000,000. inside_ratio = 16%, poly_covered_ratio = 100%, outside_ratio = 84%
    box_e = [0, 0, 1000, 1000]
    ignored_e, ratio_e, is_bk_e = check_exclusion_breakout(box_e, polygon, img_w, img_h, outside_threshold=0.40)
    print(f"Case E (Massive cloud engulfing zone): ignored={ignored_e}, outside_ratio={ratio_e:.2f}, is_breakout={is_bk_e}")
    assert ignored_e is False and is_bk_e is True, "Expected massive cloud engulfing zone to trigger breakout"

    print(">>> Test 1 Passed Successfully!\n")


def test_rest_api_and_db():
    print("=== Test 2: Database Schema & REST API Endpoints ===")
    app = create_app()
    with app.app_context():
        db.create_all()

        client = app.test_client()

        # 1. Create Exclusion Zone via POST
        poly_sample = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        resp = client.post(
            "/api/cameras/0/exclusion_zones",
            json={
                "name": "Khu vuc han xi #1",
                "zone_type": "welding",
                "polygon_points": poly_sample,
                "is_active": True,
            },
        )
        assert resp.status_code == 201, f"POST failed: {resp.data}"
        data = resp.get_json()
        zone_id = data["id"]
        print(f"Created Exclusion Zone ID={zone_id}: name={data['name']}, points={len(data['polygon_points'])}")

        # 2. Get Exclusion Zones via GET
        resp = client.get("/api/cameras/0/exclusion_zones")
        assert resp.status_code == 200
        zones = resp.get_json()
        assert any(z["id"] == zone_id for z in zones), "Created zone not found in GET"
        print(f"GET returned {len(zones)} zone(s)")

        # 3. Update zone (Toggle active / change name) via PUT
        resp = client.put(
            f"/api/exclusion_zones/{zone_id}",
            json={"name": "Khu vuc han xi #1 (Cap nhat)", "is_active": False},
        )
        assert resp.status_code == 200
        updated = resp.get_json()
        assert updated["name"] == "Khu vuc han xi #1 (Cap nhat)"
        assert updated["is_active"] is False
        print(f"PUT updated zone: is_active={updated['is_active']}, name={updated['name']}")

        # 4. Delete zone via DELETE
        resp = client.delete(f"/api/exclusion_zones/{zone_id}")
        assert resp.status_code == 200
        assert resp.get_json().get("ok") is True
        print(f"DELETE removed zone ID={zone_id}")

    print(">>> Test 2 Passed Successfully!\n")


if __name__ == "__main__":
    try:
        test_breakout_geometry()
        test_rest_api_and_db()
        print("ALL EXCLUSION ZONE & BREAKOUT GUARD TESTS PASSED!")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
