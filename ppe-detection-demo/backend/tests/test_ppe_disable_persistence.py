"""
test_ppe_disable_persistence.py
-------------------------------
Unit tests verifying that when PPE detection is disabled:
1. No stale/ghost PPE bounding box is rendered or retained on screen.
2. Visual persistence cache is strictly rejected when dt < 0 or dt >= 0.1s.
3. detector.reset() properly wipes all cached PPE detections and timestamps.
4. annotate_frame with enable_ppe=False produces zero PPE items and zero person annotations.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PPEDetector


class TestPPEDisablePersistence(unittest.TestCase):
    def setUp(self):
        self.detector = PPEDetector()

    def test_reset_clears_all_ppe_caches(self):
        # Simulate populated cache
        self.detector._last_rendered_persons = [{"bbox": [50, 50, 150, 200], "id": 1, "helmet": "ok", "vest": "ok", "mask": "ok"}]
        self.detector._last_rendered_ppe_items = [{"bbox": [60, 50, 90, 80], "type": "helmet", "status": "ok", "conf": 0.9, "label": "Mu"}]
        self.detector._last_ppe_inference_time = 1790793641.0

        self.detector.reset()

        self.assertEqual(len(self.detector._last_rendered_persons), 0)
        self.assertEqual(len(self.detector._last_rendered_ppe_items), 0)
        self.assertEqual(self.detector._last_ppe_inference_time, 0.0)

    def test_annotate_frame_enable_ppe_false_draws_no_boxes_and_ignores_stale_cache(self):
        # Populate stale cache from a past run
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "violation",
            "vest": "violation",
            "mask": "violation",
            "conf": 0.9,
        }]
        self.detector._last_rendered_ppe_items = [{
            "bbox": [60, 50, 90, 80],
            "type": "helmet",
            "status": "violation",
            "conf": 0.9,
            "label": "Khong mu",
        }]
        self.detector._last_ppe_inference_time = 1790793641.0

        # Video timestamp at t = 0.04s (typical video upload timestamp)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_copy = frame.copy()

        annotated, has_violation, ppe_stats, violations, falls, fires, pids = self.detector.annotate_frame(
            frame,
            timestamp=0.04,
            enable_ppe=False,
        )

        # 1. Returned data must be completely empty
        self.assertEqual(len(violations), 0)
        self.assertEqual(len(pids), 0)
        self.assertEqual(ppe_stats["helmet"]["violation"], 0)

        # 2. Frame pixels must NOT be modified by any PPE rectangles
        np.testing.assert_array_equal(annotated, frame_copy, "Frame should not have any PPE annotations drawn when enable_ppe=False")

        # 3. Detector cache must be wiped
        self.assertEqual(len(self.detector._last_rendered_persons), 0)
        self.assertEqual(len(self.detector._last_rendered_ppe_items), 0)

    def test_visual_persistence_cache_dt_window(self):
        # Setup valid cache at t = 100.0s
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "ok",
            "vest": "ok",
            "mask": "ok",
            "conf": 0.9,
        }]
        self.detector._last_rendered_ppe_items = []
        self.detector._last_ppe_inference_time = 100.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Case A: Negative dt (video timestamp vs wall clock) at t = 0.5s -> MUST BE REJECTED
        _ann, _viol, _stats, _v, _f, _fi, pids_neg = self.detector.annotate_frame(
            frame.copy(),
            timestamp=0.5,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_neg), 0, "Negative dt must NOT use cache")

        # Case B: Within 1-frame off-slot window at t = 100.033s (< 0.1s) -> ALLOWED
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "ok",
            "vest": "ok",
            "mask": "ok",
            "conf": 0.9,
        }]
        self.detector._last_ppe_inference_time = 100.0
        _ann, _viol, _stats, _v, _f, _fi, pids_ok = self.detector.annotate_frame(
            frame.copy(),
            timestamp=100.033,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_ok), 1, "dt < 0.1s with enable_ppe=True should use smooth persistence")

        # Case C: Expired window at t = 100.25s (> 0.1s) -> EXPIRED
        _ann, _viol, _stats, _v, _f, _fi, pids_exp = self.detector.annotate_frame(
            frame.copy(),
            timestamp=100.25,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_exp), 0, "dt > 0.1s must expire and clear")


if __name__ == "__main__":
    unittest.main()
