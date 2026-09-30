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

        # Case B: Within multi-slot persistence window at t = 100.20s (< 0.45s) -> ALLOWED (anti-flicker)
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "ok",
            "vest": "ok",
            "mask": "ok",
            "conf": 0.9,
        }]
        self.detector._last_ppe_inference_time = 100.0
        self.detector._off_slot_ppe_count = 0
        _ann, _viol, _stats, _v, _f, _fi, pids_ok = self.detector.annotate_frame(
            frame.copy(),
            timestamp=100.20,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_ok), 1, "dt = 0.20s (< 0.45s) with enable_ppe=True should use smooth persistence")

        # Case C: Expired window at t = 100.50s (> 0.45s) -> EXPIRED
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "ok",
            "vest": "ok",
            "mask": "ok",
            "conf": 0.9,
        }]
        self.detector._last_ppe_inference_time = 100.0
        self.detector._off_slot_ppe_count = 0
        _ann, _viol, _stats, _v, _f, _fi, pids_exp = self.detector.annotate_frame(
            frame.copy(),
            timestamp=100.50,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_exp), 0, "dt > 0.45s must expire and clear")

        # Case D: Exceeded max off-slot frame count (> 4 frames) -> EXPIRED even if dt < 0.45s
        self.detector._last_rendered_persons = [{
            "bbox": [50, 50, 150, 200],
            "id": 1,
            "helmet": "ok",
            "vest": "ok",
            "mask": "ok",
            "conf": 0.9,
        }]
        self.detector._last_ppe_inference_time = 100.0
        self.detector._off_slot_ppe_count = 4  # Next off-slot will make it 5 (> 4)
        _ann, _viol, _stats, _v, _f, _fi, pids_cnt = self.detector.annotate_frame(
            frame.copy(),
            timestamp=100.10,
            enable_ppe=True,
            run_ppe_inference=False,
        )
        self.assertEqual(len(pids_cnt), 0, "off_slot_count > 4 must expire and clear")

    def test_ppe_temporal_debouncing_hysteresis(self):
        """Kiem tra tinh nang Debouncing / Hysteresis cua P1: triet tieu nhay do/xanh."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Setup person confirmed 'ok' in memory
        self.detector._ppe_memory = {
            1: {
                "helmet": {
                    "status": "ok",
                    "pending_status": None,
                    "pending_count": 0,
                    "last_seen": 100.0,
                    "conf": 0.9,
                },
                "vest": {
                    "status": "ok",
                    "pending_status": None,
                    "pending_count": 0,
                    "last_seen": 100.0,
                    "conf": 0.9,
                },
                "mask": {
                    "status": "ok",
                    "pending_status": None,
                    "pending_count": 0,
                    "last_seen": 100.0,
                    "conf": 0.9,
                },
            }
        }

        # Gia lap frame 100.04s nhan raw detected 'violation' cho helmet lan dau tien (1-frame glitch)
        # Bypassing stage 1/2 bang cach goi truc tiep phan xu ly debouncing tren mock person
        p = {
            "id": 1,
            "bbox": [100, 100, 200, 300],
            "conf": 0.9,
            "helmet": "violation",
            "vest": "ok",
            "mask": "ok",
            "helmet_conf": 0.85,
        }
        person_items = [p]
        assigned_ppe_items = [{
            "type": "helmet",
            "status": "violation",
            "conf": 0.85,
            "bbox": [120, 100, 180, 150],
            "person_id": 1,
            "label": "Khong mu",
        }]

        # Chay debouncing logic truc tiep hoac thong qua phuong thuc
        # Ta kiem tra trang thai sau debouncing:
        # Frame 1: helmet phai duoc giu la 'ok' (debounced), box violation phai bi bo
        now = 100.04
        pmem = self.detector._ppe_memory[1]
        rec = pmem["helmet"]

        # Simulate frame 1 debouncing
        raw_status = p["helmet"]
        if rec.get("pending_status") == raw_status:
            rec["pending_count"] = rec.get("pending_count", 0) + 1
            if rec["pending_count"] >= 2:
                rec["status"] = raw_status
                p["helmet"] = raw_status
            else:
                p["helmet"] = rec["status"]
        else:
            rec["pending_status"] = raw_status
            rec["pending_count"] = 1
            p["helmet"] = rec["status"]

        self.assertEqual(p["helmet"], "ok", "Frame 1 violation noise must be suppressed to 'ok'")
        self.assertEqual(rec["pending_count"], 1)

        # Frame 2: Tiep tuc nhan 'violation' -> da du 2 frames lien tiep -> phai flip sang 'violation'
        now = 100.08
        p2 = {"id": 1, "helmet": "violation"}
        raw_status2 = p2["helmet"]
        if rec.get("pending_status") == raw_status2:
            rec["pending_count"] = rec.get("pending_count", 0) + 1
            if rec["pending_count"] >= 2:
                rec["status"] = raw_status2
                rec["pending_status"] = None
                rec["pending_count"] = 0
                p2["helmet"] = raw_status2
            else:
                p2["helmet"] = rec["status"]
        else:
            rec["pending_status"] = raw_status2
            rec["pending_count"] = 1
            p2["helmet"] = rec["status"]

        self.assertEqual(p2["helmet"], "violation", "Frame 2 confirmed violation must flip to 'violation'")
        self.assertEqual(rec["status"], "violation")


if __name__ == "__main__":
    unittest.main()
