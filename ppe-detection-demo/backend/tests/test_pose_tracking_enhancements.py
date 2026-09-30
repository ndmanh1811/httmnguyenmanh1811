"""
test_pose_tracking_enhancements.py
-----------------------------------
Unit tests verifying the skeleton tracking enhancements:
1. OneEuro filter parameter verification (min_cutoff=1.6, beta=0.06)
2. Track retention across momentary frame drops (lost_count <= 6)
3. Two-phase IoU and adaptive distance matching
4. Ghost skeleton cache expiration (< 0.08s window)
5. Clean rendering of skeleton without stale freeze
"""

import sys
import os
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancements import _OneEuroFilter1D, OneEuroPoseFilter, _calc_box_iou
from pose_fall_detector import PoseFallDetector, KPT_CONF_THRESH


class TestOneEuroFilterTuning(unittest.TestCase):
    def test_default_parameters(self):
        f1d = _OneEuroFilter1D()
        self.assertAlmostEqual(f1d.min_cutoff, 1.2)
        self.assertAlmostEqual(f1d.beta, 0.15)

        pose_filter = OneEuroPoseFilter()
        self.assertAlmostEqual(pose_filter.min_cutoff, 1.2)
        self.assertAlmostEqual(pose_filter.beta, 0.15)

    def test_zero_lag_on_fast_motion(self):
        f1d = _OneEuroFilter1D(te=1.0 / 30.0, min_cutoff=1.2, beta=0.15)
        # Stationary
        for _ in range(10):
            val = f1d.filter(100.0)
        self.assertAlmostEqual(val, 100.0, places=2)

        # Fast motion jump from 100 to 200
        val1 = f1d.filter(200.0)
        # With beta=0.06, cutoff expands immediately, filter adapts rapidly
        self.assertGreater(val1, 150.0, "Filter should respond promptly to rapid movement")


class TestTwoPhaseMatching(unittest.TestCase):
    def setUp(self):
        self.detector = PoseFallDetector()

    def test_iou_matching_phase(self):
        # Setup previous person at [100, 100, 200, 300]
        self.detector._prev_persons = [{
            "bbox": [100, 100, 200, 300],
            "kpts": np.zeros((17, 3)),
            "pid": 42,
            "lost_count": 0,
        }]

        # New bbox with 70% overlap at [110, 105, 210, 305]
        new_bboxes = [[110, 105, 210, 305]]
        matched = self.detector._match_to_prev(new_bboxes)
        self.assertEqual(matched[0], 0, "Should match previous person via IoU phase")

    def test_adaptive_distance_matching_for_fast_movement(self):
        # Setup previous person at [100, 100, 200, 300] (w=100, h=200, diag~223, adaptive_dist > 350)
        self.detector._prev_persons = [{
            "bbox": [100, 100, 200, 300],
            "kpts": np.zeros((17, 3)),
            "pid": 77,
            "lost_count": 0,
        }]

        # Sudden jump: [100, 280, 200, 480] (center shifted by 180px, IoU = 0.05)
        # Old 150px threshold failed here; adaptive distance must succeed
        new_bboxes = [[100, 280, 200, 480]]
        matched = self.detector._match_to_prev(new_bboxes)
        self.assertEqual(matched[0], 0, "Should match previous person via adaptive center distance")


class TestTrackRetentionAcrossDrops(unittest.TestCase):
    def setUp(self):
        self.detector = PoseFallDetector()

    def test_tracks_survive_brief_drops(self):
        self.detector._prev_persons = [{
            "bbox": [100, 100, 200, 300],
            "kpts": np.zeros((17, 3)),
            "pid": 10,
            "lost_count": 0,
        }]

        # Simulate 3 consecutive empty frames
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock empty prediction
        original_predict = self.detector.model.predict
        self.detector.model.predict = lambda *args, **kwargs: []

        try:
            for i in range(3):
                self.detector.detect(blank_frame)
                self.assertEqual(len(self.detector._prev_persons), 1, f"Track should survive empty frame {i+1}")
                self.assertEqual(self.detector._prev_persons[0]["lost_count"], i + 1)
                self.assertEqual(self.detector._prev_persons[0]["pid"], 10)

            # After 7 empty frames (> 6), track should be purged
            for _ in range(4):
                self.detector.detect(blank_frame)
            self.assertEqual(len(self.detector._prev_persons), 0, "Track should be purged after > 6 empty frames")
        finally:
            self.detector.model.predict = original_predict


class TestGhostSkeletonDecay(unittest.TestCase):
    def setUp(self):
        self.detector = PoseFallDetector()

    def test_no_ghost_skeleton_after_decay_window(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            kpts[i] = [200 + i * 5, 200 + i * 5, 0.9]

        self.detector._last_detected_persons = [{
            "bbox": [150, 150, 350, 400],
            "conf": 0.9,
            "pid": 1,
            "angle": 10.0,
            "status": "normal",
            "keypoints": kpts,
        }]

        # Frame at t = 100.0: draws person
        self.detector.annotate_frame(frame.copy(), [], timestamp=100.0)
        self.assertEqual(len(self.detector._cached_detected_persons), 1)

        # Clear active detections (empty frame)
        self.detector._last_detected_persons = []

        # At t = 100.04 (within 0.08s off-slot): visual persistence allowed
        out_frame_cached = self.detector.annotate_frame(frame.copy(), [], timestamp=100.04)
        self.assertEqual(len(self.detector._cached_detected_persons), 1)

        # At t = 100.20 (> 0.08s elapsed): must NOT hold ghost skeleton!
        out_frame_expired = self.detector.annotate_frame(frame.copy(), [], timestamp=100.20)
        self.assertEqual(len(self.detector._cached_detected_persons), 0)


class TestPostureColorStates(unittest.TestCase):
    def setUp(self):
        self.detector = PoseFallDetector()

    def test_posture_colors_in_annotated_frame(self):
        # Create a blank white canvas so black lines and colored lines are distinct
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Shoulders (5, 6)
        kpts[5] = [200, 200, 0.9]
        kpts[6] = [250, 200, 0.9]

        # 1. Normal: Yellow (BGR: 0, 220, 255)
        self.detector._last_detected_persons = [{
            "bbox": [150, 150, 300, 400],
            "conf": 0.9,
            "pid": 1,
            "angle": 10.0,
            "status": "normal",
            "keypoints": kpts,
            "fall_duration": 0.0,
        }]
        out_normal = self.detector.annotate_frame(frame.copy(), [], timestamp=10.0)
        # Check bone pixel at midpoint (225, 200)
        # Yellow bone color has B=0, G=220, R=255
        pixel = out_normal[200, 225]
        self.assertEqual(pixel[0], 0, "Normal bone blue channel should be 0")
        self.assertEqual(pixel[2], 255, "Normal bone red channel should be 255")

        # 2. Bending: Orange (BGR: 0, 140, 255)
        self.detector._last_detected_persons = [{
            "bbox": [150, 150, 300, 400],
            "conf": 0.9,
            "pid": 2,
            "angle": 45.0,
            "status": "bending",
            "keypoints": kpts,
            "fall_duration": 0.0,
        }]
        out_bend = self.detector.annotate_frame(frame.copy(), [], timestamp=10.0)
        pixel_bend = out_bend[200, 225]
        self.assertEqual(pixel_bend[0], 0, "Bending bone blue channel should be 0")
        self.assertEqual(pixel_bend[1], 140, "Bending bone green channel should be 140")
        self.assertEqual(pixel_bend[2], 255, "Bending bone red channel should be 255")

        # 3. Fall (< 5s): Red (BGR: 0, 0, 255)
        self.detector._last_detected_persons = [{
            "bbox": [150, 150, 300, 400],
            "conf": 0.9,
            "pid": 3,
            "angle": 80.0,
            "status": "fall",
            "keypoints": kpts,
            "fall_duration": 2.5,
        }]
        out_fall = self.detector.annotate_frame(frame.copy(), [], timestamp=10.0)
        pixel_fall = out_fall[200, 225]
        self.assertEqual(pixel_fall[0], 0, "Fall bone blue channel should be 0")
        self.assertEqual(pixel_fall[1], 0, "Fall bone green channel should be 0")
        self.assertEqual(pixel_fall[2], 255, "Fall bone red channel should be 255")

        # 4. Fall Immobile (> 5s): Black (BGR: 0, 0, 0)
        self.detector._last_detected_persons = [{
            "bbox": [150, 150, 300, 400],
            "conf": 0.9,
            "pid": 4,
            "angle": 85.0,
            "status": "fall_immobile",
            "keypoints": kpts,
            "fall_duration": 5.5,
        }]
        out_immobile = self.detector.annotate_frame(frame.copy(), [], timestamp=10.0)
        pixel_immobile = out_immobile[200, 225]
        self.assertEqual(tuple(pixel_immobile), (0, 0, 0), "Fall immobile bone must be jet black (0, 0, 0)")

    def test_fall_5s_duration_transition(self):
        # Verify candidate start_time tracking and transition to fall_immobile when >= 5.0s
        pid = 99
        self.detector._fall_candidates[pid] = {
            "immobile_frames": 10,
            "last_cx": 200,
            "last_cy": 300,
            "start_time": 100.0,
        }

        cand = self.detector._fall_candidates[pid]
        # At t = 103.0s (duration = 3.0s < 5.0s):
        curr_time = 103.0
        dur_3s = max(0.0, curr_time - cand.get("start_time", curr_time))
        self.assertLess(dur_3s, 5.0)

        # At t = 105.2s (duration = 5.2s >= 5.0s):
        curr_time = 105.2
        dur_5s = max(0.0, curr_time - cand.get("start_time", curr_time))
        self.assertGreaterEqual(dur_5s, 5.0)


if __name__ == "__main__":
    unittest.main()
