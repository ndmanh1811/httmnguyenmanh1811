"""
test_architecture_baseline.py
-----------------------------
Unit and Integration tests for Architecture Baseline v1 implementation:
1. FixedSlotScheduler (Phase-shifted non-colliding slots)
2. IncidentLifecycleManager Policy Extension (PPE_POLICY, FALL_POLICY, FIRE_POLICY, cls_key resolution)
3. supervision.PolygonZone worker boundary check
4. CameraManager Integration (_check_danger_zones, _check_tick, _check_fall via FALL_POLICY)
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_scheduler import FixedSlotScheduler
from incident_lifecycle import (
    IncidentLifecycleManager,
    IncidentState,
    PPE_POLICY,
    FALL_POLICY,
    FIRE_POLICY,
)
from detector import check_workers_in_polygon_zone
from camera_manager import CameraManager


class TestFixedSlotScheduler(unittest.TestCase):
    def test_slot_cycle_and_collision_free(self):
        scheduler = FixedSlotScheduler()

        expected_schedule = [
            # Slot 0: PPE + Fire
            {"slot": 0, "run_ppe": True, "run_fall": False, "run_fire": True},
            # Slot 1: Fall
            {"slot": 1, "run_ppe": False, "run_fall": True, "run_fire": False},
            # Slot 2: Fall
            {"slot": 2, "run_ppe": False, "run_fall": True, "run_fire": False},
            # Slot 3: PPE + Fire
            {"slot": 3, "run_ppe": True, "run_fall": False, "run_fire": True},
            # Slot 4: Fall
            {"slot": 4, "run_ppe": False, "run_fall": True, "run_fire": False},
            # Slot 5: PPE
            {"slot": 5, "run_ppe": True, "run_fall": False, "run_fire": False},
        ]

        for i, exp in enumerate(expected_schedule):
            res = scheduler.should_run(enable_ppe=True, enable_fall=True, enable_fire=True)
            self.assertEqual(res["slot"], exp["slot"], f"Frame {i}: slot mismatch")
            self.assertEqual(res["run_ppe"], exp["run_ppe"], f"Frame {i}: PPE mismatch")
            self.assertEqual(res["run_fall"], exp["run_fall"], f"Frame {i}: Fall mismatch")
            self.assertEqual(res["run_fire"], exp["run_fire"], f"Frame {i}: Fire mismatch")
            # Invariant: Fall and Fire must NEVER run simultaneously
            self.assertFalse(res["run_fall"] and res["run_fire"], f"Frame {i}: Collision! Fall and Fire ran together!")

    def test_relative_frequencies_over_60_frames(self):
        scheduler = FixedSlotScheduler()
        ppe_count = 0
        fall_count = 0
        fire_count = 0

        for _ in range(60):
            res = scheduler.should_run(enable_ppe=True, enable_fall=True, enable_fire=True)
            if res["run_ppe"]:
                ppe_count += 1
            if res["run_fall"]:
                fall_count += 1
            if res["run_fire"]:
                fire_count += 1

        self.assertEqual(ppe_count, 30, "PPE should run 30/60 frames (15 FPS equivalent at 30 source)")
        self.assertEqual(fall_count, 30, "Fall should run 30/60 frames (15 FPS equivalent at 30 source)")
        self.assertEqual(fire_count, 20, "Fire should run 20/60 frames (10 FPS equivalent at 30 source)")


class TestIncidentLifecyclePolicies(unittest.TestCase):
    def test_fall_policy_confirmation(self):
        mgr = IncidentLifecycleManager(policy=FALL_POLICY)
        t0 = 1000.0

        # Feed fall detection with confidence 0.50 (> trigger_threshold 0.45)
        # Needs duration >= 0.6s to confirm
        event1 = mgr.update([{"type": "fall", "confidence": 0.50, "score": 0.50}], timestamp=t0)
        self.assertIsNotNone(event1)
        self.assertEqual(event1.state, IncidentState.CANDIDATE)
        self.assertEqual(event1.event_type, "fall_detected")
        self.assertEqual(event1.details.get("cls_key"), "fall")

        # After 0.3s -> Still candidate
        event2 = mgr.update([{"type": "fall", "confidence": 0.52, "score": 0.52}], timestamp=t0 + 0.3)
        self.assertIsNotNone(event2)
        self.assertEqual(event2.state, IncidentState.CANDIDATE)

        # After 0.7s (exceeds confirm_duration_sec 0.6s) -> CONFIRMED!
        event3 = mgr.update([{"type": "fall", "confidence": 0.55, "score": 0.55}], timestamp=t0 + 0.7)
        self.assertIsNotNone(event3)
        self.assertEqual(event3.state, IncidentState.CONFIRMED)
        self.assertTrue(event3.is_new_alert)
        self.assertEqual(event3.event_type, "fall_detected")
        self.assertEqual(event3.details.get("cls_key"), "fall")

    def test_cls_key_resolution_for_non_fire_smoke(self):
        """
        Verify that cls_key correctly reflects the real event type,
        and never mistakenly defaults to 'smoke' for falls or PPE.
        """
        mgr = IncidentLifecycleManager()  # Default thresholds per-class
        t0 = 2000.0

        # 1. Fall detection -> must map to cls_key='fall'
        ev_fall = mgr.update([{"type": "fall", "confidence": 0.60, "score": 0.60}], timestamp=t0)
        self.assertIsNotNone(ev_fall)
        self.assertEqual(ev_fall.details.get("cls_key"), "fall")
        self.assertEqual(ev_fall.event_type, "fall_detected")

        # 2. PPE violation -> must map to cls_key='ppe'
        mgr_ppe = IncidentLifecycleManager()
        ev_ppe = mgr_ppe.update([{"type": "no_helmet", "confidence": 0.70, "score": 0.70}], timestamp=t0)
        self.assertIsNotNone(ev_ppe)
        self.assertEqual(ev_ppe.details.get("cls_key"), "ppe")
        self.assertEqual(ev_ppe.event_type, "no_helmet")

        # 3. Fire detection -> must map to cls_key='fire'
        mgr_fire = IncidentLifecycleManager()
        ev_fire = mgr_fire.update([{"type": "fire", "confidence": 0.75, "score": 0.75}], timestamp=t0)
        self.assertIsNotNone(ev_fire)
        self.assertEqual(ev_fire.details.get("cls_key"), "fire")
        self.assertEqual(ev_fire.event_type, "fire_detected")

        # 4. Smoke detection -> must map to cls_key='smoke'
        mgr_smoke = IncidentLifecycleManager()
        ev_smoke = mgr_smoke.update([{"type": "smoke", "confidence": 0.65, "score": 0.65}], timestamp=t0)
        self.assertIsNotNone(ev_smoke)
        self.assertEqual(ev_smoke.details.get("cls_key"), "smoke")
        self.assertEqual(ev_smoke.event_type, "smoke_detected")

    def test_default_thresholds_separation(self):
        """
        Verify default thresholds: fall confirm is 0.6s, smoke is 2.0s.
        """
        mgr = IncidentLifecycleManager()
        self.assertEqual(mgr.confirm_duration_sec["fall"], 0.6)
        self.assertEqual(mgr.confirm_duration_sec["smoke"], 2.0)
        self.assertEqual(mgr.confirm_duration_sec["fire"], 0.8)
        self.assertEqual(mgr.confirm_duration_sec["ppe"], 1.2)


class TestSupervisionPolygonZone(unittest.TestCase):
    def test_worker_feet_anchor_in_zone(self):
        # Danger zone polygon: box [100, 100] to [300, 300]
        polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]

        # Worker 1: [150, 150, 250, 280] -> bottom-center is (200, 280) -> INSIDE zone
        # Worker 2: [10, 10, 50, 80]     -> bottom-center is (30, 80)   -> OUTSIDE zone
        worker_boxes = [
            [150, 150, 250, 280],
            [10, 10, 50, 80],
        ]

        results = check_workers_in_polygon_zone(worker_boxes, polygon)
        self.assertEqual(results, [True, False])


class TestCameraManagerIntegration(unittest.TestCase):
    def setUp(self):
        # Mock detector and callbacks for CameraManager
        self.mock_detector = MagicMock()
        self.cam = CameraManager(
            source=0,
            detector=self.mock_detector,
            exclusion_zones=[
                {
                    "id": 1,
                    "name": "Khu vuc nguy hiem #1",
                    "is_active": True,
                    "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
                }
            ],
        )

    def test_check_danger_zones_integration(self):
        """
        Integration test: verify CameraManager._check_danger_zones
        calls check_workers_in_polygon_zone with self.exclusion_zones.
        """
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker_boxes = [
            [150, 150, 250, 280],  # Feet at (200, 280) -> Inside
            [10, 10, 50, 80],       # Feet at (30, 80)   -> Outside
        ]
        person_ids = [101, 102]

        violations = self.cam._check_danger_zones(
            worker_boxes=worker_boxes,
            annotated_frame=raw_frame,
            raw_frame=raw_frame,
            person_ids=person_ids,
            timestamp=100.0,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["type"], "danger_zone")
        self.assertEqual(violations[0]["person_id"], 101)
        self.assertEqual(violations[0]["zone_name"], "Khu vuc nguy hiem #1")

    def test_check_tick_danger_zone_lifecycle(self):
        """
        Verify that danger_zone violations accumulate in _check_tick and confirm.
        """
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cam.on_violation = MagicMock()
        self.cam.evidence_folder = "test_evidence"

        v = [{"type": "danger_zone", "person_id": 101, "confidence": 0.95}]

        # Frame 1 & 2: not yet confirmed (threshold is 3 frames)
        self.cam._check_tick(v, raw_frame, raw_frame, all_person_ids=[101])
        self.cam._check_tick(v, raw_frame, raw_frame, all_person_ids=[101])
        self.assertEqual(self.cam.on_violation.call_count, 0)

        # Frame 3: confirmed!
        self.cam._check_tick(v, raw_frame, raw_frame, all_person_ids=[101])
        self.assertIn("101_danger_zone", self.cam._alerted_persons)

    def test_camera_manager_check_fall_integration(self):
        """
        Integration test: verify CameraManager._check_fall uses FALL_POLICY
        and confirms with 'fall_detected' event_type.
        """
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cam.on_fall = MagicMock()
        self.cam.evidence_folder = "test_evidence"
        t0 = 500.0

        falls = [{"type": "fall", "confidence": 0.85, "score": 0.85, "bbox": [10, 20, 100, 200]}]

        # t=0: Candidate started
        self.cam._check_fall(falls, raw_frame, raw_frame, timestamp=t0)
        self.assertEqual(self.cam.fall_lifecycle.current_state, IncidentState.CANDIDATE)
        self.assertEqual(self.cam.on_fall.call_count, 0)

        # t=0.3: Still candidate (< 0.6s)
        self.cam._check_fall(falls, raw_frame, raw_frame, timestamp=t0 + 0.3)
        self.assertEqual(self.cam.fall_lifecycle.current_state, IncidentState.CANDIDATE)
        self.assertEqual(self.cam.on_fall.call_count, 0)

        # t=0.7: Exceeds 0.6s confirm duration -> CONFIRMED & on_fall triggered!
        self.cam._check_fall(falls, raw_frame, raw_frame, timestamp=t0 + 0.7)
        self.assertEqual(self.cam.fall_lifecycle.current_state, IncidentState.CONFIRMED)
        self.assertEqual(self.cam.fall_lifecycle.active_incident.event_type, "fall_detected")
        self.assertEqual(self.cam.fall_lifecycle.active_incident.details.get("cls_key"), "fall")


class TestSpatialContinuity(unittest.TestCase):
    def test_normalized_center_and_cell(self):
        from spatial_continuity import (
            compute_normalized_center,
            get_spatial_cell,
            compute_spatial_key,
        )

        # 1280x720 frame
        bbox = [128, 72, 256, 144]  # Center at (192, 108) -> normalized: 192/1280 = 0.15, 108/720 = 0.15
        cx, cy = compute_normalized_center(bbox, frame_size=(720, 1280))
        self.assertAlmostEqual(cx, 0.15, places=2)
        self.assertAlmostEqual(cy, 0.15, places=2)

        # In 10x6 grid, col = int(0.15 * 10) = 1, row = int(0.15 * 6) = 0
        col, row = get_spatial_cell(bbox, frame_size=(720, 1280), grid_cols=10, grid_rows=6)
        self.assertEqual((col, row), (1, 0))

        key = compute_spatial_key("no_helmet", bbox, zone_id="cam1", frame_size=(720, 1280))
        self.assertEqual(key, "cam1_no_helmet_c1_r0")

    def test_reopen_within_merge_distance(self):
        from spatial_continuity import SpatialContinuityManager

        mgr = SpatialContinuityManager(merge_distance=0.15, reopen_cooldown=3.0)
        # Register an incident resolved at t=100.0, center at (0.20, 0.20)
        mgr.register_resolved(
            incident_id="inc_001",
            event_type="fall_detected",
            bbox=[200, 100, 312, 188],  # Center ~ (256, 144) -> (0.20, 0.20) on 1280x720
            started_at=90.0,
            confirmed_at=91.0,
            resolved_at=100.0,
            confidence=0.85,
        )

        # Candidate 1: appears at t=101.5 (gap = 1.5s <= 3.0s), center at (0.22, 0.21) -> distance ~ 0.022 <= 0.15
        match = mgr.find_reopen_match(
            event_type="fall_detected",
            bbox=[220, 110, 340, 190],
            current_ts=101.5,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.incident_id, "inc_001")
        self.assertEqual(match.started_at, 90.0)

        # Candidate 2 at same time -> None because greedy assignment already popped inc_001 (1-to-many collision resolution)
        match2 = mgr.find_reopen_match(
            event_type="fall_detected",
            bbox=[220, 110, 340, 190],
            current_ts=101.5,
        )
        self.assertIsNone(match2)

    def test_cooldown_expiration(self):
        from spatial_continuity import SpatialContinuityManager

        mgr = SpatialContinuityManager(merge_distance=0.15, reopen_cooldown=3.0)
        mgr.register_resolved(
            incident_id="inc_002",
            event_type="fall_detected",
            bbox=[200, 100, 312, 188],
            started_at=90.0,
            confirmed_at=91.0,
            resolved_at=100.0,
            confidence=0.85,
        )

        # Candidate appears at t=104.5 (gap = 4.5s > 3.0s cooldown)
        match = mgr.find_reopen_match(
            event_type="fall_detected",
            bbox=[200, 100, 312, 188],
            current_ts=104.5,
        )
        self.assertIsNone(match, "Should not reopen after cooldown expired")

    def test_distance_exceeded(self):
        from spatial_continuity import SpatialContinuityManager

        mgr = SpatialContinuityManager(merge_distance=0.15, reopen_cooldown=3.0)
        mgr.register_resolved(
            incident_id="inc_003",
            event_type="fall_detected",
            bbox=[200, 100, 312, 188],  # Center ~ (0.20, 0.20)
            started_at=90.0,
            confirmed_at=91.0,
            resolved_at=100.0,
            confidence=0.85,
        )

        # Candidate appears on opposite side: (0.80, 0.80) -> dist ~ 0.85 > 0.15
        match = mgr.find_reopen_match(
            event_type="fall_detected",
            bbox=[1000, 500, 1100, 600],
            current_ts=101.0,
        )
        self.assertIsNone(match, "Should not reopen when distance exceeds merge_distance")


class TestIncidentLifecycleSpatialContinuityIntegration(unittest.TestCase):
    def test_lifecycle_reopen_active_and_fast_confirm(self):
        mgr = IncidentLifecycleManager(policy=FALL_POLICY, enable_spatial_continuity=True)
        bbox = [200, 100, 312, 188]
        t = 1000.0

        # Step 1: Candidate -> Confirmed
        mgr.update([{"type": "fall", "confidence": 0.85, "bbox": bbox}], timestamp=t)
        t += 0.7  # Exceeds confirm_duration 0.6s
        ev_confirmed = mgr.update([{"type": "fall", "confidence": 0.85, "bbox": bbox}], timestamp=t)
        self.assertEqual(ev_confirmed.state, IncidentState.CONFIRMED)
        orig_id = ev_confirmed.incident_id
        orig_start = ev_confirmed.started_at

        # Step 2: Object disappears -> Grace period (2.0s) -> Resolved (after 5.0s)
        t += 8.0
        ev_resolved = mgr.update([], timestamp=t)
        self.assertEqual(ev_resolved.state, IncidentState.RESOLVED)
        self.assertEqual(ev_resolved.incident_id, orig_id)

        # Step 3: Reappears within gap <= 1.0s (e.g. t + 0.8s) -> Immediate ACTIVE resume!
        t += 0.8
        ev_resumed = mgr.update([{"type": "fall", "confidence": 0.88, "bbox": bbox}], timestamp=t)
        self.assertEqual(ev_resumed.state, IncidentState.ACTIVE)
        self.assertEqual(ev_resumed.incident_id, orig_id)
        self.assertEqual(ev_resumed.started_at, orig_start)
        self.assertFalse(ev_resumed.is_new_alert)

        # Step 4: Disappears and resolves again
        t += 8.0
        mgr.update([], timestamp=t)
        self.assertEqual(mgr.current_state, IncidentState.RESOLVED)

        # Step 5: Reappears with 1.0s < gap <= 3.0s (e.g. gap = 2.0s) -> Fast-Confirm CANDIDATE!
        t += 2.0
        ev_reopened_cand = mgr.update([{"type": "fall", "confidence": 0.88, "bbox": bbox}], timestamp=t)
        self.assertEqual(ev_reopened_cand.state, IncidentState.CANDIDATE)
        self.assertEqual(ev_reopened_cand.incident_id, orig_id)

        # Fast confirm takes 50% time (0.3s instead of 0.6s)
        t += 0.35
        ev_reopened_conf = mgr.update([{"type": "fall", "confidence": 0.88, "bbox": bbox}], timestamp=t)
        self.assertEqual(ev_reopened_conf.state, IncidentState.CONFIRMED)
        self.assertEqual(ev_reopened_conf.incident_id, orig_id)
        self.assertFalse(ev_reopened_conf.is_new_alert)  # Reopened continuity, not a duplicate alert!


if __name__ == "__main__":
    unittest.main()
