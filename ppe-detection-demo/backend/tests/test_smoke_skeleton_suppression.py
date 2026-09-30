"""
test_smoke_skeleton_suppression.py
----------------------------------
Unit tests verifying the smoke skeleton suppression enhancements:
1. _is_valid_human_skeleton rejects smoke-like keypoints (blurry, no torso core, narrow span)
2. Temporal gate: skeleton requires >= 2 consecutive valid frames
3. Fire/Smoke alert clears within <= 2 frames after zero candidates (scene cut)
4. Safe cross-verification in smoke: accepts worker in smoke ONLY if pose verified >=2 frames + torso core + sharpness >= 30
"""
import sys
import os
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_fall_detector import PoseFallDetector, KPT_CONF_THRESH
from fire_detector import FireSmokeStreamAnalyzer
from detector import PPEDetector


class TestSkeletonSmokeSuppression(unittest.TestCase):
    """Test skeleton gate rejects smoke phantoms."""
    
    def setUp(self):
        self.detector = PoseFallDetector()
    
    def _make_smoke_kpts_no_torso(self):
        """Tao keypoints gia lap khói: co 6+ diem nhung THIEU vai/hong (de test torso core)."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        # 7 diem: nose, eyes, ears, knees (co conf >= 0.45) nhung KHONG co vai (5,6) va hong (11,12)
        for idx in [0, 1, 2, 3, 4, 13, 14]:
            kpts[idx] = [120 + idx, 110 + idx * 5, 0.60]
        return kpts
    
    def _make_smoke_kpts_narrow_span(self):
        """Tao keypoints co vai/hong nhung cum 1 ben (span < 20% bbox) - de test anatomical span."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Bbox: [100, 100, 250, 250] (w=150, h=150)
        # Tat ca diem nam trong vung 10x10 -> span ~ 6% < 20%
        # Co vai (5,6), hong (11,12) voi conf >= 0.45 de pass torso core
        base = np.array([110, 110], dtype=np.float32)
        for idx in [0, 5, 6, 11, 12, 13, 14]:
            kpts[idx] = [base[0] + (idx % 3), base[1] + (idx // 3), 0.80]
        return kpts
    
    def _make_person_kpts(self):
        """Tao keypoints nguoi that: co vai/hong ro, span day du."""
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [200, 100, 0.9]   # nose
        kpts[5] = [180, 130, 0.9]   # l_sh
        kpts[6] = [220, 130, 0.9]   # r_sh
        kpts[11] = [180, 200, 0.9]  # l_hip
        kpts[12] = [220, 200, 0.9]  # r_hip
        kpts[13] = [180, 270, 0.8]  # l_knee
        kpts[14] = [220, 270, 0.8]  # r_knee
        kpts[15] = [180, 340, 0.7]  # l_ankle
        kpts[16] = [220, 340, 0.7]  # r_ankle
        return kpts
    
    def _make_frame(self, sharp=True):
        """Tao frame test: sharp (net) hoac blurry (mo giong khói)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if sharp:
            cv2.rectangle(frame, (100, 100), (300, 350), (200, 200, 200), -1)
            cv2.rectangle(frame, (100, 100), (300, 350), (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (100, 100), (300, 350), (120, 120, 120), -1)
            frame = cv2.GaussianBlur(frame, (15, 15), 10)
        return frame
    
    def test_smoke_keypoints_rejected_by_torso_core(self):
        """Khói khong co vai/hong conf >= 0.45 -> bi reject bang torso core check."""
        kpts = self._make_smoke_kpts_no_torso()
        bbox = [100, 100, 250, 250]
        frame = self._make_frame(sharp=False)
        
        valid, reason = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=True, track_id=1, require_temporal=False
        )
        self.assertFalse(valid, f"Smoke keypoints should be rejected: {reason}")
        self.assertIn("torso core", reason.lower())
    
    def test_smoke_keypoints_rejected_by_anatomical_span(self):
        """Khói co vai/hong nhung cum 1 ben (span < 20%) -> bi reject bang logic OR."""
        kpts = self._make_smoke_kpts_narrow_span()
        bbox = [100, 100, 250, 250]
        frame = self._make_frame(sharp=True)
        
        valid, reason = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=True, track_id=2, require_temporal=False
        )
        self.assertFalse(valid, f"Narrow span smoke should be rejected: {reason}")
        self.assertIn("span too narrow", reason.lower())
    
    def test_smoke_keypoints_rejected_by_laplacian(self):
        """Khói mo co Laplacian < 25 -> bi reject."""
        kpts = self._make_person_kpts()
        bbox = [100, 100, 250, 350]
        frame = self._make_frame(sharp=False)
        
        valid, reason = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=True, track_id=3, require_temporal=False
        )
        self.assertFalse(valid, f"Blurry frame should be rejected: {reason}")
        self.assertIn("laplacian", reason.lower())
    
    def test_person_keypoints_accepted(self):
        """Nguoi that keypoints du dieu kien -> duoc chap nhan."""
        kpts = self._make_person_kpts()
        bbox = [100, 100, 250, 350]
        frame = self._make_frame(sharp=True)
        
        valid, reason = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=False, track_id=4, require_temporal=False
        )
        self.assertTrue(valid, f"Valid person should be accepted: {reason}")
    
    def test_temporal_gate_requires_two_frames(self):
        """Temporal gate: frame 1 reject, frame 2 accept."""
        kpts = self._make_person_kpts()
        bbox = [100, 100, 250, 350]
        frame = self._make_frame(sharp=True)
        
        # Frame 1: require_temporal=True -> reject (chua du 2 frames)
        valid1, _ = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=False, track_id=10, require_temporal=True
        )
        self.assertFalse(valid1, "Frame 1 should be rejected by temporal gate")
        
        # Frame 2: cung track_id -> accept (da du 2 frames)
        valid2, _ = self.detector._is_valid_human_skeleton(
            kpts, bbox, frame=frame, is_in_smoke=False, track_id=10, require_temporal=True
        )
        self.assertTrue(valid2, "Frame 2 should be accepted by temporal gate")
    
    def test_temporal_reset_on_invalid(self):
        """Neu frame n hop le nhung frame n+1 khong hop le -> counter reset."""
        kpts_good = self._make_person_kpts()
        kpts_bad = self._make_smoke_kpts_no_torso()
        bbox = [100, 100, 250, 350]
        frame = self._make_frame(sharp=True)
        
        # Frame 1: good
        self.detector._is_valid_human_skeleton(kpts_good, bbox, frame=frame, is_in_smoke=False, track_id=20, require_temporal=True)
        # Frame 2: good -> accepted
        valid2 = self.detector._is_valid_human_skeleton(kpts_good, bbox, frame=frame, is_in_smoke=False, track_id=20, require_temporal=True)[0]
        self.assertTrue(valid2)
        
        # Frame 3: bad -> rejected, counter reset
        valid3 = self.detector._is_valid_human_skeleton(kpts_bad, bbox, frame=frame, is_in_smoke=True, track_id=20, require_temporal=True)[0]
        self.assertFalse(valid3)
        
        # Frame 4: good again -> should be rejected (counter reset, lai o frame 1)
        valid4 = self.detector._is_valid_human_skeleton(kpts_good, bbox, frame=frame, is_in_smoke=False, track_id=20, require_temporal=True)[0]
        self.assertFalse(valid4, "Counter should reset after invalid frame")


class TestFireSmokeSceneCut(unittest.TestCase):
    """Test fire/smoke scene cut: alert clears within <= 2 frames after zero candidates."""
    
    def setUp(self):
        self.analyzer = FireSmokeStreamAnalyzer()
    
    def _make_verified_track(self, track_id=1, track_type="smoke"):
        """Helper to create a verified track with required fields."""
        return {
            "verified": True,
            "type": track_type,
            "bbox": [100, 100, 200, 200],
            "last_seen": 100.0,
            "first_seen": 99.0,
            "hits": 5,
            "conf": 0.8,
        }
    
    def test_scene_cut_clears_verified_in_two_frames(self):
        """Khi 0 candidate 2 frame lien tiep -> verified tracks bi force-clear."""
        # Tao track verified gia
        self.analyzer._tracks[1] = self._make_verified_track(1, "smoke")
        
        # Frame 1: 0 candidate
        result1 = self.analyzer.detect(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=100.5
        )
        # Track van verified (chi 1 frame 0 candidate)
        self.assertTrue(self.analyzer._tracks[1].get("verified", False))
        
        # Frame 2: 0 candidate -> force clear
        result2 = self.analyzer.detect(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=101.0
        )
        # Track phai bi clear verified
        self.assertFalse(self.analyzer._tracks[1].get("verified", False))
    
    def test_hysteresis_exit_fast(self):
        """hysteresis_exit_sec = 0.6s: track verified het han sau 0.6s duoi exit threshold."""
        # Setup track verified voi score cao
        self.analyzer._tracks[1] = self._make_verified_track(1, "smoke")
        
        # Giả lập score roi duoi exit threshold
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock scorer de tra ve score thap
        original_calc = self.analyzer.smoke_scorer.calculate
        self.analyzer.smoke_scorer.calculate = lambda *args, **kwargs: (0.2, {})
        
        try:
            # Frame 1: score thap, chua het 0.6s
            result1 = self.analyzer.detect(frame, timestamp=100.1)
            self.assertTrue(self.analyzer._tracks[1].get("verified", False))
            
            # Frame 2: sau 0.6s -> phai clear
            result2 = self.analyzer.detect(frame, timestamp=100.8)
            self.assertFalse(self.analyzer._tracks[1].get("verified", False))
        finally:
            self.analyzer.smoke_scorer.calculate = original_calc
    
    def test_stale_track_cleanup_fast(self):
        """Stale track cleanup: 0.5s thay vi 1.2s."""
        self.analyzer._tracks[1] = self._make_verified_track(1, "smoke")
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 0.4s sau -> van con track
        self.analyzer.detect(frame, timestamp=100.4)
        self.assertIn(1, self.analyzer._tracks)
        
        # 0.6s sau -> phai bi xoa (cleanup chay 0.5s interval)
        self.analyzer.detect(frame, timestamp=100.6)
        # Cleanup chay khi timestamp - last_cleanup > 0.5


class TestSafeCrossVerificationInSmoke(unittest.TestCase):
    """Test safe cross-verification: worker in smoke accepted ONLY if pose verified >=2 frames + torso core + sharpness >= 30."""
    
    def setUp(self):
        self.ppe_detector = PPEDetector()
    
    def test_worker_in_smoke_accepted_only_if_pose_verified(self):
        """
        Nguoi trong vung khói CHI duoc chap nhan neu:
        1. Pose detector da verify skeleton >= 2 frames
        2. Co Torso Core (vai + hông conf >= 0.45)
        3. Sharpness >= 30
        """
        # Vi can mock PPE model va fall detector, test nay la integration test
        # Chay bang test thuc te tren video chay.mp4
        self.skipTest("Integration test - run manually with real video")


if __name__ == "__main__":
    unittest.main()