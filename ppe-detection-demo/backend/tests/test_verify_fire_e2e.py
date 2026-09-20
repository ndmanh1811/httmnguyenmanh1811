import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import PPEDetector
from fire_detector import FireSmokeModel, FireSmokeStreamAnalyzer


def main():
    shared_model = FireSmokeModel()
    analyzer = FireSmokeStreamAnalyzer(model=shared_model)
    detector = PPEDetector()

    print("=== TEST 1: POSITIVE CAR FIRE CCTV VIDEO ===")
    fire_vid = "static/uploads/1789781394_test_fire_sample.mp4"
    out_fire = "static/outputs/test_verify_fire.mp4"
    t0 = time.time()
    stats_fire = detector.process_video(
        fire_vid,
        out_fire,
        frame_skip=2,
        fall_detector=None,
        fire_detector=analyzer,
        enable_ppe=False,
    )
    print(f"Stats: processed={stats_fire.get('processed_frames')}, fire_count={stats_fire.get('fire_count')}, smoke_count={stats_fire.get('smoke_count')} in {time.time()-t0:.1f}s")
    assert (stats_fire.get("fire_count", 0) + stats_fire.get("smoke_count", 0)) > 0
    print(">>> Test 1 Passed: Fire & Smoke reliably detected in car fire video!")

    print("\n=== TEST 2: POSITIVE WAREHOUSE FIRE VIDEO (chay.mp4) ===")
    analyzer.reset()
    chay_vid = "static/uploads/1789813910_chay.mp4"
    out_chay = "static/outputs/test_verify_chay.mp4"
    t0 = time.time()
    stats_chay = detector.process_video(
        chay_vid,
        out_chay,
        frame_skip=3,
        fall_detector=None,
        fire_detector=analyzer,
        enable_ppe=False,
    )
    print(f"Stats: processed={stats_chay.get('processed_frames')}, fire_count={stats_chay.get('fire_count')}, smoke_count={stats_chay.get('smoke_count')} in {time.time()-t0:.1f}s")
    assert (stats_chay.get("fire_count", 0) + stats_chay.get("smoke_count", 0)) > 0
    print(">>> Test 2 Passed: Smoke plume reliably detected in warehouse fire video!")

    print("\n=== TEST 3: NEGATIVE INDOOR NORMAL WALKING VIDEO ===")
    analyzer.reset()
    norm_vid = "static/uploads/1789693253_normal_walking.mp4"
    out_norm = "static/outputs/test_verify_normal.mp4"
    t0 = time.time()
    stats_norm = detector.process_video(
        norm_vid,
        out_norm,
        frame_skip=2,
        fall_detector=None,
        fire_detector=analyzer,
        enable_ppe=False,
    )
    print(f"Stats: processed={stats_norm.get('processed_frames')}, fire_count={stats_norm.get('fire_count')}, smoke_count={stats_norm.get('smoke_count')} in {time.time()-t0:.1f}s")
    assert stats_norm.get("fire_count", 0) == 0, f"False fire: {stats_norm.get('fire_count')}"
    assert stats_norm.get("smoke_count", 0) == 0, f"False smoke: {stats_norm.get('smoke_count')}"
    print(">>> Test 3 Passed: ZERO false alarms on negative indoor walking video!")

    print("\n=== TEST 4: NEGATIVE OUTDOOR SUNNY VEHICLE/GRASS VIDEO ===")
    analyzer.reset()
    trim_vid = "static/uploads/1789753125_videoplayback_-_Trim.mp4"
    out_trim = "static/outputs/test_verify_trim.mp4"
    t0 = time.time()
    stats_trim = detector.process_video(
        trim_vid,
        out_trim,
        frame_skip=3,
        fall_detector=None,
        fire_detector=analyzer,
        enable_ppe=False,
    )
    print(f"Stats: processed={stats_trim.get('processed_frames')}, fire_count={stats_trim.get('fire_count')}, smoke_count={stats_trim.get('smoke_count')} in {time.time()-t0:.1f}s")
    assert stats_trim.get("fire_count", 0) == 0, f"False fire: {stats_trim.get('fire_count')}"
    assert stats_trim.get("smoke_count", 0) == 0, f"False smoke: {stats_trim.get('smoke_count')}"
    print(">>> Test 4 Passed: ZERO false alarms on negative outdoor sunny video!")

    print("\n========================================================")
    print(">>> ALL 4 REAL-WORLD VIDEO VERIFICATIONS PASSED! <<<")
    print("========================================================")


if __name__ == "__main__":
    main()
