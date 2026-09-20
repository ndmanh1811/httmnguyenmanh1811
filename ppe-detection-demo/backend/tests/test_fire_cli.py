import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
test_fire_cli.py - Command-line test harness for Fire & Smoke Decoupled Verification
Runs video through FireSmokeStreamAnalyzer & IncidentLifecycleManager,
prints per-frame scoring metrics and state transitions, and writes annotated video.
"""

import argparse
import os
import time
import cv2
import numpy as np

from fire_detector import FireSmokeModel, FireSmokeStreamAnalyzer, check_exclusion_breakout
from incident_lifecycle import IncidentLifecycleManager, IncidentState


def run_cli_test(
    video_path: str,
    output_path: str,
    max_frames: int = 350,
    frame_skip: int = 3,
):
    if not os.path.exists(video_path):
        print(f"Error: Video path {video_path} not found.")
        return

    print("=== Starting Fire & Smoke CLI Verification ===")
    print(f"Input Video: {video_path}")
    print(f"Output Video: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))

    # Initialize shared model, stream analyzer with decoupled scorers, and lifecycle manager
    model = FireSmokeModel()
    analyzer = FireSmokeStreamAnalyzer(
        model=model,
        persistence_sec={"fire": 0.8, "smoke": 2.0},
    )
    lifecycle = IncidentLifecycleManager(
        confirm_duration_sec={"fire": 0.8, "smoke": 2.0},
        trigger_thresholds={"fire": 0.65, "smoke": 0.65},
        hold_thresholds={"fire": 0.45, "smoke": 0.40},
    )

    frame_idx = 0
    processed_count = 0
    prev_state = IncidentState.NONE
    start_wall_time = time.time()

    print("-" * 95)
    print(f"{'Frame':<8} | {'Sim Time':<9} | {'Type':<6} | {'Conf':<5} | {'Score':<6} | {'AreaTrend':<10} | {'State':<12} | {'Notes'}")
    print("-" * 95)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_idx >= max_frames):
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            processed_count += 1
            sim_time = frame_idx / fps

            # 1. Detect & evaluate decoupled scores
            detections = analyzer.detect(frame, timestamp=sim_time)

            # 2. Update Lifecycle with Hysteresis
            event = lifecycle.update(detections, timestamp=sim_time)
            curr_state = lifecycle.current_state

            # 3. Log transitions
            if curr_state != prev_state:
                print(f">>> [STATE TRANSITION] {prev_state.value.upper()} -> {curr_state.value.upper()} at {sim_time:.2f}s")
                prev_state = curr_state

            # 4. Print metrics for primary detection
            if detections:
                prim = max(detections, key=lambda x: x.get("score", 0.0))
                p_type = prim.get("type")
                p_conf = prim.get("confidence", 0.0)
                p_score = prim.get("score", 0.0)
                comp = prim.get("components", {})
                area_trend_str = f"{comp.get('area_trend', 0.5):.2f}" if p_type == "smoke" else "N/A"
                notes = f"Dur:{comp.get('duration', 0.0):.1f}s"
                if event and event.is_new_alert:
                    notes += " [NEW ALERT!]"
                print(f"{frame_idx:<8} | {sim_time:<8.2f}s | {p_type:<6} | {p_conf:<5.2f} | {p_score:<6.2f} | {area_trend_str:<10} | {curr_state.value:<12} | {notes}")

            # 5. Annotate frame
            annotated = analyzer.annotate_frame(frame.copy(), detections)

            # Draw HUD
            hud_text = f"State: {curr_state.value.upper()} | Time: {sim_time:.1f}s"
            cv2.putText(annotated, hud_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            out.write(annotated)

    finally:
        cap.release()
        out.release()

    total_time = time.time() - start_wall_time
    print("-" * 95)
    print(f"Verification Complete! Processed {processed_count} frames in {total_time:.2f}s ({(processed_count / total_time):.1f} FPS)")
    print(f"Final Output saved to: {output_path}")


if __name__ == "__main__":
    video_in = os.path.join("static", "uploads", "1789781745_chay.mp4")
    video_out = os.path.join("static", "outputs", "cli_verification_result.mp4")
    run_cli_test(video_in, video_out, max_frames=300, frame_skip=3)
