import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
test_exclusion_pipeline.py - Verification script for Exclusion Zones & Breakout Guard on video
"""

import os
import cv2
import numpy as np

from fire_detector import FireSmokeModel, FireSmokeStreamAnalyzer
from incident_lifecycle import IncidentLifecycleManager


def test_exclusion_on_video():
    video_path = os.path.join("static", "uploads", "1789781745_chay.mp4")
    output_path = os.path.join("static", "outputs", "exclusion_verification_result.mp4")

    if not os.path.exists(video_path):
        print(f"File {video_path} not found.")
        return

    print("=== Testing Video Exclusion Zone & Breakout Pipeline ===")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))

    # Define simulated exclusion zone (e.g. welding station or stove area covering center of frame)
    exclusion_zones = [
        {
            "id": 1,
            "name": "Khu vuc han xi #1",
            "zone_type": "welding",
            "polygon_points": [
                [0.15, 0.10],
                [0.85, 0.10],
                [0.85, 0.65],
                [0.15, 0.65],
            ],
            "is_active": True,
        }
    ]

    model = FireSmokeModel()
    analyzer = FireSmokeStreamAnalyzer(
        model=model,
        persistence_sec={"fire": 0.8, "smoke": 2.0},
        exclusion_zones=exclusion_zones,
    )
    lifecycle = IncidentLifecycleManager(
        confirm_duration_sec={"fire": 0.8, "smoke": 2.0},
    )

    frame_idx = 0
    ignored_count = 0
    breakout_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > 180:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        sim_ts = frame_idx / fps
        hazards = analyzer.detect(frame, timestamp=sim_ts)
        event = lifecycle.update(hazards, timestamp=sim_ts)

        for h in hazards:
            if h.get("is_breakout"):
                breakout_count += 1

        annotated = analyzer.annotate_frame(frame, hazards)
        out.write(annotated)

    cap.release()
    out.release()

    print(f"Processed {frame_idx} frames.")
    print(f"Output saved to: {output_path}")
    print(">>> Video Exclusion Zone pipeline completed successfully!")


if __name__ == "__main__":
    test_exclusion_on_video()
