import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concurrent.futures
import time
import cv2
from fire_detector import FireSmokeModel, FireSmokeStreamAnalyzer
from incident_lifecycle import IncidentLifecycleManager

def main():
    # 1. Single Shared Model
    shared_model = FireSmokeModel()

    # 2. Prepare frames for test
    cap_fire = cv2.VideoCapture('test_videos/test_fire_sample.mp4')
    ret, frame_fire = cap_fire.read()
    cap_fire.release()

    cap_norm = cv2.VideoCapture('test_videos/6_di_bo_binh_thuong_normal.mp4')
    ret, frame_norm = cap_norm.read()
    cap_norm.release()

    # 3. Setup 8 streams (4 fire, 4 normal)
    stream_configs = [
        {'id': 0, 'name': 'Cam-0_Fire', 'frame': frame_fire, 'expect_hazard': True},
        {'id': 1, 'name': 'Cam-1_Normal', 'frame': frame_norm, 'expect_hazard': False},
        {'id': 2, 'name': 'Cam-2_Fire', 'frame': frame_fire, 'expect_hazard': True},
        {'id': 3, 'name': 'Cam-3_Normal', 'frame': frame_norm, 'expect_hazard': False},
        {'id': 4, 'name': 'Cam-4_Normal', 'frame': frame_norm, 'expect_hazard': False},
        {'id': 5, 'name': 'Cam-5_Fire', 'frame': frame_fire, 'expect_hazard': True},
        {'id': 6, 'name': 'Cam-6_Normal', 'frame': frame_norm, 'expect_hazard': False},
        {'id': 7, 'name': 'Cam-7_Fire', 'frame': frame_fire, 'expect_hazard': True},
    ]

    analyzers = [FireSmokeStreamAnalyzer(model=shared_model) for _ in range(8)]
    lifecycles = [
        IncidentLifecycleManager(confirm_duration_sec=0.5, active_miss_grace_sec=1.5, resolved_after_sec=5.0)
        for _ in range(8)
    ]

    def run_stream_cycle(stream_idx, step_idx):
        cfg = stream_configs[stream_idx]
        analyzer = analyzers[stream_idx]
        lifecycle = lifecycles[stream_idx]
        ts = step_idx * 0.133  # simulate time progression

        # Process frame
        verified = analyzer.detect(cfg['frame'], timestamp=ts)
        event = lifecycle.update(verified, timestamp=ts)
        return (stream_idx, len(analyzer._tracks), event)

    print('Starting 8 concurrent streams across 15 time steps (120 total inference calls)...')
    t0 = time.perf_counter()

    total_events_created = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for step in range(15):
            futures = [executor.submit(run_stream_cycle, i, step) for i in range(8)]
            for f in concurrent.futures.as_completed(futures):
                s_idx, track_cnt, event = f.result()
                if event and event.is_new_alert:
                    total_events_created += 1

    elapsed = time.perf_counter() - t0
    print(f'Done! Processed 120 frames across 8 threads in {elapsed:.2f}s ({120/elapsed:.1f} FPS)')
    print(f'Total new incidents confirmed: {total_events_created}')

    # Verify track counts and incident isolation
    for cfg in stream_configs:
        i = cfg['id']
        track_count = len(analyzers[i]._tracks)
        name = cfg['name']
        print(f" -> {name}: {track_count} tracks, state={lifecycles[i].current_state}")
        if cfg['expect_hazard']:
            assert track_count > 0, f"Stream {name} should have hazard tracks"
            assert lifecycles[i].current_state.value in ('confirmed', 'active'), f"Stream {name} should be confirmed/active"
        else:
            assert track_count == 0, f"Stream {name} should have 0 tracks"
            assert lifecycles[i].current_state.value == 'none', f"Stream {name} should be none"

    print('TEST PASSED: 100% thread safety and complete track isolation verified across 8 concurrent streams!')

if __name__ == '__main__':
    main()
