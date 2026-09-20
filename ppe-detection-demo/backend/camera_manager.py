"""
camera_manager.py
-----------------
Tick-based alert system:
- Moi frame: chay YOLO de hien thi video muot
- Moi 10 giây (tick): kiem tra vi pham -> dem tick -> quyet dinh alert
- 3 tick lien tuc vi pham -> ALERT
- Co PPE giua chuong -> reset ve 0
"""

import logging
import os
import queue
import threading
import time
import uuid
from collections import deque

import cv2

from incident_lifecycle import IncidentLifecycleManager

logger = logging.getLogger(__name__)

TICK_INTERVAL = 10
CONFIRM_TICKS = 3


class CameraManager:
    def __init__(
        self,
        source,
        detector,
        fall_detector=None,
        fire_model=None,
        fire_detector=None,
        frame_skip=4,
        fire_interval_frames=4,
        pre_event_seconds=3.0,
        camera_id=0,
        evidence_folder=None,
        on_violation=None,
        on_fall=None,
        on_fire=None,
        tick_interval=TICK_INTERVAL,
        confirm_ticks=CONFIRM_TICKS,
        enable_ppe=True,
        enable_fall=True,
        enable_fire=True,
        exclusion_zones=None,
    ):
        self.source = source
        self.detector = detector
        self.fall_detector = fall_detector
        self.frame_skip = frame_skip
        self.fire_interval_frames = fire_interval_frames
        self.pre_event_seconds = pre_event_seconds
        self.camera_id = camera_id
        self.evidence_folder = evidence_folder
        self.on_violation = on_violation
        self.on_fall = on_fall
        self.on_fire = on_fire
        self.tick_interval = tick_interval
        self.confirm_ticks = confirm_ticks
        self.enable_ppe = bool(enable_ppe)
        self.enable_fall = bool(enable_fall)
        self.enable_fire = bool(enable_fire)
        self.exclusion_zones = exclusion_zones or []

        # Per-stream Fire Analyzer (Isolated stateful temporal association)
        from fire_detector import FireSmokeModel, FireSmokeStreamAnalyzer
        shared_model = fire_model
        if shared_model is None and fire_detector is not None:
            if hasattr(fire_detector, "model"):
                shared_model = fire_detector.model
            elif isinstance(fire_detector, FireSmokeModel):
                shared_model = fire_detector

        self.fire_analyzer = (
            FireSmokeStreamAnalyzer(model=shared_model, exclusion_zones=self.exclusion_zones)
            if shared_model is not None
            else None
        )

        # Memory-efficient ring buffer: (timestamp, jpeg_bytes)
        buffer_len = max(30, int(30 * pre_event_seconds))
        self._pre_event_buffer = deque(maxlen=buffer_len)

        # Per-stream Incident Lifecycle (Class-specific duration & hysteresis)
        self.fire_lifecycle = IncidentLifecycleManager(
            confirm_duration_sec={"fire": 0.8, "smoke": 2.0},
            trigger_thresholds={"fire": 0.50, "smoke": 0.45},
            hold_thresholds={"fire": 0.35, "smoke": 0.30},
            active_miss_grace_sec=1.5,
            resolved_after_sec=5.0,
        )

        self._running = threading.Event()
        self._frame_queue = queue.Queue(maxsize=1)
        self._thread = None
        self._cap = None

        self._tick_counts: dict[str, int] = {}
        self._last_tick_time: dict[str, float] = {}
        self._empty_counts: dict[str, int] = {}
        self._last_fall_alert_ts: float = 0
        self._fall_in_progress: bool = False
        self._fall_miss_counter: int = 0
        self._person_violation_frames: dict[str, int] = {}
        self._alerted_persons: set[str] = set()
        self._person_last_seen: dict[int, float] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("CameraManager started: source=%s tick=%ds confirm=%d",
                     self.source, self.tick_interval, self.confirm_ticks)

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        logger.info("CameraManager stopped: %s", self.source)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def update_toggles(self, enable_ppe: bool = None, enable_fall: bool = None, enable_fire: bool = None):
        """Cap nhat toggle tinh nang AI ma khong can khoi dong lai camera."""
        if enable_ppe is not None:
            self.enable_ppe = bool(enable_ppe)
        if enable_fall is not None:
            self.enable_fall = bool(enable_fall)
        if enable_fire is not None:
            self.enable_fire = bool(enable_fire)
        logger.info("CameraManager %s toggles updated: ppe=%s fall=%s fire=%s",
                    self.source, self.enable_ppe, self.enable_fall, self.enable_fire)

    def update_exclusion_zones(self, zones: list[dict]):
        """Cap nhat danh sach vung loai tru realtime ma khong ngat luong."""
        self.exclusion_zones = zones or []
        if self.fire_analyzer:
            self.fire_analyzer.set_exclusion_zones(self.exclusion_zones)
        logger.info("CameraManager %s exclusion zones updated: %d zones", self.source, len(self.exclusion_zones))

    def get_frame(self):
        try:
            return self._frame_queue.get(timeout=30)
        except queue.Empty:
            return None

    def _run(self):
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error("Cannot open camera: %s", source)
            return

        frame_idx = 0
        last_annotated = None

        try:
            while self._running.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame_idx += 1
                now = time.time()

                # Compress and store in pre-event ring buffer (memory-efficient ~60KB vs ~6MB)
                ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._pre_event_buffer.append((now, jpeg_buf.tobytes()))

                run_inference = (self.frame_skip <= 1) or (frame_idx % self.frame_skip == 0)

                if run_inference:
                    active_fall_det = self.fall_detector if self.enable_fall else None

                    # Fire inference scheduling (interleaved based on fire_interval_frames)
                    run_fire = self.enable_fire and self.fire_analyzer is not None and (
                        (self.fire_interval_frames <= 1) or (frame_idx % self.fire_interval_frames == 0)
                    )
                    active_fire_det = self.fire_analyzer if run_fire else None

                    annotated, has_violation, ppe_stats, violations, falls, fires, all_person_ids = self.detector.annotate_frame(
                        frame,
                        fall_detector=active_fall_det,
                        fire_detector=active_fire_det,
                        timestamp=now,
                        enable_ppe=self.enable_ppe,
                    )
                    last_annotated = annotated

                    if self.enable_ppe:
                        self._check_tick(violations, annotated, frame, all_person_ids)
                    if self.enable_fall:
                        self._check_fall(falls, annotated, frame)
                    if run_fire:
                        self._check_fire(fires, annotated, frame, now)
                else:
                    annotated = last_annotated if last_annotated is not None else frame

                if not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(annotated)

        except Exception as e:
            logger.error("CameraManager error: %s", e)
        finally:
            if self._cap and self._cap.isOpened():
                self._cap.release()
                self._cap = None
            logger.info("Camera thread exiting, camera released")

    def _check_tick(self, violations, annotated_frame, raw_frame, all_person_ids=None):
        now = time.time()

        # Thu thập các loại vi phạm hiện tại theo từng person
        current_violations: dict[int, dict[str, float]] = {}  # pid -> {vtype: confidence}
        for v in violations:
            vtype = v.get("type")
            pid = v.get("person_id", 0)
            conf = v.get("confidence", 0.85)
            self._person_last_seen[pid] = now

            if pid not in current_violations:
                current_violations[pid] = {}
            current_violations[pid][vtype] = conf

            v_key = f"{pid}_{vtype}"
            self._person_violation_frames[v_key] = self._person_violation_frames.get(v_key, 0) + 1

            # Ngưỡng xác nhận: >= 5 frames liên tiếp (~0.7 - 0.8 giây)
            if self._person_violation_frames[v_key] >= 5:
                if v_key not in self._alerted_persons:
                    logger.info("VIOLATION CONFIRMED: Person #%s -> %s (conf=%.2f)", pid, vtype, conf)
                    self._alerted_persons.add(v_key)
                    self._save_evidence(vtype, annotated_frame, raw_frame, count=1, person_id=pid, confidence=conf)

        # Reset bộ đếm vi phạm khi person TUÂN THỦ (đã mặc đồ bảo hộ lại)
        # Dùng ALL person IDs (kể cả người tuân thủ 100%), không chỉ người vi phạm
        visible_pids = set(all_person_ids or [])
        for pid in visible_pids:
            self._person_last_seen[pid] = now  # Cập nhật last seen cho MỌI người
            pid_violations = current_violations.get(pid, {})
            for vtype in ["no_helmet", "no_vest", "no_mask"]:
                v_key = f"{pid}_{vtype}"
                if vtype not in pid_violations:
                    # Person này KHÔNG vi phạm loại này → giảm bộ đếm
                    if v_key in self._person_violation_frames:
                        self._person_violation_frames[v_key] = max(
                            self._person_violation_frames[v_key] - 2, 0
                        )
                        # Nếu bộ đếm về 0 → cho phép alert lại nếu vi phạm tái phát
                        if self._person_violation_frames[v_key] == 0:
                            self._alerted_persons.discard(v_key)
                            self._person_violation_frames.pop(v_key, None)

        # Dọn dẹp bộ nhớ của person đã rời khỏi góc camera quá 20 giây
        for pid in list(self._person_last_seen.keys()):
            if now - self._person_last_seen[pid] > 20:
                self._person_last_seen.pop(pid, None)
                for k in list(self._alerted_persons):
                    if k.startswith(f"{pid}_"):
                        self._alerted_persons.discard(k)
                for k in list(self._person_violation_frames.keys()):
                    if k.startswith(f"{pid}_"):
                        self._person_violation_frames.pop(k, None)

    def _save_evidence(self, vtype, annotated_frame, raw_frame, count=1, person_id=None, confidence=0.85):
        if not self.on_violation or not self.evidence_folder:
            return

        try:
            fname = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(self.evidence_folder, fname)
            cv2.imwrite(image_path, annotated_frame)

            self.on_violation(
                alert_type=vtype,
                confidence=round(confidence, 3),
                image_path=image_path,
                camera_id=self.camera_id,
                count=count,
                person_id=person_id,
            )
        except Exception as e:
            logger.error("Failed to save evidence: %s", e)

    def _check_fall(self, falls, annotated_frame, raw_frame):
        now = time.time()
        if not falls:
            if self._fall_in_progress:
                self._fall_miss_counter += 1
                if self._fall_miss_counter >= 8:
                    logger.info("Fall incident resolved/cleared for camera %s", self.camera_id)
                    self._fall_in_progress = False
                    self._fall_miss_counter = 0
            return

        self._fall_miss_counter = 0

        if self._fall_in_progress:
            return

        if now - self._last_fall_alert_ts < 10:
            return

        self._last_fall_alert_ts = now
        self._fall_in_progress = True

        fall = falls[0]
        logger.info("FALL DETECTED (New Incident): conf=%.2f label=%s", fall["confidence"], fall["label"])

        if self.on_fall and self.evidence_folder:
            try:
                fname = f"fall_{uuid.uuid4().hex}.jpg"
                image_path = os.path.join(self.evidence_folder, fname)
                cv2.imwrite(image_path, annotated_frame)
                self.on_fall(
                    confidence=fall["confidence"],
                    image_path=image_path,
                    camera_id=self.camera_id,
                )
            except Exception as e:
                logger.error("Failed to save fall evidence: %s", e)

    def _check_fire(self, fires, annotated_frame, raw_frame, timestamp):
        event = self.fire_lifecycle.update(fires, timestamp=timestamp)
        if not event:
            return

        if event.is_new_alert:
            is_breakout = any(f.get("is_breakout") for f in (fires or []))
            breakout_zone = next((f.get("breakout_zone") for f in (fires or []) if f.get("breakout_zone")), None)

            logger.info("FIRE/SMOKE CONFIRMED: %s (conf=%.2f, incident_id=%s, breakout=%s, zone=%s)",
                        event.event_type, event.confidence, event.incident_id, is_breakout, breakout_zone)
            if self.on_fire and self.evidence_folder:
                try:
                    # Save dual evidence: Annotated + Raw
                    fname_annotated = f"{event.event_type}_{uuid.uuid4().hex}_annotated.jpg"
                    fname_raw = f"{event.event_type}_{uuid.uuid4().hex}_raw.jpg"
                    path_annotated = os.path.join(self.evidence_folder, fname_annotated)
                    path_raw = os.path.join(self.evidence_folder, fname_raw)

                    cv2.imwrite(path_annotated, annotated_frame)
                    cv2.imwrite(path_raw, raw_frame)

                    self.on_fire(
                        alert_type=event.event_type,
                        confidence=round(event.confidence, 3),
                        image_path=path_annotated,
                        camera_id=self.camera_id,
                        is_breakout=is_breakout,
                        breakout_zone=breakout_zone,
                    )
                except Exception as e:
                    logger.error("Failed to save fire evidence: %s", e)

