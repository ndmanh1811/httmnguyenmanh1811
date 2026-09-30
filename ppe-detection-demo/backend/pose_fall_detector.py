"""
pose_fall_detector.py  —  v5 HYBRID 3-PATH (YOLO26-Pose + PyTorch LSTM)
-----------------------------------------------------------------------
Kiến trúc 3-Path Ensemble cho phát hiện ngã thời gian thực:
  Path A (Chính)  : Bounding Box Dynamics — AR change, height drop, vertical velocity
  Path B (Bổ trợ) : Pose estimation — Khung xương 17 điểm COCO (Ưu tiên YOLO26s-Pose NMS-Free)
  Path C (Bối cảnh): Motion burst & Inactivity — Vận tốc rơi đột ngột và bất động sau ngã
  AI Model        : Mạng PyTorch LSTM chuỗi thời gian phân loại hành động ngã
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from enhancements import (
    apply_adaptive_clahe,
    slice_frame_2x2,
    merge_sliced_pose_detections,
    OneEuroPoseFilter,
    _calc_box_iou,
)

class FallLSTM(nn.Module):
    def __init__(self, input_size: int = 51, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def extract_lstm_features(kpts_array: np.ndarray) -> list[float]:
    features: list[float] = []
    valid_pts = [pt for pt in kpts_array if pt[2] > 0.3]
    if not valid_pts:
        return [0.0] * 51
    xs = [pt[0] for pt in valid_pts]
    ys = [pt[1] for pt in valid_pts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    height = max(max(ys) - min(ys), 1.0)
    for pt in kpts_array:
        x, y, conf = pt[0], pt[1], pt[2]
        norm_x = (x - cx) / height if conf > 0.3 else 0.0
        norm_y = (y - cy) / height if conf > 0.3 else 0.0
        features.extend([norm_x, norm_y, float(conf)])
    return features

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_YOLO26 = os.path.join(BASE_DIR, "models_dir", "yolo26s-pose.pt")
POSE_MODEL_YOLOV8 = os.path.join(BASE_DIR, "models_dir", "yolov8s-pose.pt")

KPT_CONF_THRESH = 0.25  # Tu 0.45 -> 0.25 giup hien thi ro net cac khop tren webcam goc gan
TORSO_AR_FALL_THRESH = 1.5

# BB Dynamics thresholds (Path A) — thắt chặt để tránh false positive
BB_HISTORY_LEN = 12
BB_AR_FALL_THRESH = 1.4       # was 1.1 → ngồi xổm ar~1.2 không trigger nữa
BB_AR_RISE_THRESH = 0.5
BB_HEIGHT_DROP_PCT = 0.25     # was 0.15 → cúi nhặt đồ ~15-20% không trigger
BB_CENTER_VEL_THRESH = 12.0   # was 8.0 → chỉ rơi nhanh thực sự mới trigger

# Motion / Impact thresholds (Path C) — cần rơi mạnh rõ ràng
MOTION_VEL_THRESH = 15.0      # was 10.0
MOTION_ACCEL_THRESH = 25.0    # was 20.0

# Inactivity Verification thresholds (Xác thực bất động sau ngã)
INACTIVITY_FRAMES_REQUIRED = 3   # Cần ~0.5 - 0.7 giây nằm bất động (nhanh, bắt tốt cả video ngắn)
MAX_IMMOBILE_MOVEMENT = 25.0      # Giới hạn dịch chuyển tâm (pixels/frame) khi nằm im

# Pose confidence gating (Path B)
POSE_CONF_GATE = 0.40

# Ensemble — yêu cầu ít nhất 2 path, single-path override cao hơn
ENSEMBLE_MIN_PATHS = 2

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class PoseFallDetector:
    def __init__(
        self,
        conf: float = 0.25,
        angle_threshold: float = 50.0,
        aspect_ratio_threshold: float = 1.05,
        required_consecutive_frames: int = 2,
    ) -> None:
        self.conf = conf
        self.angle_threshold = angle_threshold
        self.required_consecutive_frames = required_consecutive_frames

        self._smoothing_factor: float = 0.75
        self._max_match_dist: float = 250.0

        # Ưu tiên YOLO26s-Pose (NMS-Free, 63.0 mAP) -> YOLOv8s-Pose -> Tự động tải yolo26s-pose.pt
        if os.path.isfile(POSE_MODEL_YOLO26):
            weights = POSE_MODEL_YOLO26
        elif os.path.isfile(POSE_MODEL_YOLOV8):
            weights = POSE_MODEL_YOLOV8
        elif os.path.isfile(os.path.join(BASE_DIR, "yolo26s-pose.pt")):
            weights = os.path.join(BASE_DIR, "yolo26s-pose.pt")
        elif os.path.isfile(os.path.join(BASE_DIR, "yolov8s-pose.pt")):
            weights = os.path.join(BASE_DIR, "yolov8s-pose.pt")
        else:
            weights = "yolo26s-pose.pt"

        logger.info("Initializing PoseFallDetector v5 HYBRID with model: %s", weights)
        self.model = YOLO(weights)

        self._last_detected_persons: list[dict] = []
        self._prev_persons: list[dict] = []
        self._cached_detected_persons: list[dict] = []
        self._last_detected_time: float = 0.0

        self._bb_history: dict[int, deque] = {}
        self._cy_history: dict[int, deque] = {}
        self._fall_frames: dict[int, int] = {}
        self._next_person_id: int = 1

        self._is_fall_confirmed: bool = False
        self._last_fall_time: float = 0.0

        # LSTM AI
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = FallLSTM().to(self.device)
        self.use_lstm = False
        lstm_path = os.path.join(BASE_DIR, 'models_dir', 'fall_lstm.pth')
        if os.path.isfile(lstm_path):
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
                self.lstm_model.eval()
                self.use_lstm = True
                logger.info('Tích hợp thành công bộ não AI LSTM vào hệ thống!')
            except Exception as e:
                logger.error(f'Lỗi load LSTM: {e}')
                
        self._lstm_history = {}
        self.inactivity_frames_required: int = INACTIVITY_FRAMES_REQUIRED
        self._fall_candidates: dict[int, dict] = {}
        self._fall_latch: dict[int, int] = {}
        self._kpt_motion_history: dict[int, deque] = {}
        self._standing_height: dict[int, float] = {}
        self._pose_filter = OneEuroPoseFilter()

    def reset(self) -> None:
        """Reset toàn bộ trạng thái tracking và phát hiện ngã khi bắt đầu video mới."""
        self._last_detected_persons.clear()
        self._prev_persons.clear()
        self._cached_detected_persons.clear()
        self._last_detected_time = 0.0
        self._bb_history.clear()
        self._cy_history.clear()
        self._fall_frames.clear()
        self._lstm_history.clear()
        self._fall_candidates.clear()
        self._fall_latch.clear()
        self._kpt_motion_history.clear()
        self._standing_height.clear()
        self._pose_filter.reset()
        self._next_person_id = 1
        self._is_fall_confirmed = False
        self._last_fall_time = 0.0


    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_to_prev(
        self,
        new_bboxes: list[list[int]],
        ppe_track_items: list[dict] | None = None,
    ) -> list[int | None | str]:
        """
        3-phase matching cho Pose detections:
        Phase 0: Spatial IoU matching với ByteTrack items (order-independent, length-independent)
        Phase 1: IoU matching với previous frame tracks
        Phase 2: Adaptive center distance matching
        
        Returns: list of (int index | str marker "BYTE_ID:<id>" | None)
        """
        matched: list[int | None | str] = [None] * len(new_bboxes)
        if not self._prev_persons:
            # Không có track trước -> vẫn chạy Phase 0 để gán ByteTrack ID cho người mới
            pass
        else:
            used_prev: set[int] = set()

            # Giai đoạn 0: Spatial IoU matching với ByteTrack items (order-independent!)
            if ppe_track_items is not None:
                for new_i, bbox in enumerate(new_bboxes):
                    best_iou = 0.35
                    best_track_id = None
                    for item in ppe_track_items:
                        iou = _calc_box_iou(bbox, item["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_track_id = item["id"]
                    if best_track_id is not None:
                        # Tìm trong prev_persons xem track_id này đã có track chưa
                        found = False
                        for j, prev in enumerate(self._prev_persons):
                            if j in used_prev:
                                continue
                            if prev.get("track_id") == best_track_id:
                                matched[new_i] = j
                                used_prev.add(j)
                                found = True
                                break
                        if not found:
                            # Match được với PPE nhưng chưa có track -> marker để gán pid = ByteTrack ID
                            matched[new_i] = f"BYTE_ID:{best_track_id}"

            # Giai đoạn 1: IoU Matching với previous tracks
            for new_i, bbox in enumerate(new_bboxes):
                if matched[new_i] is not None:
                    continue
                best_iou = 0.20
                best_j = None
                for j, prev in enumerate(self._prev_persons):
                    if j in used_prev:
                        continue
                    iou = _calc_box_iou(bbox, prev["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j is not None:
                    matched[new_i] = best_j
                    used_prev.add(best_j)

            # Giai đoạn 2: Khoảng cách tâm thích ứng
            for new_i, bbox in enumerate(new_bboxes):
                if matched[new_i] is not None:
                    continue
                ncx = (bbox[0] + bbox[2]) / 2.0
                ncy = (bbox[1] + bbox[3]) / 2.0
                bw = max(bbox[2] - bbox[0], 1)
                bh = max(bbox[3] - bbox[1], 1)
                diag = math.hypot(bw, bh)
                adaptive_max_dist = max(self._max_match_dist, 1.8 * diag)

                best_dist = adaptive_max_dist
                best_j = None
                for j, prev in enumerate(self._prev_persons):
                    if j in used_prev:
                        continue
                    pb = prev["bbox"]
                    pcx = (pb[0] + pb[2]) / 2.0
                    pcy = (pb[1] + pb[3]) / 2.0
                    d = math.hypot(ncx - pcx, ncy - pcy)
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                if best_j is not None:
                    matched[new_i] = best_j
                    used_prev.add(best_j)

        return matched

    # ------------------------------------------------------------------
    # EMA Smoothing
    # ------------------------------------------------------------------

    def _smooth_keypoints(
        self,
        pid: int,
        current_kpts: np.ndarray,
        prev_kpts: np.ndarray | None,
    ) -> np.ndarray:
        return self._pose_filter.smooth_pose(pid, current_kpts, dt=1.0 / 25.0, kpt_conf_thresh=KPT_CONF_THRESH)

    # ------------------------------------------------------------------
    # Pose helpers (Path B)
    # ------------------------------------------------------------------

    def calculate_torso_angle(self, keypoints: np.ndarray) -> float | None:
        try:
            l_sh, r_sh = keypoints[5], keypoints[6]
            l_hip, r_hip = keypoints[11], keypoints[12]

            if l_sh[2] >= KPT_CONF_THRESH and r_sh[2] >= KPT_CONF_THRESH:
                mid_shoulder = (l_sh[:2] + r_sh[:2]) / 2.0
            elif l_sh[2] >= KPT_CONF_THRESH:
                mid_shoulder = l_sh[:2].copy()
            elif r_sh[2] >= KPT_CONF_THRESH:
                mid_shoulder = r_sh[:2].copy()
            else:
                return None

            if l_hip[2] >= KPT_CONF_THRESH and r_hip[2] >= KPT_CONF_THRESH:
                mid_hip = (l_hip[:2] + r_hip[:2]) / 2.0
            elif l_hip[2] >= KPT_CONF_THRESH:
                mid_hip = l_hip[:2].copy()
            elif r_hip[2] >= KPT_CONF_THRESH:
                mid_hip = r_hip[:2].copy()
            else:
                return None

            dx = float(mid_shoulder[0] - mid_hip[0])
            dy = float(mid_shoulder[1] - mid_hip[1])
            angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
            return math.degrees(angle_rad)
        except Exception:
            return None

    def _calculate_torso_ar(self, keypoints: np.ndarray) -> float | None:
        try:
            l_sh, r_sh = keypoints[5], keypoints[6]
            l_hip, r_hip = keypoints[11], keypoints[12]

            if l_sh[2] < KPT_CONF_THRESH or r_sh[2] < KPT_CONF_THRESH:
                return None

            shoulder_width = abs(float(r_sh[0]) - float(l_sh[0]))
            mid_sh_y = float((l_sh[1] + r_sh[1]) / 2.0)

            if l_hip[2] >= KPT_CONF_THRESH and r_hip[2] >= KPT_CONF_THRESH:
                mid_hip_y = float((l_hip[1] + r_hip[1]) / 2.0)
            elif l_hip[2] >= KPT_CONF_THRESH:
                mid_hip_y = float(l_hip[1])
            elif r_hip[2] >= KPT_CONF_THRESH:
                mid_hip_y = float(r_hip[1])
            else:
                return None

            torso_h = abs(mid_sh_y - mid_hip_y)
            if torso_h < 5.0:
                return 99.0
            return shoulder_width / torso_h
        except Exception:
            return None

    def is_head_neck_upright(self, keypoints: np.ndarray) -> bool:
        """Kiem tra xem dau va co co dang thang dung tren 2 vai khong (ngoi hoac dung truoc camera)."""
        try:
            # Nếu thấy hông và góc thân > 40 độ -> rõ ràng đang cúi hoặc ngã ngang, KHÔNG tính là ngồi thẳng!
            l_hip, r_hip = keypoints[11], keypoints[12]
            if l_hip[2] >= 0.35 and r_hip[2] >= 0.35:
                torso_angle = self.calculate_torso_angle(keypoints)
                if torso_angle is not None and torso_angle > 40.0:
                    return False

            nose = keypoints[0]
            l_sh, r_sh = keypoints[5], keypoints[6]
            if nose[2] >= 0.35 and (l_sh[2] >= 0.35 or r_sh[2] >= 0.35):
                if l_sh[2] >= 0.35 and r_sh[2] >= 0.35:
                    mid_sh_x = float((l_sh[0] + r_sh[0]) / 2.0)
                    mid_sh_y = float((l_sh[1] + r_sh[1]) / 2.0)
                elif l_sh[2] >= 0.35:
                    mid_sh_x = float(l_sh[0])
                    mid_sh_y = float(l_sh[1])
                else:
                    mid_sh_x = float(r_sh[0])
                    mid_sh_y = float(r_sh[1])

                dy = mid_sh_y - float(nose[1])  # dy > 0 nghĩa là mũi ở TRÊN vai
                dx = abs(float(nose[0]) - mid_sh_x)
                if dy > 20.0:
                    angle = math.degrees(math.atan2(dx, dy))
                    # Nếu đầu thẳng đứng trên vai (góc lệch phương đứng < 35 độ) -> Đang ngồi/đứng thẳng!
                    if angle < 35.0:
                        return True
        except Exception:
            pass
        return False

    def _avg_kpt_conf(self, kpts: np.ndarray) -> float:
        confs = [kpts[i][2] for i in range(len(kpts)) if i in (5, 6, 11, 12)]
        return sum(confs) / len(confs) if confs else 0.0

    def _is_valid_human_skeleton(
        self,
        kpts: np.ndarray,
        bbox: list[int],
        frame: np.ndarray | None = None,
        is_in_smoke: bool = False,
    ) -> tuple[bool, str]:
        """
        Xac minh cau truc giai phau nguoi thuc te (Human Skeleton Integrity Gate).
        Loai bo triet de cac diem khop ao giac sinh ra ben trong dam khoi, hoi nuoc, lua hoac do vat.
        """
        min_kpt_conf = 0.35 if is_in_smoke else 0.20
        valid_kpts = [k for k in kpts if k[2] >= min_kpt_conf]
        num_valid = len(valid_kpts)

        min_required = 6 if is_in_smoke else 4
        if num_valid < min_required:
            return False, f"Keypoint count too low ({num_valid} < {min_required})"

        # 1. Khung than tren / That lung (Torso Core: Shoulders 5,6 va Hips 11,12)
        has_shoulder = (kpts[5][2] >= min_kpt_conf or kpts[6][2] >= min_kpt_conf)
        l_hip, r_hip = kpts[11][2], kpts[12][2]
        has_hip = (l_hip >= min_kpt_conf or r_hip >= min_kpt_conf)

        # 2. Khung than duoi (Lower body: Hips hoac Knees)
        l_knee, r_knee = kpts[13][2], kpts[14][2]
        has_lower_body = has_hip or (l_knee >= min_kpt_conf or r_knee >= min_kpt_conf)
        if is_in_smoke and not has_lower_body:
            return False, "Missing lower body (smoke phantom has no hips/legs)"

        if is_in_smoke and not (has_shoulder and has_hip):
            if num_valid < 7:
                return False, "Missing human torso core (shoulders or hips missing)"

        # 3. Kiem tra do sac net ket cau bien (Laplacian Edge Variance)
        # Chi ap dung kiem tra do mo giong khoi khi dang nam trong vung khoi
        if is_in_smoke and frame is not None and getattr(frame, "size", 0) > 0:
            x1, y1, x2, y2 = bbox
            h_f, w_f = frame.shape[:2]
            crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            if crop.size > 100:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if lap_var < 20.0:
                    return False, f"Blurry texture like smoke (Laplacian={lap_var:.1f} < 20)"

        # 4. Do bao phu khong gian giai phau (Anatomical span)
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        xs = [k[0] for k in valid_kpts]
        ys = [k[1] for k in valid_kpts]
        kpt_w = max(xs) - min(xs)
        kpt_h = max(ys) - min(ys)

        if (kpt_w / bw < 0.12) and (kpt_h / bh < 0.12):
            return False, "Keypoints tightly clustered (phantom noise)"

        return True, "Valid skeleton"

    def _classify_pose(self, torso_angle: float | None, kpts: np.ndarray) -> tuple[str, float]:
        if torso_angle is not None:
            if torso_angle >= self.angle_threshold:
                return "fall", round(torso_angle, 1)
            elif torso_angle >= 35.0:
                return "bending", round(torso_angle, 1)
            else:
                return "normal", round(torso_angle, 1)
        else:
            torso_ar = self._calculate_torso_ar(kpts)
            if torso_ar is not None and torso_ar >= TORSO_AR_FALL_THRESH:
                return "fall", 0.0
            return "normal", 0.0

    # ------------------------------------------------------------------
    # PATH A: Bounding Box Dynamics
    # ------------------------------------------------------------------

    def _update_bb_history(self, pid: int, bbox: list[int]) -> None:
        if pid not in self._bb_history:
            self._bb_history[pid] = deque(maxlen=BB_HISTORY_LEN)
        self._bb_history[pid].append(bbox)

    def _check_bb_path(self, pid: int, current_bbox: list[int]) -> float:
        history = self._bb_history.get(pid)
        if not history or len(history) < 4:
            return 0.0

        x1, y1, x2, y2 = current_bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        current_ar = bw / bh
        current_cy = (y1 + y2) / 2.0

        hist_list = list(history)
        if len(hist_list) < 3:
            return 0.0

        prev_bboxes = hist_list[:-1]
        avg_h = sum(b[3] - b[1] for b in prev_bboxes) / len(prev_bboxes)

        score = 0.0
        signals = []

        if current_ar >= BB_AR_FALL_THRESH:
            score += 0.35
            signals.append(f"AR={current_ar:.2f}")

        if avg_h > 10:
            height_drop = 1.0 - (bh / avg_h)
            if height_drop >= BB_HEIGHT_DROP_PCT:
                score += 0.35
                signals.append(f"Hdrop={height_drop:.0%}")

        if len(prev_bboxes) >= 2:
            prev_cy_list = [(b[1] + b[3]) / 2.0 for b in prev_bboxes[-3:]]
            if len(prev_cy_list) >= 2:
                avg_prev_cy = sum(prev_cy_list) / len(prev_cy_list)
                cy_velocity = current_cy - avg_prev_cy
                if cy_velocity > BB_CENTER_VEL_THRESH:
                    score += 0.30
                    signals.append(f"Vcy={cy_velocity:.1f}")

        if score > 0:
            logger.debug("PathA pid=%s score=%.2f %s", pid, score, signals)

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # PATH B: Pose (gated by confidence)
    # ------------------------------------------------------------------

    def _check_pose_path(self, kpts: np.ndarray) -> float:
        avg_conf = self._avg_kpt_conf(kpts)
        if avg_conf < POSE_CONF_GATE:
            return 0.0

        torso_angle = self.calculate_torso_angle(kpts)
        if torso_angle is None:
            torso_ar = self._calculate_torso_ar(kpts)
            if torso_ar is not None and torso_ar >= TORSO_AR_FALL_THRESH:
                return 0.7 * min(avg_conf / 1.0, 1.0)
            return 0.0

        if torso_angle >= self.angle_threshold:
            return 0.8 * min(avg_conf / 1.0, 1.0)
        elif torso_angle >= 40.0:
            return 0.3 * min(avg_conf / 1.0, 1.0)

        return 0.0

    # ------------------------------------------------------------------
    # PATH C: Motion / Impact Detection
    # ------------------------------------------------------------------

    def _update_cy_history(self, pid: int, cy: float) -> None:
        if pid not in self._cy_history:
            self._cy_history[pid] = deque(maxlen=8)
        self._cy_history[pid].append(cy)

    def _check_motion_path(self, pid: int, current_cy: float) -> float:
        history = self._cy_history.get(pid)
        if not history or len(history) < 3:
            return 0.0

        cy_list = list(history)
        velocity = current_cy - cy_list[-2] if len(cy_list) >= 2 else 0.0
        if len(cy_list) >= 3:
            prev_velocity = cy_list[-2] - cy_list[-3]
            acceleration = velocity - prev_velocity
        else:
            acceleration = 0.0

        score = 0.0
        signals = []

        if velocity > MOTION_VEL_THRESH:
            score += 0.50
            signals.append(f"V={velocity:.1f}")
        if acceleration > MOTION_ACCEL_THRESH:
            score += 0.50
            signals.append(f"A={acceleration:.1f}")

        if score > 0:
            logger.debug("PathC pid=%s score=%.2f %s", pid, score, signals)

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # PATH D: Keypoint Biomechanical Velocity (Hips & Shoulders)
    # ------------------------------------------------------------------

    def _update_kpt_motion_history(self, pid: int, kpts: np.ndarray, bbox: list[int]) -> None:
        if pid not in self._kpt_motion_history:
            self._kpt_motion_history[pid] = deque(maxlen=8)

        x1, y1, x2, y2 = bbox
        bh = max(float(y2 - y1), 10.0)
        bw = max(float(x2 - x1), 5.0)

        # Cập nhật chiều cao đứng chuẩn của người khi cơ thể đứng dọc
        if bh > 1.25 * bw:
            curr_ref = self._standing_height.get(pid, bh)
            self._standing_height[pid] = max(curr_ref, bh)

        # Hông: 11 (trái), 12 (phải)
        l_hip, r_hip = kpts[11], kpts[12]
        if l_hip[2] >= 0.3 and r_hip[2] >= 0.3:
            hip_y = (float(l_hip[1]) + float(r_hip[1])) / 2.0
        elif l_hip[2] >= 0.3:
            hip_y = float(l_hip[1])
        elif r_hip[2] >= 0.3:
            hip_y = float(r_hip[1])
        else:
            hip_y = None

        # Vai: 5 (trái), 6 (phải)
        l_sh, r_sh = kpts[5], kpts[6]
        if l_sh[2] >= 0.3 and r_sh[2] >= 0.3:
            sh_y = (float(l_sh[1]) + float(r_sh[1])) / 2.0
        elif l_sh[2] >= 0.3:
            sh_y = float(l_sh[1])
        elif r_sh[2] >= 0.3:
            sh_y = float(r_sh[1])
        else:
            sh_y = None

        self._kpt_motion_history[pid].append({
            "hip_y": hip_y,
            "sh_y": sh_y,
            "height": bh,
            "time": time.time(),
        })

    def _check_kpt_velocity_path(self, pid: int) -> tuple[float, dict]:
        """Tính vận tốc rơi của trọng tâm Hông và trục Vai chuẩn hóa theo chiều cao cơ thể."""
        history = self._kpt_motion_history.get(pid)
        if not history or len(history) < 2:
            return 0.0, {"v_hip": 0.0, "v_sh": 0.0}

        hist = list(history)
        ref_h = self._standing_height.get(pid, hist[-1]["height"])
        ref_h = max(ref_h, 20.0)

        valid_hips = [h["hip_y"] for h in hist if h["hip_y"] is not None]
        valid_shs = [h["sh_y"] for h in hist if h["sh_y"] is not None]

        score = 0.0
        v_hip = 0.0
        v_sh = 0.0

        if len(valid_hips) >= 2:
            v_hip = (valid_hips[-1] - valid_hips[-2]) / ref_h
            if len(valid_hips) >= 3:
                prev_v_hip = (valid_hips[-2] - valid_hips[-3]) / ref_h
                a_hip = v_hip - prev_v_hip
            else:
                a_hip = 0.0

            if v_hip > 0.10:
                score += 0.50
            elif v_hip > 0.05:
                score += 0.30

            if a_hip > 0.07:
                score += 0.25

        if len(valid_shs) >= 2:
            v_sh = (valid_shs[-1] - valid_shs[-2]) / ref_h
            if v_sh > 0.10:
                score += 0.25
            elif v_sh > 0.05:
                score += 0.15

        return min(score, 1.0), {"v_hip": round(v_hip, 2), "v_sh": round(v_sh, 2)}

    # ------------------------------------------------------------------
    # PATH E: Ground Plane Proximity & Post-Fall Relative Pose
    # ------------------------------------------------------------------

    def _check_ground_proximity(self, kpts: np.ndarray, bbox: list[int], pid: int) -> tuple[float, dict]:
        """Kiểm tra khoảng cách với mặt sàn và tư thế tương đối sau ngã (Ground Plane & Relative Pose)."""
        x1, y1, x2, y2 = bbox
        ref_h = self._standing_height.get(pid, float(y2 - y1))
        ref_h = max(ref_h, 20.0)

        # 1. Tọa độ mắt cá chân (Ankle level ~ Ground level)
        l_ank, r_ank = kpts[15], kpts[16]
        if l_ank[2] >= 0.3 and r_ank[2] >= 0.3:
            ground_y = max(float(l_ank[1]), float(r_ank[1]))
        elif l_ank[2] >= 0.3:
            ground_y = float(l_ank[1])
        elif r_ank[2] >= 0.3:
            ground_y = float(r_ank[1])
        else:
            ground_y = float(y2)

        # 2. Tọa độ hông
        l_hip, r_hip = kpts[11], kpts[12]
        if l_hip[2] >= 0.3 and r_hip[2] >= 0.3:
            hip_y = (float(l_hip[1]) + float(r_hip[1])) / 2.0
        elif l_hip[2] >= 0.3:
            hip_y = float(l_hip[1])
        elif r_hip[2] >= 0.3:
            hip_y = float(r_hip[1])
        else:
            hip_y = None

        # 3. Tọa độ đầu
        nose = kpts[0]
        head_y = float(nose[1]) if nose[2] >= 0.3 else None

        score = 0.0
        delta_hip_ground = 1.0
        delta_head_hip = 1.0

        if hip_y is not None:
            delta_hip_ground = abs(ground_y - hip_y) / ref_h
            if delta_hip_ground < 0.20:
                score += 0.50
            elif delta_hip_ground < 0.26:
                score += 0.25

        # QUAN TRỌNG: Đầu ngang hông CHỈ là nằm sàn nếu hông cũng đang sát mặt sàn!
        # Nếu hông ở trên cao (delta_hip_ground >= 0.26) -> Đây là cúi người (Bending), không phải nằm sàn!
        if head_y is not None and hip_y is not None:
            delta_head_hip = abs(head_y - hip_y) / ref_h
            if delta_hip_ground < 0.26:
                if delta_head_hip < 0.22:
                    score += 0.35
                elif delta_head_hip < 0.35:
                    score += 0.15

        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        if bw > 1.15 * bh and delta_hip_ground < 0.26:
            score += 0.25

        return min(score, 1.0), {
            "hip_ground": round(delta_hip_ground, 2),
            "head_hip": round(delta_head_hip, 2),
            "ar": round(bw / bh, 2),
        }

    # ------------------------------------------------------------------
    # ENSEMBLE VOTING
    # ------------------------------------------------------------------

    def _ensemble_vote(
        self,
        bb_score: float,
        pose_score: float,
        motion_score: float,
        kpt_vel_score: float = 0.0,
        ground_score: float = 0.0,
        lstm_score: float = 0.0,
    ) -> tuple[bool, float]:
        is_falling_motion = (kpt_vel_score >= 0.25 or motion_score >= 0.35)
        is_ground = (ground_score >= 0.25)
        is_bb_drop = (bb_score >= 0.35)

        # ĐIỀU KIỆN TIÊN QUYẾT: Để là một cú ngã, BẮT BUỘC phải có ít nhất 1 trong 3 yếu tố:
        # - Hoặc đang nằm sát sàn (is_ground)
        # - Hoặc có động học rơi sụp sàn (is_falling_motion)
        # - Hoặc có sự sụp đổ chiều cao hộp bao (is_bb_drop)
        # Nếu KHÔNG CÓ bất kỳ yếu tố nào ở trên, thì dù lưng gập ngang (pose cao) cũng chỉ là CÚI NGƯỜI!
        if not (is_ground or is_falling_motion or is_bb_drop):
            return False, 0.0

        signals = [bb_score, pose_score, motion_score, kpt_vel_score, ground_score]
        if lstm_score > 0.5:
            signals.append(lstm_score)

        paths_triggered = sum(1 for s in signals if s >= 0.25)
        max_score = max(signals)

        # 1. Dáng nằm ngang rõ rệt VÀ sát sàn
        if pose_score >= 0.65 and is_ground:
            return True, max(max_score, 0.80)

        # 2. Có động học rơi mạnh + tư thế tiếp đất hoặc pose nằm
        if is_falling_motion and (is_ground or pose_score >= 0.30 or bb_score >= 0.30):
            return True, max(max_score, 0.75)

        # 3. Ít nhất 2 path đồng thuận vượt ngưỡng (BẮT BUỘC PHẢI CÓ TƯ THẾ HOẶC TIẾP ĐẤT)
        has_posture_signal = (pose_score >= 0.25 or ground_score >= 0.25 or lstm_score > 0.50)
        if paths_triggered >= ENSEMBLE_MIN_PATHS and max_score >= 0.45 and has_posture_signal:
            return True, max_score

        # Single-path override nếu cực kỳ rõ rệt sát sàn
        if ground_score >= 0.75:
            return True, max_score

        return False, 0.0

    # ------------------------------------------------------------------
    # Cleanup stale tracking data
    # ------------------------------------------------------------------

    def _cleanup_stale(self, active_pids: set[int]) -> None:
        for stale_pid in set(self._bb_history.keys()) - active_pids:
            self._bb_history.pop(stale_pid, None)
            self._cy_history.pop(stale_pid, None)
            self._fall_frames.pop(stale_pid, None)
            self._lstm_history.pop(stale_pid, None)
            self._fall_candidates.pop(stale_pid, None)
            self._fall_latch.pop(stale_pid, None)
            self._kpt_motion_history.pop(stale_pid, None)
            self._standing_height.pop(stale_pid, None)
        self._pose_filter.cleanup_stale(active_pids)

    # ------------------------------------------------------------------
    # Detect chinh
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: Any,
        smoke_boxes: list | None = None,
        imgsz: int | None = None,
        use_clahe: bool = False,
        use_sahi: bool = False,
        timestamp: float | None = None,
        ppe_track_items: list[dict] | None = None,
    ) -> list[dict]:
        proc_frame = apply_adaptive_clahe(frame) if (use_clahe and frame is not None and getattr(frame, "size", 0) > 0) else frame
        curr_time = time.time() if timestamp is None else timestamp

        predict_kwargs: dict[str, Any] = {"conf": self.conf, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        results = self.model.predict(proc_frame, **predict_kwargs)
        falls: list[dict] = []
        self._last_detected_persons = []

        global_bboxes: list[list[int]] = []
        global_confs: list[float] = []
        global_kpts: list[np.ndarray] = []

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
                global_bboxes = [list(map(int, b.xyxy[0])) for b in r.boxes]
                global_confs = [float(b.conf[0]) for b in r.boxes]
                global_kpts = [k for k in r.keypoints.data.cpu().numpy()]

        # SAHI Tiled Slicing if enabled
        if use_sahi and proc_frame is not None and getattr(proc_frame, "size", 0) > 0:
            slice_imgs, slice_coords = slice_frame_2x2(proc_frame)
            slice_res = self.model.predict(slice_imgs, conf=self.conf, imgsz=640, verbose=False)
            global_bboxes, global_confs, global_kpts = merge_sliced_pose_detections(
                global_bboxes, global_confs, global_kpts, slice_res, slice_coords, iou_thresh=0.35
            )

        if not global_bboxes:
            # Tăng lost_count cho các prev_persons hiện tại (duy trì bộ nhớ tối đa 6 frames ~0.25s)
            surviving_prev: list[dict] = []
            active_pids: set[int] = set()
            for p in self._prev_persons:
                lost = p.get("lost_count", 0) + 1
                if lost <= 6:
                    p_copy = dict(p)
                    p_copy["lost_count"] = lost
                    surviving_prev.append(p_copy)
                    active_pids.add(p["pid"])
            self._prev_persons = surviving_prev
            self._cleanup_stale(active_pids)
            self._last_detected_persons = []
            if not falls:
                self._is_fall_confirmed = False
            return falls

        matched_prev = self._match_to_prev(global_bboxes, ppe_track_items)
        new_prev_persons: list[dict] = []
        active_pids: set[int] = set()

        for i in range(len(global_bboxes)):
            conf = global_confs[i]
            x1, y1, x2, y2 = global_bboxes[i]
            raw_kpts = global_kpts[i]

            # Kiem tra xem candidate box co nam trong vung khoi hoac lua khong
            is_in_smoke = False
            if smoke_boxes:
                b_area = max(1.0, (x2 - x1) * (y2 - y1))
                for s_box in smoke_boxes:
                    sx1, sy1, sx2, sy2 = s_box
                    ix1, iy1 = max(x1, sx1), max(y1, sy1)
                    ix2, iy2 = min(x2, sx2), min(y2, sy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if (inter / b_area) >= 0.30:
                            is_in_smoke = True
                            break

            # Kiem tra tinh toan ven cua khung xuong nguoi that (loai bo ao giac khoi/vat the)
            is_valid_human, reason = self._is_valid_human_skeleton(raw_kpts, [x1, y1, x2, y2], frame=proc_frame, is_in_smoke=is_in_smoke)
            if not is_valid_human:
                logger.debug("Bo qua candidate pose khong phai nguoi that: %s", reason)
                continue

            prev_idx = matched_prev[i]
            
            # Handle Phase 0 marker: "BYTE_ID:<track_id>" -> assign ByteTrack ID as pid directly
            if isinstance(prev_idx, str) and prev_idx.startswith("BYTE_ID:"):
                pid = int(prev_idx.split(":")[1])
            elif prev_idx is not None and "pid" in self._prev_persons[prev_idx]:
                pid = self._prev_persons[prev_idx]["pid"]
            else:
                pid = self._next_person_id
                self._next_person_id += 1
            active_pids.add(pid)

            prev_kpts = self._prev_persons[prev_idx]["kpts"] if prev_idx is not None else None
            kpts = self._smooth_keypoints(pid, raw_kpts, prev_kpts)

            # Path A: BB Dynamics
            self._update_bb_history(pid, [x1, y1, x2, y2])
            bb_score = self._check_bb_path(pid, [x1, y1, x2, y2])

            # Path B: Pose
            pose_score = self._check_pose_path(kpts)

            # Path C: Motion
            center_y = (y1 + y2) / 2.0
            self._update_cy_history(pid, center_y)
            motion_score = self._check_motion_path(pid, center_y)

            # Path D: Keypoint Biomechanical Velocity (Hông & Vai)
            self._update_kpt_motion_history(pid, kpts, [x1, y1, x2, y2])
            kpt_vel_score, kpt_vel_info = self._check_kpt_velocity_path(pid)

            # Path E: Ground Plane Proximity & Post-Fall Relative Pose
            ground_score, ground_info = self._check_ground_proximity(kpts, [x1, y1, x2, y2], pid)

            # --- AI LSTM PATH ---
            lstm_fall_score = 0.0
            if self.use_lstm:
                if pid not in self._lstm_history:
                    self._lstm_history[pid] = deque(maxlen=30)
                feats = extract_lstm_features(kpts)
                self._lstm_history[pid].append(feats)
                if len(self._lstm_history[pid]) == 30:
                    with torch.no_grad():
                        input_tensor = torch.tensor([self._lstm_history[pid]], dtype=torch.float32).to(self.device)
                        out = self.lstm_model(input_tensor)
                        probs = torch.softmax(out, dim=1)[0]
                        lstm_fall_score = probs[1].item() # Xác suất là ngã (label 1)

            # Tinh toan goc than minh (Torso Angle)
            torso_angle = self.calculate_torso_angle(kpts)
            angle_val = round(torso_angle, 1) if torso_angle is not None else 0.0

            # Kiem tra xem nguoi co dang dung/ngoi thang truc khong
            head_upright = self.is_head_neck_upright(kpts)
            torso_upright = (torso_angle is not None and torso_angle < 35.0)
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            tall_bbox = (bbox_h > 1.25 * bbox_w) and (pose_score < 0.25 and bb_score < 0.25 and ground_score < 0.25)
            # is_upright: Dang dung/ngoi thang dung (truc dau-vai thang HOAC than dung thang HOAC bbox thang dung)
            is_upright = head_upright or (torso_upright and tall_bbox) or (torso_upright and pose_score == 0.0 and ground_score < 0.25)

            # Phân biệt Đang đứng cúi người (Bending) vs Ngã (Fall):
            # Nếu hông vẫn ở trên cao (delta_hip_ground >= 0.26) VÀ không có lực sụp rơi mạnh
            # -> ĐÂY LÀ ĐANG ĐỨNG CÚI NGƯỜI, KHÔNG PHẢI NGÃ!
            delta_hip_ground = ground_info.get("hip_ground", 1.0)
            is_motion = (kpt_vel_score >= 0.25 or motion_score >= 0.35)
            is_standing_bend = (
                delta_hip_ground >= 0.26
                and not is_motion
                and (torso_angle is not None and torso_angle >= 35.0)
            )

            # Ensemble Vote (5 paths: BB, Pose, Motion, Kpt Velocity, Ground Plane + LSTM)
            # NẾU ĐANG ĐỨNG/NGỒI THẲNG HOẶC ĐANG ĐỨNG CÚI: TUYỆT ĐỐI KHÔNG THỂ LÀ NGÃ!
            if is_upright:
                is_fall, fall_conf = False, 0.0
                self._fall_frames[pid] = 0
                self._fall_candidates.pop(pid, None)
                self._fall_latch[pid] = 0
            elif is_standing_bend:
                is_fall, fall_conf = False, 0.0
                self._fall_frames[pid] = 0
                self._fall_candidates.pop(pid, None)
                self._fall_latch[pid] = 0
            else:
                is_fall, fall_conf = self._ensemble_vote(
                    bb_score=bb_score,
                    pose_score=pose_score,
                    motion_score=motion_score,
                    kpt_vel_score=kpt_vel_score,
                    ground_score=ground_score,
                    lstm_score=lstm_fall_score,
                )

            if is_fall:
                self._fall_frames[pid] = self._fall_frames.get(pid, 0) + 1
            else:
                self._fall_frames[pid] = max(self._fall_frames.get(pid, 0) - 1, 0)

            fall_count = self._fall_frames[pid]
            is_verified_fall = False
            immobile_count = 0
            fall_duration = 0.0

            if is_upright:
                status = "normal"
                self._fall_latch[pid] = 0
                self._fall_candidates.pop(pid, None)
            elif is_standing_bend:
                status = "bending"
                self._fall_latch[pid] = 0
                self._fall_candidates.pop(pid, None)
            elif fall_count >= self.required_consecutive_frames:
                # Đã phát hiện dáng ngã! Kích hoạt quy trình kiểm tra bất động (Post-fall Inactivity)
                # Bắt buộc phải có yếu tố tiếp đất hoặc rơi mạnh mới kiểm tra bất động nằm sàn
                if ground_score >= 0.25 or bb_score >= 0.35 or is_motion:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if pid not in self._fall_candidates:
                        self._fall_candidates[pid] = {
                            "immobile_frames": 1,
                            "last_cx": cx,
                            "last_cy": cy,
                            "start_time": curr_time,
                        }
                        immobile_count = 1
                    else:
                        cand = self._fall_candidates[pid]
                        dist = math.hypot(cx - cand["last_cx"], cy - cand["last_cy"])
                        if dist <= MAX_IMMOBILE_MOVEMENT:
                            cand["immobile_frames"] += 1
                        else:
                            cand["immobile_frames"] = max(1, cand["immobile_frames"] - 1)
                        cand["last_cx"] = cx
                        cand["last_cy"] = cy
                        immobile_count = cand["immobile_frames"]

                    cand = self._fall_candidates[pid]
                    fall_duration = max(0.0, curr_time - cand.get("start_time", curr_time))
                    is_immobile_5s = (fall_duration >= 5.0) or (immobile_count >= 120)

                    if is_immobile_5s:
                        status = "fall_immobile"
                        is_verified_fall = True
                        self._fall_latch[pid] = 40
                    elif immobile_count >= self.inactivity_frames_required:
                        status = "fall"
                        is_verified_fall = True
                        self._fall_latch[pid] = 25
                    else:
                        status = "fall"
                else:
                    status = "bending" if (pose_score > 0.25 or bb_score > 0.2 or (torso_angle is not None and torso_angle >= 35.0)) else "normal"
            else:
                if fall_count == 0:
                    self._fall_candidates.pop(pid, None)

                cand = self._fall_candidates.get(pid)
                if cand:
                    fall_duration = max(0.0, curr_time - cand.get("start_time", curr_time))
                    immobile_count = cand.get("immobile_frames", 0)
                else:
                    fall_duration = 0.0
                    immobile_count = 0
                is_immobile_5s = (fall_duration >= 5.0) or (immobile_count >= 120)

                # Giữ trạng thái fall đỏ hoặc đen nếu vừa mới ngã và chưa đứng dậy
                latch_remain = self._fall_latch.get(pid, 0)
                if latch_remain > 0 and not is_upright:
                    self._fall_latch[pid] = latch_remain - 1
                    status = "fall_immobile" if is_immobile_5s else "fall"
                    is_verified_fall = True
                elif pose_score > 0.3 or bb_score > 0.2 or (torso_angle is not None and torso_angle >= 35.0):
                    status = "bending"
                else:
                    status = "normal"

            person_data: dict = {
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "pid": pid,
                "angle": angle_val,
                "status": status,
                "fall_frames": fall_count,
                "immobile_frames": immobile_count,
                "fall_duration": round(fall_duration, 1),
                "keypoints": kpts,
                "bb_score": bb_score,
                "pose_score": pose_score,
                "motion_score": motion_score,
                "kpt_vel_score": kpt_vel_score,
                "ground_score": ground_score,
                "lstm_score": lstm_fall_score,
            }
            self._last_detected_persons.append(person_data)
            new_prev_persons.append({
                "bbox": [x1, y1, x2, y2],
                "kpts": kpts,
                "pid": pid,
                "track_id": pid,  # pid is now ByteTrack ID when matched via Phase 0
                "lost_count": 0,
            })

            if is_verified_fall:
                self._is_fall_confirmed = True
                self._last_fall_time = curr_time
                falls.append({
                    "label": "Fall-Immobile (>5s)" if status == "fall_immobile" else "Fall-Detected",
                    "confidence": fall_conf if fall_conf > 0 else conf,
                    "bbox": [x1, y1, x2, y2],
                    "angle": angle_val,
                    "pid": pid,
                    "immobile_frames": immobile_count,
                    "fall_duration": round(fall_duration, 1),
                    "status": status,
                    "keypoints": kpts,
                    "paths": {
                        "bb": round(bb_score, 2),
                        "pose": round(pose_score, 2),
                        "motion": round(motion_score, 2),
                        "vel": round(kpt_vel_score, 2),
                        "grd": round(ground_score, 2),
                        "lstm": round(lstm_fall_score, 2),
                    },
                })

        # Giữ lại các track trước đó bị mất dấu tạm thời (tối đa 6 frames ~0.25s)
        matched_prev_indices = {idx for idx in matched_prev if isinstance(idx, int) and idx is not None}
        for j, prev in enumerate(self._prev_persons):
            if j not in matched_prev_indices:
                lost = prev.get("lost_count", 0) + 1
                if lost <= 6:
                    prev_copy = dict(prev)
                    prev_copy["lost_count"] = lost
                    new_prev_persons.append(prev_copy)
                    active_pids.add(prev["pid"])

        self._prev_persons = new_prev_persons

        self._cleanup_stale(active_pids)

        if not falls:
            self._is_fall_confirmed = False

        return falls

    # ------------------------------------------------------------------
    # Ve annotation
    # ------------------------------------------------------------------

    @property
    def last_detected_persons(self) -> list[dict]:
        """Danh sach tat ca nguoi da duoc xac thuc khung xuong trong frame gan nhat."""
        return self._last_detected_persons

    def annotate_frame(self, frame: Any, falls: list[dict], timestamp: float | None = None) -> Any:
        fall_bboxes = {tuple(f["bbox"]) for f in falls}

        # Visual persistence: chỉ dùng trong khoảng cực ngắn (~0.08s, tối đa 2 frame off-slot của FixedSlotScheduler)
        # Tuyệt đối không lưu vết 3.0s gây đứng hình / đóng băng khung xương
        persons_to_draw = self._last_detected_persons
        now = time.time() if timestamp is None else timestamp
        if not persons_to_draw:
            if (now - getattr(self, "_last_detected_time", 0.0) < 0.08):
                persons_to_draw = getattr(self, "_cached_detected_persons", [])
            else:
                self._cached_detected_persons = []
        else:
            self._cached_detected_persons = list(persons_to_draw)
            self._last_detected_time = now

        frame_h = frame.shape[0] if hasattr(frame, "shape") else 720

        for person in persons_to_draw:
            status = person["status"]
            kpts = person["keypoints"]
            angle = person["angle"]
            x1, y1, x2, y2 = person["bbox"]
            pid = person.get("pid", 0)
            fall_dur = person.get("fall_duration", 0.0)

            is_immobile = (status == "fall_immobile") or (status == "fall" and fall_dur >= 5.0)

            if is_immobile:
                # 4. Ngã bất động quá 5s -> MÀU ĐEN (Black)
                bone_color = (0, 0, 0)
                joint_color = (0, 0, 0)
                core_color = (255, 255, 255)
                has_white_halo = True
                status_text = f"NGA BAT DONG (>5s) [{fall_dur:.1f}s] P#{pid}"
            elif status in ("fall", "fall_candidate"):
                # 3. Tư thế ngã (< 5s) -> MÀU ĐỎ (Red)
                bone_color = (0, 0, 255)
                joint_color = (50, 50, 255)
                core_color = (255, 255, 255)
                has_white_halo = False
                dur_str = f" [{fall_dur:.1f}s]" if fall_dur > 0 else ""
                status_text = f"NGA!{dur_str} P#{pid}"
            elif status == "bending":
                # 2. Tư thế cúi nghiêng -> MÀU CAM (Orange)
                bone_color = (0, 140, 255)
                joint_color = (30, 165, 255)
                core_color = (255, 255, 255)
                has_white_halo = False
                status_text = f"Cui nghieng ({angle:.0f}deg) P#{pid}"
            else:
                # 1. Tư thế bình thường -> MÀU VÀNG (Yellow)
                bone_color = (0, 220, 255)
                joint_color = (0, 240, 255)
                core_color = (255, 255, 255)
                has_white_halo = False
                status_text = f"P#{pid}"

            # Ve duong xuong (Bones)
            for p1_idx, p2_idx in SKELETON_PAIRS:
                p1 = kpts[p1_idx]
                p2 = kpts[p2_idx]
                if p1[2] >= KPT_CONF_THRESH and p2[2] >= KPT_CONF_THRESH:
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    if has_white_halo:
                        # Halo trang 4px de xuong den noi bat tuyet doi tren quan ao/nen toi mau
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            # Ve cac khop (Joints)
            for pt in kpts:
                if pt[2] >= KPT_CONF_THRESH:
                    px, py = int(pt[0]), int(pt[1])
                    if has_white_halo:
                        # Halo trang cho khop den
                        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 3, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (px, py), 3, joint_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 1, core_color, -1, cv2.LINE_AA)

            # Hien thi badge trang thai
            if is_immobile:
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                tag_y = max(min(y2 + th + 8, frame_h - 8), 18)
                cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), (255, 255, 255), 1)
                cv2.putText(
                    frame, status_text,
                    (x1 + 4, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
                )
            elif status != "normal":
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                tag_y = max(min(y2 + th + 8, frame_h - 8), 18)
                cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), (20, 20, 20), -1)
                cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), bone_color, 1)
                cv2.putText(
                    frame, status_text,
                    (x1 + 4, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
                )
            else:
                # Dang binh thuong (Vang): badge gon gang o goc tren
                mini_tag = f"P#{pid}"
                (tw, th), _ = cv2.getTextSize(mini_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                tag_y = max(y1 - 6, th + 4)
                cv2.rectangle(frame, (x1, tag_y - th - 3), (x1 + tw + 6, tag_y + 3), (20, 20, 20), -1)
                cv2.rectangle(frame, (x1, tag_y - th - 3), (x1 + tw + 6, tag_y + 3), bone_color, 1)
                cv2.putText(
                    frame, mini_tag,
                    (x1 + 3, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA,
                )

        for f in falls:
            x1, y1, x2, y2 = f["bbox"]
            conf = f["confidence"]
            angle = f.get("angle", 0.0)
            paths = f.get("paths", {})
            f_status = f.get("status", "fall")
            f_dur = f.get("fall_duration", 0.0)
            angle_str = f" ({angle:.0f}deg)" if angle > 0 else ""

            if f_status == "fall_immobile" or f_dur >= 5.0:
                label = f"CAP CUU: NGA BAT DONG (>5s)! {conf:.2f}{angle_str}"
                path_label = f"Pose:{paths.get('pose', 0)} Grd:{paths.get('grd', 0)} Vel:{paths.get('vel', 0)} Mot:{paths.get('motion', 0)}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(
                    frame,
                    (x1, max(y1 - th - 24, 0)),
                    (x1 + tw + 8, y1),
                    (0, 0, 0), -1,
                )
                cv2.rectangle(
                    frame,
                    (x1, max(y1 - th - 24, 0)),
                    (x1 + tw + 8, y1),
                    (255, 255, 255), 2,
                )
                cv2.putText(
                    frame, label,
                    (x1 + 4, max(y1 - 18, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, path_label,
                    (x1 + 4, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
                )
            else:
                label = f"CANH BAO: NGA! {conf:.2f}{angle_str}"
                path_label = f"Pose:{paths.get('pose', 0)} Grd:{paths.get('grd', 0)} Vel:{paths.get('vel', 0)} Mot:{paths.get('motion', 0)}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(
                    frame,
                    (x1, max(y1 - th - 24, 0)),
                    (x1 + tw + 6, y1),
                    (0, 0, 255), -1,
                )
                cv2.putText(
                    frame, label,
                    (x1 + 3, max(y1 - 18, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, path_label,
                    (x1 + 3, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
                )

        return frame
