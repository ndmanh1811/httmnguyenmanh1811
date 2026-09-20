"""
fire_detector.py - YOLO-based Fire & Smoke Detection Module
Architectural Split:
  1. FireSmokeModel (Stateless / Shared):
     - Pure YOLO inference
     - Robust normalized class name discovery from model.names
     - Stateless frame annotation
  2. FireSmokeStreamAnalyzer (Stateful / Per-Stream):
     - Isolated spatial-temporal association tracks
     - Normalized soft additive scoring for Fire vs Smoke
     - No cross-stream state contamination
"""

import math
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _calculate_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(0.0, min(1.0, iou))


def _center_distance_norm(boxA: list[float], boxB: list[float], img_w: float, img_h: float) -> float:
    """Compute normalized Euclidean distance between centers of two boxes."""
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0

    diag = math.sqrt(img_w ** 2 + img_h ** 2) if (img_w > 0 and img_h > 0) else 1000.0
    dist = math.sqrt((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2)
    return dist / diag


class FireSmokeModel:
    """
    Stateless, shared model wrapper for YOLO Fire & Smoke inference.
    Can be safely called concurrently across multiple camera threads or video uploads.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ):
        if weights_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_dir, "models_dir", "fire_smoke_yolov8n.pt")
            weights_path = default_path if os.path.exists(default_path) else "fire_smoke_yolov8n.pt"

        self.weights_path = weights_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inference resolution: 960 bắt lửa/khói xa tốt hơn hẳn 640 trên camera góc rộng.
        # Override bằng biến môi trường FIRE_SMOKE_IMGSZ hoặc tham số imgsz.
        self.imgsz = int(imgsz or os.environ.get("FIRE_SMOKE_IMGSZ", "960"))

        # Serialized inference lock to prevent Ultralytics GPU context race conditions across threads
        self._inference_lock = threading.Lock()

        print(f"[FireSmokeModel] Loading weights: {self.weights_path} on {self.device}")
        self.model = YOLO(self.weights_path)

        # Dynamic normalized class discovery (case-insensitive, stripped)
        self.fire_class_ids: list[int] = []
        self.smoke_class_ids: list[int] = []
        for cls_id, name in self.model.names.items():
            norm_name = str(name).strip().lower()
            if "fire" in norm_name or "flame" in norm_name:
                self.fire_class_ids.append(int(cls_id))
            elif "smoke" in norm_name:
                self.smoke_class_ids.append(int(cls_id))

        if not self.fire_class_ids and not self.smoke_class_ids:
            raise ValueError(
                f"[FireSmokeModel] Could not find 'fire' or 'smoke' in model.names: {self.model.names}"
            )

        print(
            f"[FireSmokeModel] Discovered normalized classes - "
            f"Fire IDs: {self.fire_class_ids}, Smoke IDs: {self.smoke_class_ids} from {self.model.names}"
        )

    def predict_candidates(
        self,
        frame: np.ndarray,
        fire_conf: float = 0.35,
        smoke_conf: float = 0.40,
    ) -> list[dict]:
        """
        Pure forward inference with thread-safe lock serialization.
        Returns candidates without saving internal state.
        """
        with self._inference_lock:
            results = self.model(frame, device=self.device, imgsz=self.imgsz, verbose=False)[0]

        candidates = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]

            item_type = None
            min_conf = 0.50
            if cls_id in self.fire_class_ids:
                item_type = "fire"
                min_conf = fire_conf
            elif cls_id in self.smoke_class_ids:
                item_type = "smoke"
                min_conf = smoke_conf

            if item_type is not None and conf >= min_conf:
                candidates.append({
                    "type": item_type,
                    "confidence": conf,
                    "bbox": xyxy,
                    "cls_id": cls_id,
                })

        return candidates

    def annotate_frame(
        self,
        frame: np.ndarray,
        detected_hazards: list[dict],
        exclusion_zones: Optional[list[dict]] = None,
    ) -> np.ndarray:
        """
        Pure rendering function to draw bounding boxes, exclusion zones, and banners.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw active exclusion zones (if any)
        if exclusion_zones:
            for zone in exclusion_zones:
                if not zone.get("is_active", True):
                    continue
                raw_pts = zone.get("polygon_points") or zone.get("polygon") or []
                if len(raw_pts) < 3:
                    continue

                poly_pts = np.array(
                    [[int(p[0] * w), int(p[1] * h)] for p in raw_pts],
                    dtype=np.int32,
                )

                # Subtle semi-transparent polygon fill
                zone_overlay = out.copy()
                cv2.fillPoly(zone_overlay, [poly_pts], (255, 200, 0))  # BGR Cyan-Amber
                cv2.addWeighted(zone_overlay, 0.15, out, 0.85, 0, out)

                # Border line
                cv2.polylines(out, [poly_pts], isClosed=True, color=(255, 220, 50), thickness=2)

                # Label tag at first point
                z_name = zone.get("name", "Vùng loại trừ")
                first_pt = poly_pts[0]
                label_text = f"[VÙNG BỎ QUA: {z_name}]"
                cv2.putText(
                    out,
                    label_text,
                    (max(10, first_pt[0]), max(20, first_pt[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 230, 80),
                    1,
                    cv2.LINE_AA,
                )

        if not detected_hazards:
            return out

        has_breakout = any(d.get("is_breakout") for d in detected_hazards)
        has_fire = any(d.get("type") == "fire" for d in detected_hazards)
        has_smoke = any(d.get("type") == "smoke" for d in detected_hazards)

        for d in detected_hazards:
            box = d.get("bbox", [])
            if len(box) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            d_type = d.get("type", "fire")
            conf = d.get("confidence", 0.0)
            is_breakout = d.get("is_breakout", False)

            if is_breakout:
                b_zone = d.get("breakout_zone", "Vùng loại trừ")
                b_ratio = d.get("breakout_ratio", 0.0)
                color = (0, 0, 255)  # Bright Red
                prefix = "KHOI LAN RA NGOAI" if d_type == "smoke" else "CHAY LAN RA NGOAI"
                label = f"{prefix} [{b_zone}] ({b_ratio:.0%})"
            elif d_type == "fire":
                color = (0, 69, 255)  # Vibrant Orange-Red (BGR)
                label = f"LUA / FIRE {conf:.0%}"
            else:
                color = (215, 190, 140)  # Smoke Slate-Blue (BGR)
                upi = d.get("components", {}).get("upi", 0.5)
                if upi >= 0.65:
                    label = f"KHOI / SMOKE {conf:.0%} [UPI {upi:.0%} ^]"
                else:
                    label = f"KHOI / SMOKE {conf:.0%}"

            # 1. Base bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # 2. Corner brackets for modern AI HUD style
            d_len = max(10, min(24, int(min(x2 - x1, y2 - y1) * 0.15)))
            cv2.line(out, (x1, y1), (x1 + d_len, y1), color, 4)
            cv2.line(out, (x1, y1), (x1, y1 + d_len), color, 4)
            cv2.line(out, (x2, y1), (x2 - d_len, y1), color, 4)
            cv2.line(out, (x2, y1), (x2, y1 + d_len), color, 4)
            cv2.line(out, (x1, y2), (x1 + d_len, y2), color, 4)
            cv2.line(out, (x1, y2), (x1, y2 - d_len), color, 4)
            cv2.line(out, (x2, y2), (x2 - d_len, y2), color, 4)
            cv2.line(out, (x2, y2), (x2, y2 - d_len), color, 4)

            # 3. Label tag
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if has_breakout:
            if any(d.get("is_breakout") and d.get("type") == "fire" for d in detected_hazards) or has_fire:
                banner_text = "NGUY HIEM: PHAT HIEN CHAY LAN RA NGOAI VUNG LOAI TRU!"
                banner_color = (0, 0, 255)
            else:
                banner_text = "CANH BAO: PHAT HIEN KHOI LAN RA NGOAI VUNG LOAI TRU!"
                banner_color = (0, 100, 255)
        elif has_fire:
            banner_text = "NGUY HIEM: PHAT HIEN HOA HOAN (CHAY)!"
            banner_color = (0, 0, 230)
        elif has_smoke:
            banner_text = "CANH BAO: PHAT HIEN KHOI (NGUY CO CHAY)!"
            banner_color = (0, 140, 255)
        else:
            banner_text = "CANH BAO NGUY HIEM"
            banner_color = (0, 0, 255)

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), banner_color, -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.putText(
            out,
            banner_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return out


def extract_fire_smoke_contour(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float],
    cand_type: str,
) -> tuple[Optional[np.ndarray], float]:
    """
    Extracts the actual morphological contour and pixel area of fire or smoke inside a candidate bounding box.
    Returns: (contour_pts, area) where contour_pts is Nx2 in full frame coordinates.
    """
    h_f, w_f = frame.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w_f, bx2), min(h_f, by2)
    bw = bx2 - bx1
    bh = by2 - by1
    if bw < 8 or bh < 8:
        return None, 0.0

    patch = frame[by1:by2, bx1:bx2]

    if cand_type == "fire":
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 110]), np.array([38, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 50, 110]), np.array([180, 255, 255]))
        mask_hsv = cv2.bitwise_or(mask1, mask2)

        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        mask_ycrcb = ((y >= 100) & (cr >= 135) & (cr >= cb)).astype(np.uint8) * 255
        mask_target = cv2.bitwise_or(mask_hsv, mask_ycrcb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_target, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
    else:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        cond_dark = (v < 70)
        cond_gray = (s < 75) & (v >= 70) & (v < 220)
        mask_smoke = np.zeros(patch.shape[:2], dtype=np.uint8)
        mask_smoke[cond_dark | cond_gray] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_clean = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(50.0, 0.05 * (bw * bh))
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid:
        rect_pts = np.array([
            [bx1, by1],
            [bx2, by1],
            [bx2, by2],
            [bx1, by2]
        ], dtype=np.int32)
        return rect_pts, float(bw * bh)

    largest = max(valid, key=cv2.contourArea)
    epsilon = 0.015 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 3:
        approx = cv2.convexHull(largest)

    contour_full = approx.reshape(-1, 2) + [bx1, by1]
    return contour_full, float(cv2.contourArea(largest))


def compute_plume_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    bbox: list[float],
) -> float:
    """
    Measures vertical motion direction of a smoke candidate patch across consecutive frames.
    Returns: upi (Upward Plume Index) in [0.0, 1.0], where >= 0.70 means strong upward buoyant motion.
    """
    h_f, w_f = curr_gray.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w_f, bx2), min(h_f, by2)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 10 or bh < 10:
        return 0.5

    p1 = prev_gray[by1:by2, bx1:bx2]
    p2 = curr_gray[by1:by2, bx1:bx2]

    corners = cv2.goodFeaturesToTrack(p1, maxCorners=35, qualityLevel=0.02, minDistance=8)
    if corners is None or len(corners) < 4:
        return 0.5

    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        p1, p2, corners, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    if next_corners is None or status is None:
        return 0.5

    valid = (status.flatten() == 1)
    if np.sum(valid) < 4:
        return 0.5

    c0 = corners[valid]
    c1 = next_corners[valid]

    dx = c1[:, 0, 0] - c0[:, 0, 0]
    dy = c1[:, 0, 1] - c0[:, 0, 1]

    mean_dy = float(np.mean(dy))
    mean_dx = float(np.mean(dx))
    speed = math.sqrt(mean_dx ** 2 + mean_dy ** 2)

    if speed < 0.15:
        return 0.35  # Stationary object

    upward_ratio = float(np.mean(dy < -0.1))
    vel_upi = max(0.0, min(1.0, -mean_dy / max(0.5, speed)))
    upi = 0.5 * upward_ratio + 0.5 * vel_upi
    return float(np.clip(upi, 0.0, 1.0))


def measure_region_intensity(
    frame_gray: np.ndarray,
    bbox: list[float],
    contour: Optional[np.ndarray] = None,
) -> float:
    """
    Mean brightness inside the candidate contour (falls back to bbox).
    Sampled once per frame into the track history — input for flicker FFT.
    Returns -1.0 when the region is degenerate.
    """
    h_f, w_f = frame_gray.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w_f, bx2), min(h_f, by2)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 4 or bh < 4:
        return -1.0

    patch = frame_gray[by1:by2, bx1:bx2]
    if contour is not None and len(contour) >= 3:
        mask = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(contour, dtype=np.int32) - [bx1, by1]], 255)
        vals = patch[mask > 0]
        if vals.size == 0:
            return -1.0
        return float(vals.mean())
    return float(patch.mean())


def compute_flicker_score(intensity_history: list[tuple[float, float]]) -> float:
    """
    Flame-flicker detector: FFT of the tracked region brightness over time.
    Real flames oscillate (~1.5-12 Hz band); static look-alikes (lamps, red wall
    patches, sun-lit surfaces, printed images) are near-DC. Returns [0,1];
    0.5 = neutral while history is too short to judge.
    """
    hist = [(t, v) for t, v in intensity_history if v >= 0]
    if len(hist) < 8:
        return 0.5

    ts = np.array([t for t, _ in hist], dtype=np.float64)
    vals = np.array([v for _, v in hist], dtype=np.float64)
    if len(ts) < 3:
        return 0.5
    dt = float(np.median(np.diff(ts)))
    if dt <= 1e-3:
        return 0.5

    x = vals - vals.mean()
    if float(np.std(x)) < 1e-3:
        return 0.05  # constant brightness over 8+ frames -> definitely not a flame

    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    total = float(spec.sum()) + 1e-9
    band = (freqs >= 1.5) & (freqs <= 12.0)
    ratio = float(spec[band].sum() / total)
    return float(np.clip(ratio * 1.8, 0.0, 1.0))


def compute_smoke_texture(
    frame: np.ndarray,
    bbox: list[float],
    contour: Optional[np.ndarray] = None,
) -> float:
    """
    Smoke-likeness from texture: real smoke has soft, diffuse boundaries and a
    smooth interior; solid gray look-alikes (walls, machinery, fabric, hair)
    have sharp edges. Combines contour irregularity and boundary softness.
    Returns [0,1]; 0.5 = neutral when nothing can be measured.
    """
    h_f, w_f = frame.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w_f, bx2), min(h_f, by2)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 8 or bh < 8:
        return 0.5

    patch_gray = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)

    mask = np.zeros(patch_gray.shape[:2], dtype=np.uint8)
    if contour is not None and len(contour) >= 3:
        cnt = np.array(contour, dtype=np.int32) - [bx1, by1]
        cv2.fillPoly(mask, [cnt], 255)
        cnt_area = float(cv2.countNonZero(mask))
        if cnt_area > 50:
            per = float(cv2.arcLength(cnt, True))
            compact = per / math.sqrt(cnt_area)  # circle ~3.5, plumes > 6
            fuzziness = float(np.clip((compact - 4.0) / 6.0, 0.0, 1.0))
        else:
            fuzziness = 0.5
    else:
        fuzziness = 0.5

    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    k = np.ones((5, 5), np.uint8)
    band = cv2.dilate(mask, k) - cv2.erode(mask, k)
    band_vals = grad[band > 0]
    if band_vals.size < 30:
        softness = 0.5
    else:
        soft_frac = float((band_vals < 25.0).mean())
        hard_frac = float((band_vals > 90.0).mean())
        softness = float(np.clip(0.5 + 0.6 * (soft_frac - 0.40) - 0.8 * hard_frac, 0.0, 1.0))

    return float(np.clip(0.5 * fuzziness + 0.5 * softness, 0.0, 1.0))


def check_exclusion_breakout(
    bbox: list[float],
    polygon_normalized: list[list[float]],
    img_w: int,
    img_h: int,
    outside_threshold: float = 0.40,
    contour: Optional[np.ndarray] = None,
) -> tuple[bool, float, bool]:
    """
    Evaluates whether a candidate (using its actual contour if provided, else bbox)
    intersects an exclusion polygon and whether it has broken out.
    Returns: (is_ignored, outside_ratio, is_breakout)
      - is_ignored=True: Candidate is comfortably inside the exclusion zone -> ignore.
      - is_breakout=True: Candidate originates in or substantially covers the zone
                          (inside_ratio >= 0.15 or poly_covered_ratio >= 0.30)
                          AND has spilled > outside_threshold (40%) outside -> Genuine Breakout!
      - Neither (both False): External object outside the zone (< 15% intersection and < 30% coverage).
    """
    bx1, by1, bx2, by2 = map(int, bbox)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(img_w, bx2), min(img_h, by2)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    poly_pts = np.array(
        [[int(p[0] * img_w), int(p[1] * img_h)] for p in polygon_normalized],
        dtype=np.int32,
    )
    poly_area = max(1.0, float(cv2.contourArea(poly_pts)))

    poly_shifted = poly_pts - [bx1, by1]
    mask_poly = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poly_shifted], 255)

    if contour is not None and len(contour) >= 3:
        cnt_shifted = np.array(contour, dtype=np.int32) - [bx1, by1]
        mask_obj = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask_obj, [cnt_shifted], 255)
        obj_area = max(1.0, float(cv2.countNonZero(mask_obj)))
        inter_area = cv2.countNonZero(cv2.bitwise_and(mask_poly, mask_obj))
    else:
        obj_area = float(bw * bh)
        inter_area = cv2.countNonZero(mask_poly)

    if inter_area <= 0:
        return False, 1.0, False

    inside_ratio = float(inter_area) / float(obj_area)
    outside_ratio = 1.0 - inside_ratio
    poly_covered_ratio = float(inter_area) / poly_area

    if outside_ratio <= outside_threshold:
        return True, outside_ratio, False

    if inside_ratio >= 0.15 or poly_covered_ratio >= 0.30:
        return False, outside_ratio, True

    return False, outside_ratio, False


class FireScorer:
    """
    Evaluates fire candidates based on:
      - Confidence: Model prediction confidence (EMA-smoothed in the tracker)
      - Persistence: Temporal continuity over confirmation window
      - Overlap: Spatial IoU consistency between frames
      - AspectStability: Ratio stability of bounding box (distinguishes from fast light sweeps)
      - Flicker: FFT band-energy of region brightness (~1.5-12 Hz) — real flames flicker,
        static look-alikes (lamp, red wall, sun reflection) do not
      - Static Object Rejection: Penalizes static objects without flame flickering
    Weights: 0.30 * conf + 0.30 * persist + 0.10 * overlap + 0.15 * aspect_stability + 0.15 * flicker
    """

    @staticmethod
    def calculate(track: dict, current_ts: float, persistence_window: float = 0.8) -> tuple[float, dict]:
        conf = min(1.0, max(0.0, float(track["conf"])))
        # Persistence đo thời gian PHÁT HIỆN liên tục (last_seen), không phải thời gian
        # kể từ khi sinh track — nếu không, track stale vẫn tăng điểm theo đồng hồ
        # trong 1.2s trước khi bị dọn và tiếp tục giữ alert sống.
        duration = max(0.0, float(track.get("last_seen", current_ts)) - track["first_seen"])
        persist_ratio = min(1.0, duration / max(0.1, persistence_window))
        overlap = min(1.0, max(0.0, float(track.get("last_iou", 0.0))))

        hist = track.get("history", [])
        if len(hist) >= 2:
            ratios = []
            for _, b, _ in hist[-4:]:
                bw = max(1.0, b[2] - b[0])
                bh = max(1.0, b[3] - b[1])
                ratios.append(bw / bh)
            mean_r = sum(ratios) / len(ratios)
            var_r = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
            aspect_stability = max(0.0, min(1.0, 1.0 - var_r * 2.0))
        else:
            aspect_stability = 0.5

        flicker = compute_flicker_score(track.get("intensity_history", []))

        score = (
            0.30 * conf
            + 0.30 * persist_ratio
            + 0.10 * overlap
            + 0.15 * aspect_stability
            + 0.15 * flicker
        )

        # Static Object Rejection (geometry): real flames change shape and area.
        # A static look-alike (lamp, wall patch, print) holds bbox geometry steady
        # AND shows no flicker evidence. Loosened thresholds (overlap/aspect) so it
        # still fires after the bbox settles from a camera pan.
        if len(hist) >= 4 and conf < 0.70 and overlap > 0.60 and aspect_stability > 0.30 and flicker <= 0.55:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for _, b, _ in hist[-6:]]
            mean_area = sum(areas) / len(areas)
            rel_area_std = (sum((a - mean_area) ** 2 for a in areas) / len(areas)) ** 0.5 / max(1.0, mean_area)
            if rel_area_std < 0.035:
                score = max(0.20, score - 0.28)

        # Brightness-constancy Rejection: real flames vary strongly in emitted
        # brightness frame to frame (flicker). A static glare/reflection through
        # leaves or off glass holds near-constant mean brightness — kill it even
        # when the bbox jitters (which defeats the geometry test above).
        ih = [v for _, v in track.get("intensity_history", []) if v >= 0]
        if len(ih) >= 4 and conf < 0.70:
            arr = np.array(ih, dtype=np.float64)
            rel_std = float(arr.std() / max(1.0, arr.mean()))
            if rel_std < 0.12:
                score = max(0.20, score - 0.28)

        components = {
            "conf": conf,
            "persistence": persist_ratio,
            "overlap": overlap,
            "aspect_stability": aspect_stability,
            "flicker": flicker,
            "duration": duration,
        }
        return score, components


class SmokeScorer:
    """
    Evaluates smoke candidates based on:
      - Confidence: Model prediction confidence (EMA-smoothed in the tracker)
      - Persistence: Temporal continuity over confirmation window
      - Overlap: Soft spatial continuity
      - AreaTrend: Linear regression growth slope over recent N<=8 frames (Soft evidence, no veto!)
      - UPI: Upward Plume Index from optical flow (buoyancy)
      - Texture: boundary softness + contour irregularity — real smoke is diffuse,
        solid gray look-alikes (walls, hair, fabric) have sharp edges
    Weights: 0.20*conf + 0.20*persist + 0.10*overlap + 0.15*area_trend + 0.15*upi + 0.20*texture
    """

    @staticmethod
    def calculate_area_trend(history: list[tuple[float, list[float], float]]) -> float:
        """
        Computes soft area trend score in [0.25, 1.0] using linear regression slope over up to 8 recent frames.
        Does NOT veto or return 0 if slope <= 0, because smoke area can fluctuate or expand non-monotonically.
        """
        if len(history) < 2:
            return 0.5  # neutral baseline

        recent = history[-8:]
        ts_list = []
        area_list = []
        t0 = recent[0][0]

        for t, b, _ in recent:
            ts_list.append(t - t0)
            area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            area_list.append(area)

        mean_t = sum(ts_list) / len(ts_list)
        mean_a = sum(area_list) / len(area_list)

        denom = sum((t - mean_t) ** 2 for t in ts_list)
        if denom < 1e-6:
            return 0.5

        num = sum((t - mean_t) * (a - mean_a) for t, a in zip(ts_list, area_list))
        raw_slope = num / denom
        rel_slope = raw_slope / max(1.0, mean_a)

        # Soft mapping:
        # If expanding (rel_slope > 0): maps smoothly from 0.50 -> 1.00
        # If shrinking / stationary (rel_slope <= 0): maps smoothly from 0.25 -> 0.50 (Never 0, never hard veto!)
        if rel_slope > 0:
            area_score = min(1.0, 0.50 + 0.50 * math.tanh(rel_slope * 1.5))
        else:
            area_score = max(0.25, 0.50 + 0.25 * math.tanh(rel_slope * 1.5))

        return area_score

    @staticmethod
    def calculate(track: dict, current_ts: float, persistence_window: float = 2.0) -> tuple[float, dict]:
        conf = min(1.0, max(0.0, float(track["conf"])))
        # Như FireScorer: persistence theo last_seen để track stale không cộng điểm.
        duration = max(0.0, float(track.get("last_seen", current_ts)) - track["first_seen"])
        persist_ratio = min(1.0, duration / max(0.1, persistence_window))
        overlap = min(1.0, max(0.0, float(track.get("last_iou", 0.0))))
        area_trend = SmokeScorer.calculate_area_trend(track.get("history", []))
        upi = float(track.get("upi", 0.50))
        texture = float(track.get("texture", 0.50))

        score = (
            0.20 * conf
            + 0.20 * persist_ratio
            + 0.10 * overlap
            + 0.15 * area_trend
            + 0.15 * upi
            + 0.20 * texture
        )

        # Stationary gray-object penalty: real smoke drifts upward (UPI) or expands
        # (area trend) and has diffuse texture. A motionless patch with weak texture
        # evidence (hair, shadow, dark furniture) is likely not smoke — soft penalty.
        if upi < 0.40 and area_trend < 0.45 and texture < 0.55:
            score = max(0.20, score - 0.10)

        # Sharp solid-object penalty: real smoke is diffuse and semi-transparent.
        # Hard, sharp edges (clothing, human bodies, machines, walls, fallback rectangles)
        # have very low texture softness (< 0.35).
        if texture < 0.35:
            score = max(0.20, score - 0.15)

        components = {
            "conf": conf,
            "persistence": persist_ratio,
            "overlap": overlap,
            "area_trend": area_trend,
            "upi": upi,
            "texture": texture,
            "duration": duration,
        }
        return score, components


class FireSmokeStreamAnalyzer:
    """
    Stateful, per-stream analyzer.
    Manages isolated temporal association tracks and decoupled normalized soft scoring.
    """

    def __init__(
        self,
        model: Optional[FireSmokeModel] = None,
        weights_path: Optional[str] = None,
        fire_conf: float = 0.35,
        smoke_conf: float = 0.40,
        persistence_sec: Optional[dict[str, float] | float] = None,
        exclusion_zones: Optional[list[dict]] = None,
    ):
        if model is None:
            self.model = FireSmokeModel(weights_path=weights_path)
        else:
            self.model = model

        self.fire_conf = fire_conf
        self.smoke_conf = smoke_conf
        self.exclusion_zones = exclusion_zones or []

        if isinstance(persistence_sec, dict):
            self.persistence_sec = persistence_sec
        elif isinstance(persistence_sec, (int, float)):
            self.persistence_sec = {"fire": float(persistence_sec), "smoke": float(persistence_sec)}
        else:
            self.persistence_sec = {"fire": 0.8, "smoke": 2.0}

        # Thời gian score phải nằm dưới ngưỡng exit trước khi một track verified
        # bị hủy — chống nhấp nháy alert khi score dao động quanh ngưỡng.
        self.hysteresis_exit_sec = 1.5

        self.fire_scorer = FireScorer()
        self.smoke_scorer = SmokeScorer()

        # Stream-isolated candidate tracks
        self._tracks: dict[int, dict] = {}
        self._next_track_id: int = 1
        self._last_cleanup_time: float = 0.0
        self._last_gray_frame: Optional[np.ndarray] = None

    def set_exclusion_zones(self, zones: list[dict]):
        """Dynamically update exclusion zones for this stream."""
        self.exclusion_zones = zones or []

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        person_boxes: Optional[list] = None,
    ) -> list[dict]:
        """
        Runs candidate detection on shared model, updates isolated stream tracks,
        and evaluates decoupled normalized soft scores with contour refinement and optical flow.
        """
        if timestamp is None:
            timestamp = time.time()

        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = self.model.predict_candidates(frame, self.fire_conf, self.smoke_conf)

        # Exclusion zone and breakout evaluation
        active_candidates = []
        for cand in candidates:
            c_type = cand["type"]
            c_bbox = cand["bbox"]
            c_conf = cand["confidence"]

            # Sub-box Morphological / Color Contour Extraction
            c_contour, c_area = extract_fire_smoke_contour(frame, c_bbox, c_type)
            cand["contour"] = c_contour
            cand["contour_area"] = c_area
            cand["intensity"] = measure_region_intensity(curr_gray, c_bbox, c_contour)

            # Person Head/Face False Alarm Suppression
            if c_type == "fire" and person_boxes and c_conf < 0.85:
                c_area_eff = max(1.0, c_area if c_area > 0 else (c_bbox[2] - c_bbox[0]) * (c_bbox[3] - c_bbox[1]))
                is_person_head = False
                for pbox in person_boxes:
                    px1, py1, px2, py2 = pbox
                    # Head/face region is upper 40% of person box
                    head_y2 = py1 + (py2 - py1) * 0.40
                    ix1 = max(c_bbox[0], px1)
                    iy1 = max(c_bbox[1], py1)
                    ix2 = min(c_bbox[2], px2)
                    iy2 = min(c_bbox[3], head_y2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if (inter / c_area_eff) >= 0.25:
                            is_person_head = True
                            break
                if is_person_head:
                    continue  # Discard candidate on person's face/head

            is_ignored = False
            is_breakout = False
            breakout_zone = None
            breakout_ratio = 0.0

            for zone in self.exclusion_zones:
                if not zone.get("is_active", True):
                    continue
                raw_pts = zone.get("polygon_points") or zone.get("polygon") or []
                if len(raw_pts) < 3:
                    continue

                ignored, outside_ratio, is_bk = check_exclusion_breakout(
                    bbox=c_bbox,
                    polygon_normalized=raw_pts,
                    img_w=w,
                    img_h=h,
                    outside_threshold=0.40,
                    contour=c_contour,
                )

                if ignored:
                    is_ignored = True
                    break
                elif is_bk:
                    is_breakout = True
                    breakout_zone = zone.get("name", "Vùng loại trừ")
                    breakout_ratio = outside_ratio

            if is_ignored:
                continue

            cand["is_breakout"] = is_breakout
            cand["breakout_zone"] = breakout_zone
            cand["breakout_ratio"] = breakout_ratio
            active_candidates.append(cand)

        matched_track_ids = set()

        for cand in active_candidates:
            c_type = cand["type"]
            c_bbox = cand["bbox"]
            c_conf = cand["confidence"]

            best_match_id = None
            best_score = 0.0
            best_iou = 0.0

            for tid, track in self._tracks.items():
                if track["type"] != c_type:
                    continue

                t_bbox = track["bbox"]
                iou = _calculate_iou(c_bbox, t_bbox)
                center_dist = _center_distance_norm(c_bbox, t_bbox, w, h)

                if c_type == "fire":
                    if iou >= 0.15 or center_dist < 0.20:
                        match_metric = iou * 0.7 + (1.0 - min(1.0, center_dist)) * 0.3
                        if match_metric > best_score:
                            best_score = match_metric
                            best_match_id = tid
                            best_iou = iou
                elif c_type == "smoke":
                    # Non-rigid soft matching
                    if iou >= 0.05 or center_dist < 0.35:
                        match_metric = iou * 0.4 + (1.0 - min(1.0, center_dist)) * 0.6
                        if match_metric > best_score:
                            best_score = match_metric
                            best_match_id = tid
                            best_iou = iou

            if best_match_id is not None and best_match_id not in matched_track_ids:
                matched_track_ids.add(best_match_id)
                track = self._tracks[best_match_id]
                track["bbox"] = c_bbox
                track["contour"] = cand.get("contour")
                # EMA confidence: phản ánh trạng thái hiện tại, tự hồi phục khi model
                # ngừng tin vào region (thay vì max() niêm foreground conf cao mãi mãi).
                track["conf"] = 0.65 * float(track["conf"]) + 0.35 * c_conf
                track["last_seen"] = timestamp
                track["hits"] += 1
                track["last_iou"] = best_iou
                track["is_breakout"] = cand.get("is_breakout", False)
                track["breakout_zone"] = cand.get("breakout_zone")
                track["breakout_ratio"] = cand.get("breakout_ratio", 0.0)

                # Flicker input: region brightness each frame
                if cand.get("intensity", -1.0) >= 0:
                    track.setdefault("intensity_history", []).append((timestamp, cand["intensity"]))
                    if len(track["intensity_history"]) > 32:
                        track["intensity_history"].pop(0)

                # Upward Plume Optical Flow for Smoke
                if c_type == "smoke" and self._last_gray_frame is not None:
                    upi = compute_plume_optical_flow(self._last_gray_frame, curr_gray, c_bbox)
                    track["upi"] = 0.65 * track.get("upi", upi) + 0.35 * upi

                # Smoke texture re-evaluation (contour evolves as plume grows)
                if c_type == "smoke":
                    new_tex = compute_smoke_texture(frame, c_bbox, cand.get("contour"))
                    track["texture"] = 0.60 * track.get("texture", new_tex) + 0.40 * new_tex

                track["history"].append((timestamp, c_bbox, c_conf))
                if len(track["history"]) > 20:
                    track["history"].pop(0)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                init_upi = 0.50
                if c_type == "smoke" and self._last_gray_frame is not None:
                    init_upi = compute_plume_optical_flow(self._last_gray_frame, curr_gray, c_bbox)
                init_texture = 0.50
                if c_type == "smoke":
                    init_texture = compute_smoke_texture(frame, c_bbox, cand.get("contour"))

                self._tracks[tid] = {
                    "id": tid,
                    "type": c_type,
                    "bbox": c_bbox,
                    "contour": cand.get("contour"),
                    "conf": c_conf,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "hits": 1,
                    "last_iou": 0.0,
                    "upi": init_upi,
                    "texture": init_texture,
                    "intensity_history": (
                        [(timestamp, cand["intensity"])] if cand.get("intensity", -1.0) >= 0 else []
                    ),
                    "is_breakout": cand.get("is_breakout", False),
                    "breakout_zone": cand.get("breakout_zone"),
                    "breakout_ratio": cand.get("breakout_ratio", 0.0),
                    "history": [(timestamp, c_bbox, c_conf)],
                }
                matched_track_ids.add(tid)

        # Update last gray frame for next optical flow step
        self._last_gray_frame = curr_gray

        # Cleanup stale tracks (> 1.2 seconds without detection)
        if timestamp - self._last_cleanup_time > 0.5:
            self._last_cleanup_time = timestamp
            stale_ids = [
                tid for tid, tr in self._tracks.items()
                if (timestamp - tr["last_seen"]) > 1.2
            ]
            for tid in stale_ids:
                del self._tracks[tid]

        # Decoupled soft scoring per class, với hysteresis chống nhấp nháy cảnh báo:
        # một track đã verified chỉ thoát khi score rơi xuống ngưỡng thấp hơn
        # (exit < enter) và duy trì liên tục >= HYSTERESIS_EXIT_SEC giây.
        verified = []
        for tid, track in self._tracks.items():
            duration = track["last_seen"] - track["first_seen"]
            hits = track["hits"]
            c_type = track["type"]

            if c_type == "fire":
                persistence_win = self.persistence_sec.get("fire", 0.8)
                score, components = self.fire_scorer.calculate(track, timestamp, persistence_win)
                enter_t, min_dur = 0.50, 0.3
            elif c_type == "smoke":
                persistence_win = self.persistence_sec.get("smoke", 2.0)
                score, components = self.smoke_scorer.calculate(track, timestamp, persistence_win)
                enter_t, min_dur = 0.45, 0.4
            else:
                continue

            exit_t = enter_t - 0.15
            if track.get("verified", False):
                if score < exit_t:
                    low_since = track.get("low_since")
                    if low_since is None:
                        track["low_since"] = timestamp
                    elif (timestamp - low_since) >= self.hysteresis_exit_sec:
                        track["verified"] = False
                        track.pop("low_since", None)
                else:
                    track.pop("low_since", None)
            else:
                track.pop("low_since", None)
                if score >= enter_t and duration >= min_dur and hits >= 2:
                    track["verified"] = True

            if track.get("verified", False):
                verified.append({
                    "track_id": tid,
                    "type": c_type,
                    "confidence": track["conf"],
                    "score": score,
                    "components": components,
                    "bbox": track["bbox"],
                    "contour": track.get("contour"),
                    "duration": duration,
                    "label": "Cháy" if c_type == "fire" else "Khói",
                    "is_breakout": track.get("is_breakout", False),
                    "breakout_zone": track.get("breakout_zone"),
                    "breakout_ratio": track.get("breakout_ratio", 0.0),
                })

        return verified

    def annotate_frame(
        self,
        frame: np.ndarray,
        detected_hazards: list[dict],
    ) -> np.ndarray:
        return self.model.annotate_frame(frame, detected_hazards, exclusion_zones=self.exclusion_zones)

    def reset(self):
        """Reset per-stream state."""
        self._tracks.clear()
        self._next_track_id = 1
        self._last_gray_frame = None
        self._last_cleanup_time = 0.0


# Compatibility alias: FireSmokeDetector maps to FireSmokeStreamAnalyzer
FireSmokeDetector = FireSmokeStreamAnalyzer
