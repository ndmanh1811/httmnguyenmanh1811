"""
detector.py
------------
Hệ thống giám sát PPE YOLO (YOLO Multi-Class PPE Detection System).
Hỗ trợ nhận diện đa lớp trang bị bảo hộ lao động: Mũ bảo hộ, Áo phản quang, Khẩu trang và Người.
Tích hợp ByteTrack, Two-Stage Verification chống báo động giả khói/vật thể.

Ưu tiên nạp model theo thứ tự:
    1. models_dir/best_hardhat.pt -> Model YOLO (YOLO26s / YOLOv8s) tự huấn luyện
    2. Dự phòng trực tuyến qua Hugging Face nếu thiếu file weights cục bộ.

Bộ dữ liệu chuẩn: Construction Site Safety (Roboflow)
    Lớp: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask, Person
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

import cv2
import imageio
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_WEIGHTS = os.path.join(BASE_DIR, "models_dir", "best_hardhat.pt")
FALLBACK_REPO_ID = "keremberke/yolov8n-hard-hat-detection"


# ---------------------------------------------------------------------------
# PPE classification mapping
# ---------------------------------------------------------------------------

_HELMET_SAFE = {"hardhat", "helmet"}
_HELMET_VIOLATION = {"no-hardhat", "no-helmet", "non-helmet", "head"}
_VEST_SAFE = {"safety vest", "vest"}
_VEST_VIOLATION = {"no-safety vest", "no-vest"}
_MASK_SAFE = {"mask"}
_MASK_VIOLATION = {"no-mask", "no mask"}

_SKIP_LABELS = {"person", "worker", "people", "gloves", "shoes", "boots",
                "truck", "bus", "van", "car", "excavator", "machinery",
                "ladder", "safety cone", "fire hydrant"}


def _classify_label(label: str) -> tuple[str, str] | None:
    low = label.lower().strip()

    if any(kw in low for kw in _SKIP_LABELS):
        return None

    if any(kw in low for kw in _HELMET_VIOLATION):
        return ("helmet", "violation")
    if any(kw in low for kw in _HELMET_SAFE):
        return ("helmet", "ok")

    if any(kw in low for kw in _VEST_VIOLATION):
        return ("vest", "violation")
    if any(kw in low for kw in _VEST_SAFE):
        return ("vest", "ok")

    if any(kw in low for kw in _MASK_VIOLATION):
        return ("mask", "violation")
    if any(kw in low for kw in _MASK_SAFE):
        return ("mask", "ok")

    return None


PPE_COLORS: dict[str, dict[str, tuple[int, int, int]]] = {
    "helmet": {"ok": (0, 200, 0),      "violation": (0, 0, 255)},
    "vest":   {"ok": (0, 180, 200),    "violation": (0, 140, 255)},
    "mask":   {"ok": (200, 200, 0),    "violation": (0, 100, 255)},
}

PPE_LABELS_VI = {
    "helmet": "Mu bao ho",
    "vest":   "Ao phan quang",
    "mask":   "Khau trang",
}


def _is_in_smoke_or_fire(box: list[int], hazard_boxes: list[list[float]], threshold: float = 0.28) -> bool:
    """Kiem tra xem mot bounding box co giao thoa dang ke voi vung khoi hoac lua khong."""
    if not hazard_boxes:
        return False
    bx1, by1, bx2, by2 = box
    b_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    for sbox in hazard_boxes:
        sx1, sy1, sx2, sy2 = sbox
        ix1 = max(bx1, sx1)
        iy1 = max(by1, sy1)
        ix2 = min(bx2, sx2)
        iy2 = min(by2, sy2)
        if ix2 > ix1 and iy2 > iy1:
            inter = float((ix2 - ix1) * (iy2 - iy1))
            if (inter / b_area) >= threshold:
                return True
    return False


def _calc_box_iou(b1: list[int], b2: list[int]) -> float:
    """Tinh IoU giua hai bounding box [x1, y1, x2, y2]."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    a1 = float(max(b1[2] - b1[0], 0) * max(b1[3] - b1[1], 0))
    a2 = float(max(b2[2] - b2[0], 0) * max(b2[3] - b2[1], 0))
    union = a1 + a2 - inter
    return inter / max(union, 1.0)


class PPEDetector:
    def __init__(self, conf: float = 0.35, iou: float = 0.45) -> None:
        if os.path.isfile(LOCAL_WEIGHTS):
            logger.info("Dung model tu train: %s", LOCAL_WEIGHTS)
            weights_path = LOCAL_WEIGHTS
        else:
            logger.info("Khong thay model tu train, dung pretrained: %s", FALLBACK_REPO_ID)
            weights_path = hf_hub_download(repo_id=FALLBACK_REPO_ID, filename="best.pt")

        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.ppe_memory_duration: float = 6.0  # seconds to remember verified PPE status during fast motion/occlusion
        self._ppe_memory: dict[int, dict[str, dict]] = {}

    # ---------- Xu ly video upload ----------

    def process_video(
        self,
        input_path: str,
        output_path: str,
        frame_skip: int = 1,
        fall_detector=None,
        fire_detector=None,
        enable_ppe: bool = True,
        on_progress: Callable[[int, int, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, int]:
        """Doc video, chay detection, ghi video ket qua, tra ve thong ke."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Khong the mo video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        start_time = time.time()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width -= width % 2
        height -= height % 2

        out = imageio.get_writer(
            output_path, fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p",
            macro_block_size=None,
        )

        self._ppe_memory.clear()
        if fall_detector and hasattr(fall_detector, "reset"):
            fall_detector.reset()
        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "violation_frames": 0,
            "helmet_ok": 0, "helmet_violation": 0,
            "vest_ok": 0, "vest_violation": 0,
            "mask_ok": 0, "mask_violation": 0,
            "fall_count": 0,
            "fire_count": 0,
            "smoke_count": 0,
        }

        frame_idx = 0
        last_annotated = None

        while True:
            if cancel_event and cancel_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            stats["total_frames"] += 1

            elapsed = time.time() - start_time
            if on_progress and frame_idx % 5 == 0:
                on_progress(frame_idx, total_frames, elapsed)

            run_inference = (frame_skip <= 1) or (frame_idx % frame_skip == 0)

            if run_inference:
                video_ts = frame_idx / fps
                annotated, violation, frame_ppe, _, falls, fires, _pids = self.annotate_frame(
                    frame,
                    fall_detector=fall_detector,
                    fire_detector=fire_detector,
                    timestamp=video_ts,
                    enable_ppe=enable_ppe,
                )
                last_annotated = annotated
                stats["processed_frames"] += 1
                for ppe_type in ("helmet", "vest", "mask"):
                    stats[f"{ppe_type}_ok"] += frame_ppe[ppe_type]["ok"]
                    stats[f"{ppe_type}_violation"] += frame_ppe[ppe_type]["violation"]
                if falls:
                    stats["fall_count"] += len(falls)
                if fires:
                    stats["fire_count"] += sum(1 for f in fires if f.get("type") == "fire")
                    stats["smoke_count"] += sum(1 for f in fires if f.get("type") == "smoke")
                if violation:
                    stats["violation_frames"] += 1
                write_frame = annotated
            else:
                write_frame = last_annotated if last_annotated is not None else frame

            write_frame = write_frame[:height, :width]
            out.append_data(cv2.cvtColor(write_frame, cv2.COLOR_BGR2RGB))

        cap.release()
        out.close()
        return stats

    # ---------- Xu ly 1 khung hinh ----------

    def annotate_frame(
        self,
        frame: Any,
        fall_detector=None,
        fire_detector=None,
        timestamp: float | None = None,
        enable_ppe: bool = True,
    ) -> tuple[Any, bool, dict, list, list, list, list]:
        """Chay detection va ByteTrack tren 1 frame, lien ket Nguoi - PPE, ve annotation thong minh."""
        ppe_stats: dict[str, dict[str, int]] = {
            "helmet": {"ok": 0, "violation": 0},
            "vest":   {"ok": 0, "violation": 0},
            "mask":   {"ok": 0, "violation": 0},
        }
        violations = []
        person_items = []
        assigned_ppe_items = []
        now = time.time() if timestamp is None else timestamp

        # 1. FIRE / SMOKE DETECT (chay truoc tren frame sach de co thong tin vung khoi / lua)
        fires = []
        smoke_boxes = []
        fire_boxes = []
        if fire_detector is not None:
            fires = fire_detector.detect(frame, timestamp=timestamp)
            for f in fires:
                fb = f.get("bbox")
                if fb:
                    if f.get("type") == "smoke":
                        smoke_boxes.append(fb)
                    else:
                        fire_boxes.append(fb)
        hazard_boxes = smoke_boxes + fire_boxes

        # 2. FALL DETECT (chay tren frame sach, xac thuc khung xuong nguoi that, loai bo ao giac do khoi)
        falls = []
        verified_pose_persons = []
        if fall_detector is not None:
            falls = fall_detector.detect(frame, smoke_boxes=smoke_boxes)
            verified_pose_persons = getattr(fall_detector, "last_detected_persons", [])

        # 3. PPE DETECTION (Stage 1: Xac thuc nguoi; Stage 2: Danh gia PPE gan voi nguoi)
        if enable_ppe:
            try:
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf, iou=self.iou, verbose=False)
            except Exception:
                results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)

            r = results[0]
            boxes = r.boxes
            names = r.names

            candidate_persons = []
            candidate_ppe = []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                low_label = label.lower().strip()
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if low_label in ("person", "worker", "people"):
                    track_id = int(box.id[0]) if (hasattr(box, "id") and box.id is not None) else (i + 1)
                    candidate_persons.append({
                        "id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf_score,
                    })
                else:
                    ppe_res = _classify_label(label)
                    if ppe_res is not None:
                        # Bo qua ppe ao giac nam trong vung khoi neu conf thap
                        in_hazard = _is_in_smoke_or_fire([x1, y1, x2, y2], hazard_boxes, threshold=0.35)
                        if in_hazard and conf_score < 0.60:
                            continue

                        ppe_type, status = ppe_res
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        candidate_ppe.append({
                            "type": ppe_type,
                            "status": status,
                            "conf": conf_score,
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "center": (cx, cy),
                        })

            # STAGE 1: XAC THUC NGUOI THAT (Cross-verification voi Pose Model & Do sac net canh)
            h_f, w_f = frame.shape[:2]
            for cp in candidate_persons:
                cx1, cy1, cx2, cy2 = cp["bbox"]
                c_conf = cp["conf"]
                in_hazard = _is_in_smoke_or_fire([cx1, cy1, cx2, cy2], hazard_boxes, threshold=0.28)

                rx1, ry1 = max(0, cx1), max(0, cy1)
                rx2, ry2 = min(w_f, cx2), min(h_f, cy2)
                roi = frame[ry1:ry2, rx1:rx2]
                sharpness = 0.0
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

                is_valid = False
                bw = cx2 - cx1
                bh = cy2 - cy1
                # Loai bo cac dam khoi hoac mang tuong to lon o phia tren khung hinh bi nhan nham thanh nguoi
                is_sky_or_smoke_plume = (cy1 <= 25 and (bw > 240 or bh > 0.55 * h_f))

                if fall_detector is not None:
                    # He thong Pose dang chay: uu tien tuyet doi nguoi co skeleton thuc te
                    matched_pose = any(
                        _calc_box_iou(cp["bbox"], vp["bbox"]) >= 0.15
                        or (vp["bbox"][0] <= (cx1 + cx2)/2 <= vp["bbox"][2] and vp["bbox"][1] <= (cy1 + cy2)/2 <= vp["bbox"][3])
                        for vp in verified_pose_persons
                    )
                    if matched_pose:
                        is_valid = True
                    else:
                        # Khong co skeleton khop: chi chap nhan neu khong o ria troi/khoi, khong trong hazard, ro net va conf cao
                        if (not is_sky_or_smoke_plume) and (not in_hazard) and c_conf >= 0.72 and sharpness >= 45.0:
                            is_valid = True
                else:
                    # Fall detector khong bat: ap dung loc quang hoc nghiem ngat
                    if not is_sky_or_smoke_plume:
                        if in_hazard:
                            if c_conf >= 0.70 and sharpness >= 45.0:
                                is_valid = True
                        else:
                            if c_conf >= 0.60 and sharpness >= 30.0:
                                is_valid = True

                if is_valid:
                    person_items.append({
                        "id": cp["id"],
                        "bbox": cp["bbox"],
                        "conf": c_conf,
                        "helmet": "unknown",
                        "vest": "unknown",
                        "mask": "unknown",
                        "violations": [],
                    })

            # Bo sung nguoi duoc xac thuc boi Pose Model neu model PPE bi sot
            for vp in verified_pose_persons:
                vp_box = vp["bbox"]
                if not any(_calc_box_iou(vp_box, p["bbox"]) > 0.25 for p in person_items):
                    person_items.append({
                        "id": vp["pid"],
                        "bbox": vp_box,
                        "conf": vp["conf"],
                        "helmet": "unknown",
                        "vest": "unknown",
                        "mask": "unknown",
                        "violations": [],
                    })

            # STAGE 2: LIEN KET KHONG GIAN (CHI NHUNG VAT PHAM PPE THUOC VE NGUOI XAC THUC MOI DUOC TINH)
            if person_items:
                for ppe in candidate_ppe:
                    cx, cy = ppe["center"]
                    ppe_type = ppe["type"]
                    ppe_status = ppe["status"]

                    best_person = None
                    best_dist = float("inf")

                    for p in person_items:
                        px1, py1, px2, py2 = p["bbox"]
                        pw = max(px2 - px1, 1)
                        ph = max(py2 - py1, 1)

                        in_zone = False
                        if ppe_type == "helmet":
                            in_zone = (
                                px1 - 0.20 * pw <= cx <= px2 + 0.20 * pw
                                and py1 - 0.20 * ph <= cy <= py1 + 0.35 * ph
                            )
                        elif ppe_type == "vest":
                            in_zone = (
                                px1 - 0.20 * pw <= cx <= px2 + 0.20 * pw
                                and py1 + 0.18 * ph <= cy <= py1 + 0.85 * ph
                            )
                        elif ppe_type == "mask":
                            in_zone = (
                                px1 - 0.20 * pw <= cx <= px2 + 0.20 * pw
                                and py1 + 0.12 * ph <= cy <= py1 + 0.48 * ph
                            )

                        if not in_zone:
                            continue

                        pcx = (px1 + px2) / 2.0
                        pcy = (py1 + py2) / 2.0
                        dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_person = p

                    if best_person is not None:
                        best_person[ppe_type] = ppe_status
                        best_person[f"{ppe_type}_conf"] = ppe["conf"]
                        assigned_ppe_items.append(ppe)

            # Cap nhat thong ke PPE CHI TU NHUNG VAT PHAM DA GAN VOI NGUOI XAC THUC
            for ppe in assigned_ppe_items:
                ppe_stats[ppe["type"]][ppe["status"]] += 1

            # Ap dung bo nho trang thai PPE theo thoi gian (Temporal PPE Memory)
            for p in person_items:
                pid = p["id"]
                if pid not in self._ppe_memory:
                    self._ppe_memory[pid] = {}
                pmem = self._ppe_memory[pid]

                for ppe_type in ("helmet", "vest", "mask"):
                    curr = p[ppe_type]
                    if curr == "ok":
                        pmem[ppe_type] = {
                            "status": "ok",
                            "last_seen": now,
                            "conf": p.get(f"{ppe_type}_conf", p["conf"]),
                        }
                    elif curr == "violation":
                        pmem[ppe_type] = {
                            "status": "violation",
                            "last_seen": now,
                            "conf": p.get(f"{ppe_type}_conf", p["conf"]),
                        }
                    else:  # curr == "unknown" (mat dau do chuyen dong nhanh, mo, quay mat)
                        if ppe_type in pmem:
                            last_rec = pmem[ppe_type]
                            if last_rec["status"] == "ok" and (now - last_rec["last_seen"]) <= self.ppe_memory_duration:
                                p[ppe_type] = "ok"
                                p[f"{ppe_type}_conf"] = last_rec["conf"]
                                p[f"{ppe_type}_from_memory"] = True
                                ppe_stats[ppe_type]["ok"] += 1

            # Don dep bo nho nguoi da roi khoi goc camera qua 20s
            for old_pid in list(self._ppe_memory.keys()):
                last_activity = max([m.get("last_seen", 0) for m in self._ppe_memory[old_pid].values()] or [0])
                if now - last_activity > 20.0:
                    self._ppe_memory.pop(old_pid, None)

            # Xac dinh vi pham cho tung person
            for p in person_items:
                if p["helmet"] == "violation":
                    violations.append({
                        "type": "no_helmet",
                        "confidence": p.get("helmet_conf", p["conf"]),
                        "person_id": p["id"],
                        "label": PPE_LABELS_VI["helmet"],
                    })
                if p["vest"] == "violation":
                    violations.append({
                        "type": "no_vest",
                        "confidence": p.get("vest_conf", p["conf"]),
                        "person_id": p["id"],
                        "label": PPE_LABELS_VI["vest"],
                    })
                if p["mask"] == "violation":
                    violations.append({
                        "type": "no_mask",
                        "confidence": p.get("mask_conf", p["conf"]),
                        "person_id": p["id"],
                        "label": PPE_LABELS_VI["mask"],
                    })

        # VE PPE ANNOTATION LEN FRAME (chi khi enable_ppe va co vat pham da gan voi nguoi)
        if enable_ppe:
            for ppe in assigned_ppe_items:
                x1, y1, x2, y2 = ppe["bbox"]
                color = PPE_COLORS[ppe["type"]][ppe["status"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"{ppe['label']} {ppe['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 2, y1), color, -1)
                cv2.putText(frame, lbl, (x1 + 1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for p in person_items:
                px1, py1, px2, py2 = p["bbox"]
                is_person_violation = (p["helmet"] == "violation" or p["vest"] == "violation" or p["mask"] == "violation")
                box_color = (0, 0, 255) if is_person_violation else (0, 220, 0)

                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

                h_str = "OK" if p["helmet"] == "ok" else ("THIEU" if p["helmet"] == "violation" else "-")
                v_str = "OK" if p["vest"] == "ok" else ("THIEU" if p["vest"] == "violation" else "-")
                m_str = "OK" if p["mask"] == "ok" else ("THIEU" if p["mask"] == "violation" else "-")
                person_tag = f"ID #{p['id']} | Mu:{h_str} | Ao:{v_str} | KT:{m_str}"

                (tw, th), _ = cv2.getTextSize(person_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y = max(py1 - th - 8, 0)
                cv2.rectangle(frame, (px1, tag_y), (px1 + tw + 6, py1), box_color, -1)
                cv2.putText(frame, person_tag, (px1 + 3, max(py1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Ve fall annotation SAU CUNG (sau PPE) — LUÔN vẽ skeleton
        if fall_detector is not None:
            frame = fall_detector.annotate_frame(frame, falls)

        # Ve fire/smoke annotation SAU CUNG
        if fire_detector is not None and fires:
            frame = fire_detector.annotate_frame(frame, fires)

        has_violation = (enable_ppe and len(violations) > 0) or len(falls) > 0 or len(fires) > 0

        # Ưu tiên hiển thị cảnh báo: CHÁY/KHÓI > NGÃ > VI PHẠM PPE
        if fires:
            # fire_detector.annotate_frame da ve top emergency bar
            pass
        elif falls:
            cv2.putText(frame, "CAP CUU: PHAT HIEN NGA BAT DONG!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        elif has_violation and enable_ppe:
            cv2.putText(frame, "CANH BAO: Phat hien vi pham PPE!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Danh sách TẤT CẢ person IDs trong frame (kể cả người tuân thủ 100%)
        all_person_ids = [p["id"] for p in person_items]

        return frame, has_violation, ppe_stats, violations, falls, fires, all_person_ids
