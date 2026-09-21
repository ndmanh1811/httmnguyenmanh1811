"""
fall_detector.py
----------------
Bộ phát hiện ngã nâng cấp: Sử dụng PoseFallDetector (YOLO-Pose 17 Keypoints + PyTorch LSTM).
Hỗ trợ tương thích ngược toàn bộ API cũ:
    - detect(frame) -> list[dict]
    - annotate_frame(frame, falls) -> frame
"""

from __future__ import annotations

import logging
from typing import Any

from pose_fall_detector import PoseFallDetector

logger = logging.getLogger(__name__)


class FallDetector:
    def __init__(
        self,
        conf: float = 0.50,
        angle_threshold: float = 50.0,
        aspect_ratio_threshold: float = 1.15,
        required_consecutive_frames: int = 2,
    ) -> None:
        logger.info("Initializing FallDetector via PoseFallDetector (SOTA Keypoints)...")
        self._impl = PoseFallDetector(
            conf=conf,
            angle_threshold=angle_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
            required_consecutive_frames=required_consecutive_frames,
        )

    def detect(self, frame: Any, smoke_boxes: list | None = None) -> list[dict]:
        """Phat hien nga dong hoc dua tren goc than minh va thoi gian bat dong."""
        return self._impl.detect(frame, smoke_boxes=smoke_boxes)

    def annotate_frame(self, frame: Any, falls: list[dict]) -> Any:
        """Ve khung xuong va thong bao tai nan nga len khung hinh."""
        return self._impl.annotate_frame(frame, falls)

    @property
    def last_detected_persons(self) -> list[dict]:
        """Danh sach tat ca nguoi da duoc xac thuc khung xuong trong frame gan nhat."""
        return getattr(self._impl, "_last_detected_persons", [])

    def reset(self) -> None:
        """Reset trạng thái phát hiện ngã khi bắt đầu video mới."""
        if hasattr(self._impl, "reset"):
            self._impl.reset()
