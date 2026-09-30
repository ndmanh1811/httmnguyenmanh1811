"""
frame_scheduler.py
------------------
Bộ điều phối khung hình lệch pha (Phase-Shifted FixedSlotScheduler).
Theo đặc tả mục 3.2 trong ai_architecture_plan.md:
  Chu kỳ 6 frame. Fall và Fire KHÔNG BAO GIỜ trùng slot:
    Frame 0: PPE + Fire
    Frame 1: Fall
    Frame 2: Fall
    Frame 3: PPE + Fire
    Frame 4: Fall
    Frame 5: PPE
  Tần số tương đối (khi camera source = 30 FPS):
    PPE  = 3/6 = 15 FPS
    Fall = 3/6 = 15 FPS
    Fire = 2/6 = 10 FPS
    Fall ∩ Fire = ∅  và  PPE ∩ Fall ∩ Fire = ∅
Mỗi CameraManager sở hữu một instance riêng (per-stream state isolation).
"""

from __future__ import annotations


class FixedSlotScheduler:
    def __init__(self) -> None:
        self.frame_idx = 0

    def should_run(
        self,
        enable_ppe: bool = True,
        enable_fall: bool = True,
        enable_fire: bool = True,
    ) -> dict[str, bool]:
        slot = self.frame_idx % 6
        self.frame_idx += 1

        run_ppe = enable_ppe and (slot in (0, 3, 5))
        run_fall = enable_fall and (slot in (1, 2, 4))
        run_fire = enable_fire and (slot in (0, 3))

        return {
            "run_ppe": bool(run_ppe),
            "run_fall": bool(run_fall),
            "run_fire": bool(run_fire),
            "slot": slot,
            "frame_idx": self.frame_idx,
        }

    def reset(self) -> None:
        """Đặt lại bộ đếm khi đổi stream hoặc nguồn camera."""
        self.frame_idx = 0
