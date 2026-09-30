"""
spatial_continuity.py
---------------------
Continuity resolution and coarse spatial key implementation for SafeGuard AI:
Implements Section 4 of ai_architecture_plan.md:
- Coarse spatial cell partitioning (grid_cols x grid_rows)
- Euclidean distance metric on normalized center coordinates [0.0, 1.0]
- Reopen cooldown window with nearest-neighbor matching (greedy argmin)
- Incident continuation without individual tracking/Re-ID
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResolvedIncidentRecord:
    incident_id: str
    event_type: str
    last_bbox: list[int | float]
    last_center: tuple[float, float]
    started_at: float
    confirmed_at: Optional[float]
    resolved_at: float
    confidence: float
    details: dict = field(default_factory=dict)


def compute_normalized_center(
    bbox: list[int | float],
    frame_size: tuple[int, int] = (720, 1280),
) -> tuple[float, float]:
    """
    Computes normalized (cx, cy) in [0.0, 1.0] from [x1, y1, x2, y2].
    frame_size format: (height, width).
    """
    if not bbox or len(bbox) < 4:
        return (0.5, 0.5)

    h, w = frame_size[:2]
    h = max(1, h)
    w = max(1, w)

    x1, y1, x2, y2 = bbox[:4]
    cx = float(x1 + x2) / (2.0 * w)
    cy = float(y1 + y2) / (2.0 * h)

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    return (cx, cy)


def get_spatial_cell(
    bbox: list[int | float],
    frame_size: tuple[int, int] = (720, 1280),
    grid_cols: int = 10,
    grid_rows: int = 6,
) -> tuple[int, int]:
    """
    Maps bounding box to coarse grid cell (col, row).
    Grid cols: default 10, Grid rows: default 6 (matches 16:9 ratio).
    """
    cx, cy = compute_normalized_center(bbox, frame_size)
    col = min(int(cx * grid_cols), grid_cols - 1)
    row = min(int(cy * grid_rows), grid_rows - 1)
    return (col, row)


def compute_spatial_key(
    event_type: str,
    bbox: list[int | float],
    zone_id: str = "global",
    frame_size: tuple[int, int] = (720, 1280),
    grid_cols: int = 10,
    grid_rows: int = 6,
) -> str:
    """
    Constructs deterministic spatial key: {zone_id}_{event_type}_c{col}_r{row}.
    """
    col, row = get_spatial_cell(bbox, frame_size, grid_cols, grid_rows)
    return f"{zone_id}_{event_type}_c{col}_r{row}"


def euclidean_normalized_distance(
    c1: tuple[float, float],
    c2: tuple[float, float],
) -> float:
    """Euclidean distance between two normalized centers."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


class SpatialContinuityManager:
    """
    Per-stream continuity resolver for recently resolved incidents.
    Prevents incident fragmentation when an object crosses cell boundaries or
    experiences momentary occlusion.
    """

    def __init__(
        self,
        merge_distance: float = 0.15,
        reopen_cooldown: float = 3.0,
        grid_cols: int = 10,
        grid_rows: int = 6,
    ):
        self.merge_distance = merge_distance
        self.reopen_cooldown = reopen_cooldown
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self._resolved_records: list[ResolvedIncidentRecord] = []

    def register_resolved(
        self,
        incident_id: str,
        event_type: str,
        bbox: list[int | float],
        started_at: float,
        confirmed_at: Optional[float],
        resolved_at: float,
        confidence: float,
        frame_size: tuple[int, int] = (720, 1280),
        details: Optional[dict] = None,
    ) -> None:
        """Register a recently resolved incident into the continuity buffer."""
        center = compute_normalized_center(bbox, frame_size)
        rec = ResolvedIncidentRecord(
            incident_id=incident_id,
            event_type=event_type,
            last_bbox=list(bbox) if bbox else [],
            last_center=center,
            started_at=started_at,
            confirmed_at=confirmed_at,
            resolved_at=resolved_at,
            confidence=confidence,
            details=details or {},
        )
        self._resolved_records.append(rec)
        self._cleanup(resolved_at)

    def find_reopen_match(
        self,
        event_type: str,
        bbox: list[int | float],
        current_ts: float,
        frame_size: tuple[int, int] = (720, 1280),
    ) -> Optional[ResolvedIncidentRecord]:
        """
        Looks for a recent resolved incident matching event_type and within merge_distance.
        Applies Greedy Nearest Neighbor (argmin distance).
        If match found, removes the record from resolved buffer so it cannot be matched again (1-to-many collision resolution).
        """
        self._cleanup(current_ts)
        if not self._resolved_records or not bbox or len(bbox) < 4:
            return None

        curr_center = compute_normalized_center(bbox, frame_size)

        best_idx: Optional[int] = None
        min_dist = float("inf")

        for idx, rec in enumerate(self._resolved_records):
            if rec.event_type != event_type:
                continue

            # Check cooldown window
            if (current_ts - rec.resolved_at) > self.reopen_cooldown:
                continue

            dist = euclidean_normalized_distance(curr_center, rec.last_center)
            if dist <= self.merge_distance and dist < min_dist:
                min_dist = dist
                best_idx = idx

        if best_idx is not None:
            # Match found! Pop record to avoid duplicate merges (greedy assignment)
            return self._resolved_records.pop(best_idx)

        return None

    def _cleanup(self, current_ts: float) -> None:
        """Prunes records older than reopen_cooldown."""
        self._resolved_records = [
            r for r in self._resolved_records
            if (current_ts - r.resolved_at) <= (self.reopen_cooldown + 1.0)
        ]

    def reset(self) -> None:
        """Clear all records."""
        self._resolved_records.clear()
