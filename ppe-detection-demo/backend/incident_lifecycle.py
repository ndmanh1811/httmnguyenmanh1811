"""
incident_lifecycle.py - Stream-isolated Incident Lifecycle State Machine
Handles lifecycle transitions:
  NONE -> CANDIDATE -> CONFIRMED -> ACTIVE (Grace Period) -> RESOLVED

Ensures:
  1. A single continuing incident generates only ONE initial database record.
  2. Distinct handling of Active Miss Grace (temporary flicker) vs Resolved Cooldown.
  3. Clean tracking of started_at, confirmed_at, ended_at, and total duration.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from spatial_continuity import SpatialContinuityManager


class IncidentState(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    RESOLVED = "resolved"


@dataclass
class IncidentEvent:
    incident_id: str
    event_type: str  # "fire_detected" | "smoke_detected" | "fall_detected"
    state: IncidentState
    confidence: float
    started_at: float
    confirmed_at: Optional[float]
    last_seen: float
    ended_at: Optional[float]
    duration_seconds: float
    is_new_alert: bool = False
    in_miss_grace: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class IncidentPolicy:
    """Chính sách vòng đời sự cố theo từng loại đối tượng."""
    confirm_duration_sec: float = 1.0
    trigger_threshold: float = 0.50
    hold_threshold: float = 0.35
    active_miss_grace_sec: float = 1.5
    resolved_after_sec: float = 5.0


PPE_POLICY = IncidentPolicy(
    confirm_duration_sec=1.2,
    trigger_threshold=0.50,
    hold_threshold=0.35,
    active_miss_grace_sec=2.0,
    resolved_after_sec=4.0,
)

FALL_POLICY = IncidentPolicy(
    confirm_duration_sec=0.6,
    trigger_threshold=0.45,
    hold_threshold=0.30,
    active_miss_grace_sec=2.0,
    resolved_after_sec=5.0,
)

FIRE_POLICY = IncidentPolicy(
    confirm_duration_sec=0.8,
    trigger_threshold=0.50,
    hold_threshold=0.35,
    active_miss_grace_sec=1.5,
    resolved_after_sec=5.0,
)


class IncidentLifecycleManager:
    """
    Manages the lifecycle of an incident for a single stream.
    Supports either explicit threshold dicts or an IncidentPolicy object.
    """

    def __init__(
        self,
        confirm_duration_sec: Optional[dict[str, float] | float] = None,
        trigger_thresholds: Optional[dict[str, float] | float] = None,
        hold_thresholds: Optional[dict[str, float] | float] = None,
        active_miss_grace_sec: float = 1.5,
        resolved_after_sec: float = 5.0,
        policy: Optional[IncidentPolicy] = None,
        continuity_manager: Optional[SpatialContinuityManager] = None,
        enable_spatial_continuity: bool = True,
    ):
        self.policy = policy
        if continuity_manager is not None:
            self.continuity_mgr = continuity_manager
        elif enable_spatial_continuity:
            self.continuity_mgr = SpatialContinuityManager()
        else:
            self.continuity_mgr = None

        if policy is not None:
            self.confirm_duration_sec = {
                "default": policy.confirm_duration_sec,
                "fire": policy.confirm_duration_sec,
                "smoke": policy.confirm_duration_sec,
                "fall": policy.confirm_duration_sec,
                "ppe": policy.confirm_duration_sec,
                "danger_zone": policy.confirm_duration_sec,
            }
            self.trigger_thresholds = {
                "default": policy.trigger_threshold,
                "fire": policy.trigger_threshold,
                "smoke": policy.trigger_threshold,
                "fall": policy.trigger_threshold,
                "ppe": policy.trigger_threshold,
                "danger_zone": policy.trigger_threshold,
            }
            self.hold_thresholds = {
                "default": policy.hold_threshold,
                "fire": policy.hold_threshold,
                "smoke": policy.hold_threshold,
                "fall": policy.hold_threshold,
                "ppe": policy.hold_threshold,
                "danger_zone": policy.hold_threshold,
            }
            self.active_miss_grace_sec = policy.active_miss_grace_sec
            self.resolved_after_sec = policy.resolved_after_sec
        else:
            if isinstance(confirm_duration_sec, dict):
                self.confirm_duration_sec = confirm_duration_sec
            elif isinstance(confirm_duration_sec, (int, float)):
                v = float(confirm_duration_sec)
                self.confirm_duration_sec = {"default": v, "fire": v, "smoke": v, "fall": v, "ppe": v, "danger_zone": v}
            else:
                self.confirm_duration_sec = {
                    "default": 1.0,
                    "fire": 0.8,
                    "smoke": 2.0,
                    "fall": 0.6,
                    "ppe": 1.2,
                    "danger_zone": 0.5,
                }

            if isinstance(trigger_thresholds, dict):
                self.trigger_thresholds = trigger_thresholds
            elif isinstance(trigger_thresholds, (int, float)):
                v = float(trigger_thresholds)
                self.trigger_thresholds = {"default": v, "fire": v, "smoke": v, "fall": v, "ppe": v, "danger_zone": v}
            else:
                self.trigger_thresholds = {
                    "default": 0.50,
                    "fire": 0.50,
                    "smoke": 0.45,
                    "fall": 0.45,
                    "ppe": 0.50,
                    "danger_zone": 0.50,
                }

            if isinstance(hold_thresholds, dict):
                self.hold_thresholds = hold_thresholds
            elif isinstance(hold_thresholds, (int, float)):
                v = float(hold_thresholds)
                self.hold_thresholds = {"default": v, "fire": v, "smoke": v, "fall": v, "ppe": v, "danger_zone": v}
            else:
                self.hold_thresholds = {
                    "default": 0.35,
                    "fire": 0.35,
                    "smoke": 0.30,
                    "fall": 0.30,
                    "ppe": 0.35,
                    "danger_zone": 0.30,
                }

            self.active_miss_grace_sec = active_miss_grace_sec
            self.resolved_after_sec = resolved_after_sec

        self.current_state: IncidentState = IncidentState.NONE
        self.active_incident: Optional[IncidentEvent] = None
        self._candidate_start_time: Optional[float] = None
        self._candidate_type: Optional[str] = None
        self._candidate_id: Optional[str] = None
        self._candidate_original_started_at: Optional[float] = None
        self._candidate_max_conf: float = 0.0
        self._candidate_max_score: float = 0.0

    def update(
        self,
        detected_items: list[dict],
        timestamp: Optional[float] = None,
        frame_size: tuple[int, int] = (720, 1280),
    ) -> Optional[IncidentEvent]:
        """
        Update the lifecycle with detections from the current frame.
        """
        if timestamp is None:
            timestamp = time.time()

        has_detection = len(detected_items) > 0

        if has_detection:
            # Priority: fire > smoke
            fire_items = [d for d in detected_items if d.get("type") == "fire"]
            smoke_items = [d for d in detected_items if d.get("type") == "smoke"]

            if fire_items:
                primary = max(fire_items, key=lambda x: x.get("score", x.get("confidence", 0.0)))
                event_type = "fire_detected"
                cls_key = "fire"
            elif smoke_items:
                primary = max(smoke_items, key=lambda x: x.get("score", x.get("confidence", 0.0)))
                event_type = "smoke_detected"
                cls_key = "smoke"
            else:
                primary = max(detected_items, key=lambda x: x.get("score", x.get("confidence", 0.0)))
                raw_type = str(primary.get("type", "hazard_detected")).strip()
                low_type = raw_type.lower()
                if "fall" in low_type:
                    event_type = "fall_detected"
                    cls_key = "fall"
                elif "danger_zone" in low_type or "exclusion" in low_type or "zone" in low_type:
                    event_type = "danger_zone_entry"
                    cls_key = "danger_zone"
                elif low_type.startswith("no_") or "ppe" in low_type or low_type in ("helmet", "vest", "mask"):
                    event_type = raw_type
                    cls_key = "ppe"
                else:
                    event_type = raw_type
                    cls_key = low_type

            score = float(primary.get("score", primary.get("confidence", 0.0)))
            conf = float(primary.get("confidence", score))

            if self.policy is not None:
                target_confirm_duration = self.policy.confirm_duration_sec
                target_trigger = self.policy.trigger_threshold
                target_hold = self.policy.hold_threshold
            else:
                target_confirm_duration = self.confirm_duration_sec.get(
                    cls_key, self.confirm_duration_sec.get("default", 1.0)
                )
                target_trigger = self.trigger_thresholds.get(
                    cls_key, self.trigger_thresholds.get("default", 0.50)
                )
                target_hold = self.hold_thresholds.get(
                    cls_key, self.hold_thresholds.get("default", 0.35)
                )

            if self.current_state in (IncidentState.NONE, IncidentState.RESOLVED):
                if score >= target_hold:
                    # Check for spatial continuity with recently resolved incident
                    reopened = None
                    if self.continuity_mgr and primary.get("bbox"):
                        reopened = self.continuity_mgr.find_reopen_match(
                            event_type=event_type,
                            bbox=primary["bbox"],
                            current_ts=timestamp,
                            frame_size=frame_size,
                        )

                    if reopened is not None:
                        inc_id = reopened.incident_id
                        cand_start = reopened.started_at
                        gap = timestamp - reopened.resolved_at

                        if gap <= 1.0 and score >= target_trigger:
                            # Immediate resume to ACTIVE (transient flicker <= 1.0s)
                            self.current_state = IncidentState.ACTIVE
                            self.active_incident = IncidentEvent(
                                incident_id=inc_id,
                                event_type=event_type,
                                state=IncidentState.ACTIVE,
                                confidence=max(reopened.confidence, conf),
                                started_at=cand_start,
                                confirmed_at=reopened.confirmed_at or timestamp,
                                last_seen=timestamp,
                                ended_at=None,
                                duration_seconds=timestamp - cand_start,
                                is_new_alert=False,
                                details={
                                    "bbox": primary.get("bbox", []),
                                    "score": score,
                                    "cls_key": cls_key,
                                    "reopened_from": inc_id,
                                    "reopen_gap_sec": round(gap, 2),
                                    "components": primary.get("components", {}),
                                },
                            )
                            return self.active_incident
                        else:
                            # 1.0s < gap <= 3.0s: Fast-Confirm CANDIDATE (50% confirm duration)
                            self.current_state = IncidentState.CANDIDATE
                            self._candidate_start_time = timestamp - (target_confirm_duration * 0.5)
                            self._candidate_type = event_type
                            self._candidate_id = inc_id
                            self._candidate_original_started_at = cand_start
                            self._candidate_max_conf = max(reopened.confidence, conf)
                            self._candidate_max_score = score
                            return IncidentEvent(
                                incident_id=inc_id,
                                event_type=event_type,
                                state=IncidentState.CANDIDATE,
                                confidence=conf,
                                started_at=cand_start,
                                confirmed_at=None,
                                last_seen=timestamp,
                                ended_at=None,
                                duration_seconds=timestamp - cand_start,
                                is_new_alert=False,
                                details={
                                    "bbox": primary.get("bbox", []),
                                    "score": score,
                                    "cls_key": cls_key,
                                    "reopened_from": inc_id,
                                    "reopen_gap_sec": round(gap, 2),
                                    "components": primary.get("components", {}),
                                },
                            )

                    # Start new CANDIDATE phase
                    self.current_state = IncidentState.CANDIDATE
                    self._candidate_start_time = timestamp
                    self._candidate_type = event_type
                    self._candidate_id = None
                    self._candidate_original_started_at = None
                    self._candidate_max_conf = conf
                    self._candidate_max_score = score
                    return IncidentEvent(
                        incident_id=uuid.uuid4().hex[:8],
                        event_type=event_type,
                        state=IncidentState.CANDIDATE,
                        confidence=conf,
                        started_at=timestamp,
                        confirmed_at=None,
                        last_seen=timestamp,
                        ended_at=None,
                        duration_seconds=0.0,
                        is_new_alert=False,
                        details={
                            "bbox": primary.get("bbox", []),
                            "score": score,
                            "cls_key": cls_key,
                            "components": primary.get("components", {}),
                        },
                    )

            elif self.current_state == IncidentState.CANDIDATE:
                if score < target_hold:
                    # Candidate score faded below hold threshold -> cancel candidate
                    self.current_state = IncidentState.NONE
                    self._candidate_start_time = None
                    self._candidate_id = None
                    self._candidate_original_started_at = None
                    return None

                self._candidate_max_conf = max(self._candidate_max_conf, conf)
                self._candidate_max_score = max(self._candidate_max_score, score)
                cand_start = self._candidate_start_time if self._candidate_start_time is not None else timestamp
                time_in_candidate = timestamp - cand_start

                # Hysteresis confirmation: requires trigger threshold AND sufficient duration
                if time_in_candidate >= target_confirm_duration and score >= target_trigger:
                    # Promoted to CONFIRMED!
                    self.current_state = IncidentState.CONFIRMED
                    inc_id = self._candidate_id or uuid.uuid4().hex[:8]
                    orig_start = self._candidate_original_started_at or cand_start
                    is_reopened = self._candidate_id is not None
                    self.active_incident = IncidentEvent(
                        incident_id=inc_id,
                        event_type=self._candidate_type or event_type,
                        state=IncidentState.CONFIRMED,
                        confidence=self._candidate_max_conf,
                        started_at=orig_start,
                        confirmed_at=timestamp,
                        last_seen=timestamp,
                        ended_at=None,
                        duration_seconds=timestamp - orig_start,
                        is_new_alert=not is_reopened,
                        details={
                            "bbox": primary.get("bbox", []),
                            "score": score,
                            "cls_key": cls_key,
                            "reopened": is_reopened,
                            "components": primary.get("components", {}),
                        },
                    )
                    return self.active_incident
                else:
                    inc_id = self._candidate_id or "candidate"
                    orig_start = self._candidate_original_started_at or cand_start
                    return IncidentEvent(
                        incident_id=inc_id,
                        event_type=self._candidate_type or event_type,
                        state=IncidentState.CANDIDATE,
                        confidence=conf,
                        started_at=orig_start,
                        confirmed_at=None,
                        last_seen=timestamp,
                        ended_at=None,
                        duration_seconds=timestamp - orig_start,
                        is_new_alert=False,
                        details={
                            "bbox": primary.get("bbox", []),
                            "score": score,
                            "cls_key": cls_key,
                            "components": primary.get("components", {}),
                        },
                    )

            elif self.current_state in (IncidentState.CONFIRMED, IncidentState.ACTIVE):
                if score >= target_hold:
                    # Maintain ACTIVE state
                    self.current_state = IncidentState.ACTIVE
                    if self.active_incident is not None:
                        self.active_incident.state = IncidentState.ACTIVE
                        self.active_incident.last_seen = timestamp
                        self.active_incident.duration_seconds = timestamp - self.active_incident.started_at
                        self.active_incident.confidence = max(self.active_incident.confidence, conf)
                        self.active_incident.is_new_alert = False
                        self.active_incident.in_miss_grace = False
                        self.active_incident.details["bbox"] = primary.get("bbox", [])
                        self.active_incident.details["score"] = score
                        self.active_incident.details["cls_key"] = cls_key
                        self.active_incident.details["components"] = primary.get("components", {})
                        return self.active_incident

        else:
            # No detection in current frame
            if self.current_state == IncidentState.CANDIDATE:
                # Transitory noise vanished before reaching confirmation threshold
                self.current_state = IncidentState.NONE
                self._candidate_start_time = None
                self._candidate_id = None
                self._candidate_original_started_at = None
                return None

            elif self.current_state in (IncidentState.CONFIRMED, IncidentState.ACTIVE):
                if self.active_incident is not None:
                    absence_duration = timestamp - self.active_incident.last_seen
                    if absence_duration < self.active_miss_grace_sec:
                        # Inside active miss grace: stay ACTIVE with in_miss_grace flag
                        self.active_incident.state = IncidentState.ACTIVE
                        self.active_incident.is_new_alert = False
                        self.active_incident.in_miss_grace = True
                        return self.active_incident
                    elif absence_duration < self.resolved_after_sec:
                        # Between miss grace and full resolution: still in cooldown, holding active incident
                        self.active_incident.state = IncidentState.ACTIVE
                        self.active_incident.is_new_alert = False
                        self.active_incident.in_miss_grace = True
                        return self.active_incident
                    else:
                        # Full absence threshold exceeded -> RESOLVED
                        self.current_state = IncidentState.RESOLVED
                        resolved_event = IncidentEvent(
                            incident_id=self.active_incident.incident_id,
                            event_type=self.active_incident.event_type,
                            state=IncidentState.RESOLVED,
                            confidence=self.active_incident.confidence,
                            started_at=self.active_incident.started_at,
                            confirmed_at=self.active_incident.confirmed_at,
                            last_seen=self.active_incident.last_seen,
                            ended_at=self.active_incident.last_seen,
                            duration_seconds=self.active_incident.last_seen - self.active_incident.started_at,
                            is_new_alert=False,
                            in_miss_grace=False,
                            details=self.active_incident.details,
                        )
                        if self.continuity_mgr:
                            self.continuity_mgr.register_resolved(
                                incident_id=resolved_event.incident_id,
                                event_type=resolved_event.event_type,
                                bbox=resolved_event.details.get("bbox", []),
                                started_at=resolved_event.started_at,
                                confirmed_at=resolved_event.confirmed_at,
                                resolved_at=timestamp,
                                confidence=resolved_event.confidence,
                                frame_size=frame_size,
                                details=resolved_event.details,
                            )
                        self.active_incident = None
                        return resolved_event

        return None

    def reset(self):
        """Reset state machine (e.g. when video stream restarts)."""
        self.current_state = IncidentState.NONE
        self.active_incident = None
        self._candidate_start_time = None
        self._candidate_type = None
        self._candidate_id = None
        self._candidate_original_started_at = None
        self._candidate_max_conf = 0.0
        self._candidate_max_score = 0.0
        if self.continuity_mgr:
            self.continuity_mgr.reset()
