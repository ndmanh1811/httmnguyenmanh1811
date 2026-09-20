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


class IncidentLifecycleManager:
    """
    Manages the lifecycle of an incident for a single stream.
    
    Parameters:
      - confirm_duration_sec: Class-specific continuous presence required to confirm an incident (e.g. fire=0.8s, smoke=2.0s).
      - trigger_thresholds: Class-specific soft score threshold to promote CANDIDATE -> CONFIRMED (e.g. fire=0.65, smoke=0.65).
      - hold_thresholds: Class-specific soft score threshold to maintain ACTIVE state (e.g. fire=0.45, smoke=0.40).
      - active_miss_grace_sec: Grace period for intermittent detection misses while active (e.g. 1.5s).
      - resolved_after_sec: Continuous absence required to declare incident resolved (e.g. 5.0s).
    """

    def __init__(
        self,
        confirm_duration_sec: Optional[dict[str, float] | float] = None,
        trigger_thresholds: Optional[dict[str, float] | float] = None,
        hold_thresholds: Optional[dict[str, float] | float] = None,
        active_miss_grace_sec: float = 1.5,
        resolved_after_sec: float = 5.0,
    ):
        if isinstance(confirm_duration_sec, dict):
            self.confirm_duration_sec = confirm_duration_sec
        elif isinstance(confirm_duration_sec, (int, float)):
            self.confirm_duration_sec = {"fire": float(confirm_duration_sec), "smoke": float(confirm_duration_sec)}
        else:
            self.confirm_duration_sec = {"fire": 0.8, "smoke": 2.0}

        if isinstance(trigger_thresholds, dict):
            self.trigger_thresholds = trigger_thresholds
        elif isinstance(trigger_thresholds, (int, float)):
            self.trigger_thresholds = {"fire": float(trigger_thresholds), "smoke": float(trigger_thresholds)}
        else:
            self.trigger_thresholds = {"fire": 0.50, "smoke": 0.45}

        if isinstance(hold_thresholds, dict):
            self.hold_thresholds = hold_thresholds
        elif isinstance(hold_thresholds, (int, float)):
            self.hold_thresholds = {"fire": float(hold_thresholds), "smoke": float(hold_thresholds)}
        else:
            self.hold_thresholds = {"fire": 0.35, "smoke": 0.30}

        self.active_miss_grace_sec = active_miss_grace_sec
        self.resolved_after_sec = resolved_after_sec

        self.current_state: IncidentState = IncidentState.NONE
        self.active_incident: Optional[IncidentEvent] = None
        self._candidate_start_time: Optional[float] = None
        self._candidate_type: Optional[str] = None
        self._candidate_max_conf: float = 0.0
        self._candidate_max_score: float = 0.0

    def update(
        self,
        detected_items: list[dict],
        timestamp: Optional[float] = None,
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
                primary = detected_items[0]
                event_type = primary.get("type", "hazard_detected")
                cls_key = "fire" if "fire" in event_type else "smoke"

            score = float(primary.get("score", primary.get("confidence", 0.0)))
            conf = float(primary.get("confidence", score))

            target_confirm_duration = self.confirm_duration_sec.get(cls_key, 1.0)
            target_trigger = self.trigger_thresholds.get(cls_key, 0.65)
            target_hold = self.hold_thresholds.get(cls_key, 0.40)

            if self.current_state in (IncidentState.NONE, IncidentState.RESOLVED):
                if score >= target_hold:
                    # Start CANDIDATE phase
                    self.current_state = IncidentState.CANDIDATE
                    self._candidate_start_time = timestamp
                    self._candidate_type = event_type
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
                            "components": primary.get("components", {}),
                        },
                    )

            elif self.current_state == IncidentState.CANDIDATE:
                if score < target_hold:
                    # Candidate score faded below hold threshold -> cancel candidate
                    self.current_state = IncidentState.NONE
                    self._candidate_start_time = None
                    return None

                self._candidate_max_conf = max(self._candidate_max_conf, conf)
                self._candidate_max_score = max(self._candidate_max_score, score)
                cand_start = self._candidate_start_time if self._candidate_start_time is not None else timestamp
                time_in_candidate = timestamp - cand_start

                # Hysteresis confirmation: requires trigger threshold AND sufficient duration
                if time_in_candidate >= target_confirm_duration and score >= target_trigger:
                    # Promoted to CONFIRMED!
                    self.current_state = IncidentState.CONFIRMED
                    inc_id = uuid.uuid4().hex[:8]
                    self.active_incident = IncidentEvent(
                        incident_id=inc_id,
                        event_type=self._candidate_type or event_type,
                        state=IncidentState.CONFIRMED,
                        confidence=self._candidate_max_conf,
                        started_at=cand_start,
                        confirmed_at=timestamp,
                        last_seen=timestamp,
                        ended_at=None,
                        duration_seconds=time_in_candidate,
                        is_new_alert=True,  # Triggers DB insert & SocketIO alert
                        details={
                            "bbox": primary.get("bbox", []),
                            "score": score,
                            "components": primary.get("components", {}),
                        },
                    )
                    return self.active_incident
                else:
                    return IncidentEvent(
                        incident_id="candidate",
                        event_type=self._candidate_type or event_type,
                        state=IncidentState.CANDIDATE,
                        confidence=conf,
                        started_at=cand_start,
                        confirmed_at=None,
                        last_seen=timestamp,
                        ended_at=None,
                        duration_seconds=time_in_candidate,
                        is_new_alert=False,
                        details={
                            "bbox": primary.get("bbox", []),
                            "score": score,
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
                        self.active_incident.details["components"] = primary.get("components", {})
                        return self.active_incident

        else:
            # No detection in current frame
            if self.current_state == IncidentState.CANDIDATE:
                # Transitory noise vanished before reaching confirmation threshold
                self.current_state = IncidentState.NONE
                self._candidate_start_time = None
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
                        self.active_incident = None
                        return resolved_event

        return None

    def reset(self):
        """Reset state machine (e.g. when video stream restarts)."""
        self.current_state = IncidentState.NONE
        self.active_incident = None
        self._candidate_start_time = None
        self._candidate_type = None
        self._candidate_max_conf = 0.0
