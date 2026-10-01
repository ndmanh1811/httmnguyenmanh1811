"""
iot_routes.py - Endpoints for IoT sensor and alarm control
"""

from flask import Blueprint, jsonify, request
from iot_service import get_iot_status, set_alarm_state

iot_bp = Blueprint("iot", __name__)


@iot_bp.route("/api/iot/status", methods=["GET"])
def iot_status():
    return jsonify(get_iot_status())


@iot_bp.route("/api/iot/alarm", methods=["POST"])
def iot_alarm():
    data = request.get_json(silent=True) or {}
    # Accept boolean True/False or string "ON"/"OFF"
    raw_state = data.get("state", False)
    if isinstance(raw_state, str):
        state = raw_state.upper() in ("ON", "1", "TRUE")
    else:
        state = bool(raw_state)

    actual_state = set_alarm_state(state)
    return jsonify({"ok": True, "alarm_state": actual_state})
