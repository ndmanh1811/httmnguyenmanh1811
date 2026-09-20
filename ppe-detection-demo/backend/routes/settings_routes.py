"""
settings_routes.py - Settings CRUD endpoints
"""

from flask import Blueprint, jsonify, request

from models import db, Setting

settings_bp = Blueprint("settings", __name__)

DEFAULT_SETTINGS = {
    "enable_ppe": {"value": "true", "description": "Bật/Tắt module phát hiện trang bị bảo hộ (PPE)"},
    "enable_fall": {"value": "true", "description": "Bật/Tắt module theo dõi và phát hiện ngã"},
    "enable_fire": {"value": "true", "description": "Bật/Tắt module phát hiện cháy và khói"},
    "confidence_threshold": {"value": "0.35", "description": "Ngưỡng confidence PPE detection"},
    "frame_skip": {"value": "4", "description": "Bỏ qua mỗi N frame khi detect"},
    "tick_interval": {"value": "10", "description": "Thời gian giữa các tick (giây)"},
    "confirm_ticks": {"value": "3", "description": "Số tick cần thiết để xác nhận vi phạm"},
    "fall_angle_threshold": {"value": "60.0", "description": "Góc tối thiểu phát hiện ngã (độ)"},
    "fall_consecutive_frames": {"value": "8", "description": "Số frame liên tiếp để xác nhận ngã"},
}


@settings_bp.route("/api/settings", methods=["GET"])
def get_settings():
    settings = Setting.query.all()
    result = {}
    for s in settings:
        result[s.key] = {"value": s.value, "description": s.description}

    for key, default in DEFAULT_SETTINGS.items():
        if key not in result:
            result[key] = default

    return jsonify(result)


@settings_bp.route("/api/settings", methods=["PUT"])
def update_settings():
    data = request.get_json() or {}

    for key, val in data.items():
        setting = Setting.query.filter_by(key=key).first()
        if setting:
            setting.value = str(val)
        else:
            desc = DEFAULT_SETTINGS.get(key, {}).get("description", "")
            setting = Setting(key=key, value=str(val), description=desc)
            db.session.add(setting)

    db.session.commit()
    return jsonify({"ok": True})

