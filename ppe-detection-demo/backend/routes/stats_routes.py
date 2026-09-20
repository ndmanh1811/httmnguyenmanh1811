"""
stats_routes.py - Statistics endpoints
"""

from datetime import date, timedelta

from flask import Blueprint, jsonify, request

from models import db, Violation, Camera

stats_bp = Blueprint("stats", __name__)


@stats_bp.route("/api/stats/summary", methods=["GET"])
def get_summary():
    today = date.today()

    total_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today
    ).count()

    helmet_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "no_helmet",
    ).count()

    vest_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "no_vest",
    ).count()

    mask_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "no_mask",
    ).count()

    fall_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "fall_detected",
    ).count()

    fire_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "fire_detected",
    ).count()

    smoke_today = Violation.query.filter(
        db.func.date(Violation.timestamp) == today,
        Violation.violation_type == "smoke_detected",
    ).count()

    total_all = Violation.query.count()

    active_cameras = Camera.query.filter_by(is_active=True).count()

    return jsonify({
        "total_today": total_today,
        "helmet_today": helmet_today,
        "vest_today": vest_today,
        "mask_today": mask_today,
        "fall_today": fall_today,
        "fire_today": fire_today,
        "smoke_today": smoke_today,
        "total_all": total_all,
        "active_cameras": active_cameras,
    })


@stats_bp.route("/api/stats/recent", methods=["GET"])
def get_recent_violations():
    limit = request.args.get("limit", 10, type=int)
    violations = Violation.query.order_by(Violation.timestamp.desc()).limit(limit).all()

    return jsonify([
        {
            "id": v.id,
            "type": v.violation_type,
            "confidence": round(v.confidence * 100, 1) if v.confidence else 0,
            "image": f"/api/violations/{v.id}/image",
            "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            "camera_id": v.camera_id,
        }
        for v in violations
    ])
