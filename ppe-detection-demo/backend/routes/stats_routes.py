"""
stats_routes.py - Statistics endpoints
"""

from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify, request

from models import db, Violation, Camera

stats_bp = Blueprint("stats", __name__)


@stats_bp.route("/api/stats/summary", methods=["GET"])
def get_summary():
    # Tinh thoi diem bat dau ngay hom nay theo gio dia phuong (UTC+7)
    now_local = datetime.now()
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Vi timestamp trong SQLite luu theo UTC naive datetime
    start_utc = today_start_local.astimezone(timezone.utc).replace(tzinfo=None)
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)

    # Uu tien dem tu dau ngay hom nay (gio dia phuong)
    today_filter = (Violation.timestamp >= start_utc)
    total_today = Violation.query.filter(today_filter).count()

    # Neu sang som chua co vi pham moi, lay cua 24h qua de hien thi thong so y nghia
    if total_today == 0:
        today_filter = (Violation.timestamp >= twenty_four_hours_ago)
        total_today = Violation.query.filter(today_filter).count()

    helmet_today = Violation.query.filter(
        today_filter,
        Violation.violation_type == "no_helmet",
    ).count()

    vest_today = Violation.query.filter(
        today_filter,
        Violation.violation_type == "no_vest",
    ).count()

    mask_today = Violation.query.filter(
        today_filter,
        Violation.violation_type == "no_mask",
    ).count()

    fall_today = Violation.query.filter(
        today_filter,
        Violation.violation_type.in_(["fall_detected", "fall_immobile"]),
    ).count()

    fire_today = Violation.query.filter(
        today_filter,
        Violation.violation_type == "fire_detected",
    ).count()

    smoke_today = Violation.query.filter(
        today_filter,
        Violation.violation_type == "smoke_detected",
    ).count()

    total_all = Violation.query.count()

    active_cameras = Camera.query.filter_by(is_active=True).count()
    if active_cameras == 0:
        active_cameras = 1  # Camera truc tiep mac dinh (Webcam 0)

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
