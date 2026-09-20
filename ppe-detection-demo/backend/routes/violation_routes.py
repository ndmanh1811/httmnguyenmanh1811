"""
violation_routes.py - Violation history endpoints
"""

from collections import defaultdict
from datetime import date, timedelta

from flask import Blueprint, jsonify, request

from models import db, Violation

violation_bp = Blueprint("violation", __name__)


@violation_bp.route("/api/violations", methods=["GET"])
def get_violations():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    vtype = request.args.get("type", "")

    query = Violation.query.order_by(Violation.timestamp.desc())

    if vtype:
        query = query.filter(Violation.violation_type == vtype)

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "violations": [
            {
                "id": v.id,
                "type": v.violation_type,
                "confidence": round(v.confidence * 100, 1) if v.confidence else 0,
                "image": f"/api/violations/{v.id}/image",
                "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                "camera_id": v.camera_id,
            }
            for v in paginated.items
        ],
        "total": paginated.total,
        "pages": paginated.pages,
        "current_page": page,
    })


@violation_bp.route("/api/violations/stats/hourly", methods=["GET"])
def get_hourly_violations():
    today = date.today()
    start = today
    end = today + timedelta(days=1)

    violations = Violation.query.filter(
        Violation.timestamp >= start,
        Violation.timestamp < end,
    ).all()

    hourly = defaultdict(lambda: {"count": 0, "fall_count": 0, "fire_count": 0, "smoke_count": 0})
    for v in violations:
        hour = v.timestamp.hour
        hourly[hour]["count"] += 1
        if v.violation_type == "fall_detected":
            hourly[hour]["fall_count"] += 1
        elif v.violation_type == "fire_detected":
            hourly[hour]["fire_count"] += 1
        elif v.violation_type == "smoke_detected":
            hourly[hour]["smoke_count"] += 1

    result = []
    for h in range(24):
        result.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "count": hourly[h]["count"],
            "fall_count": hourly[h]["fall_count"],
            "fire_count": hourly[h]["fire_count"],
            "smoke_count": hourly[h]["smoke_count"],
        })

    return jsonify(result)


@violation_bp.route("/api/violations/stats/daily", methods=["GET"])
def get_daily_violations():
    days = request.args.get("days", 7, type=int)
    today = date.today()

    result = []
    for d in range(days - 1, -1, -1):
        day = today - timedelta(days=d)
        next_day = day + timedelta(days=1)
        count = Violation.query.filter(
            Violation.timestamp >= day,
            Violation.timestamp < next_day,
        ).count()
        fall_count = Violation.query.filter(
            Violation.timestamp >= day,
            Violation.timestamp < next_day,
            Violation.violation_type == "fall_detected",
        ).count()
        fire_count = Violation.query.filter(
            Violation.timestamp >= day,
            Violation.timestamp < next_day,
            Violation.violation_type == "fire_detected",
        ).count()
        smoke_count = Violation.query.filter(
            Violation.timestamp >= day,
            Violation.timestamp < next_day,
            Violation.violation_type == "smoke_detected",
        ).count()
        result.append({
            "date": day.isoformat(),
            "count": count,
            "fall_count": fall_count,
            "fire_count": fire_count,
            "smoke_count": smoke_count,
        })

    return jsonify(result)
