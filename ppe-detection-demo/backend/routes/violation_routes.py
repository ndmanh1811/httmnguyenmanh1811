"""
violation_routes.py - Violation history endpoints
"""

from collections import defaultdict
from datetime import datetime, timezone, timedelta, date

from flask import Blueprint, jsonify, request

from models import db, Violation

violation_bp = Blueprint("violation", __name__)


def _parse_date(date_str: str) -> datetime | None:
    """Parse date string in format YYYY-MM-DD to UTC datetime at start of day."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return datetime.combine(d, datetime.min.time())
    except Exception:
        return None


@violation_bp.route("/api/violations", methods=["GET"])
def get_violations():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 15, type=int)
    vtype = request.args.get("type", "").strip()
    time_range = request.args.get("time_range", "all").strip()
    camera_id = request.args.get("camera_id", None, type=int)
    from_date = request.args.get("from_date", "").strip()
    to_date = request.args.get("to_date", "").strip()

    query = Violation.query.order_by(Violation.timestamp.desc())

    # 1. Loc theo loai su co / vi pham
    if vtype:
        if vtype in ("fall", "fall_detected"):
            query = query.filter(Violation.violation_type.in_(["fall_detected", "fall_immobile"]))
        else:
            query = query.filter(Violation.violation_type == vtype)

    # 2. Loc theo moc thoi gian (quick filters)
    now_utc = datetime.utcnow()
    if time_range == "today":
        now_local = datetime.now()
        today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = today_start_local.astimezone(timezone.utc).replace(tzinfo=None)
        query = query.filter(Violation.timestamp >= start_utc)
    elif time_range == "24h":
        query = query.filter(Violation.timestamp >= now_utc - timedelta(hours=24))
    elif time_range == "7d":
        query = query.filter(Violation.timestamp >= now_utc - timedelta(days=7))
    elif time_range == "30d":
        query = query.filter(Violation.timestamp >= now_utc - timedelta(days=30))

    # 3. Loc theo khoang ngay tu-chon (custom date range) - override time_range neu co
    if from_date:
        start_dt = _parse_date(from_date)
        if start_dt:
            query = query.filter(Violation.timestamp >= start_dt)
    if to_date:
        end_dt = _parse_date(to_date)
        if end_dt:
            # include entire end day: < next day 00:00
            end_dt_next = end_dt + timedelta(days=1)
            query = query.filter(Violation.timestamp < end_dt_next)

    # 4. Loc theo camera
    if camera_id is not None:
        query = query.filter(Violation.camera_id == camera_id)

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
    # Lay phan bo trong 24 gio qua
    start = datetime.utcnow() - timedelta(hours=24)

    violations = Violation.query.filter(
        Violation.timestamp >= start,
    ).all()

    hourly = defaultdict(lambda: {"count": 0, "fall_count": 0, "fire_count": 0, "smoke_count": 0})
    for v in violations:
        # Chuyen gio sang local gio Viet Nam (+7)
        hour = (v.timestamp.hour + 7) % 24 if v.timestamp else 0
        hourly[hour]["count"] += 1
        if v.violation_type in ("fall_detected", "fall_immobile"):
            hourly[hour]["fall_count"] += 1
        elif v.violation_type == "fire_detected":
            hourly[hour]["fire_count"] += 1
        elif v.violation_type == "smoke_detected":
            hourly[hour]["smoke_count"] += 1

    return jsonify([
        {
            "hour": h,
            "label": f"{h:02d}:00",
            "count": hourly[h]["count"],
            "fall_count": hourly[h]["fall_count"],
            "fire_count": hourly[h]["fire_count"],
            "smoke_count": hourly[h]["smoke_count"],
        }
        for h in range(24)
    ])


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
