"""
exclusion_routes.py - REST CRUD endpoints for Camera Exclusion Zones (ROI)
"""

import json
import logging
from flask import Blueprint, jsonify, request

from models import db, ExclusionZone

logger = logging.getLogger(__name__)

exclusion_bp = Blueprint("exclusion", __name__)

_zone_change_listeners = []


def register_zone_listener(callback):
    """Registers a callback(camera_id) invoked whenever exclusion zones change."""
    _zone_change_listeners.append(callback)


def _notify_listeners(camera_id):
    for cb in _zone_change_listeners:
        try:
            cb(camera_id)
        except Exception as e:
            logger.error("Error invoking zone listener for cam %s: %s", camera_id, e)


@exclusion_bp.route("/api/exclusion_zones", methods=["GET"])
def get_all_exclusion_zones():
    camera_id = request.args.get("camera_id", type=int)
    query = ExclusionZone.query
    if camera_id is not None:
        query = query.filter_by(camera_id=camera_id)
    zones = query.order_by(ExclusionZone.id.asc()).all()
    return jsonify([z.to_dict() for z in zones])


@exclusion_bp.route("/api/cameras/<int:camera_id>/exclusion_zones", methods=["GET"])
def get_camera_exclusion_zones(camera_id):
    zones = ExclusionZone.query.filter_by(camera_id=camera_id).order_by(ExclusionZone.id.asc()).all()
    return jsonify([z.to_dict() for z in zones])


@exclusion_bp.route("/api/cameras/<int:camera_id>/exclusion_zones", methods=["POST"])
@exclusion_bp.route("/api/exclusion_zones", methods=["POST"])
def add_exclusion_zone(camera_id=None):
    data = request.get_json() or {}
    cam_id = camera_id if camera_id is not None else data.get("camera_id", 0)
    name = data.get("name", "").strip()
    zone_type = data.get("zone_type", "welding").strip()
    polygon_points = data.get("polygon_points", [])
    is_active = data.get("is_active", True)

    if not name:
        return jsonify({"error": "Tên vùng không được để trống"}), 400

    if not isinstance(polygon_points, list) or len(polygon_points) < 3:
        return jsonify({"error": "Đa giác vùng loại trừ cần ít nhất 3 điểm"}), 400

    cleaned_points = []
    for pt in polygon_points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            cleaned_points.append([round(x, 4), round(y, 4)])
        else:
            return jsonify({"error": "Định dạng tọa độ điểm không hợp lệ"}), 400

    zone = ExclusionZone(
        camera_id=cam_id,
        name=name,
        zone_type=zone_type,
        polygon_points=json.dumps(cleaned_points),
        is_active=bool(is_active),
    )
    db.session.add(zone)
    db.session.commit()

    _notify_listeners(cam_id)
    return jsonify(zone.to_dict()), 201


@exclusion_bp.route("/api/exclusion_zones/<int:zone_id>", methods=["PUT"])
def update_exclusion_zone(zone_id):
    zone = ExclusionZone.query.get_or_404(zone_id)
    data = request.get_json() or {}

    if "name" in data:
        name = data["name"].strip()
        if not name:
            return jsonify({"error": "Tên vùng không được để trống"}), 400
        zone.name = name

    if "zone_type" in data:
        zone.zone_type = data["zone_type"].strip()

    if "is_active" in data:
        zone.is_active = bool(data["is_active"])

    if "polygon_points" in data:
        polygon_points = data["polygon_points"]
        if not isinstance(polygon_points, list) or len(polygon_points) < 3:
            return jsonify({"error": "Đa giác vùng loại trừ cần ít nhất 3 điểm"}), 400
        cleaned_points = []
        for pt in polygon_points:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            cleaned_points.append([round(x, 4), round(y, 4)])
        zone.polygon_points = json.dumps(cleaned_points)

    db.session.commit()
    _notify_listeners(zone.camera_id)
    return jsonify(zone.to_dict())


@exclusion_bp.route("/api/exclusion_zones/<int:zone_id>", methods=["DELETE"])
def delete_exclusion_zone(zone_id):
    zone = ExclusionZone.query.get_or_404(zone_id)
    cam_id = zone.camera_id
    db.session.delete(zone)
    db.session.commit()

    _notify_listeners(cam_id)
    return jsonify({"ok": True, "deleted_id": zone_id})
