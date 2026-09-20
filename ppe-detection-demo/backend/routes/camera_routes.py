"""
camera_routes.py - Camera CRUD endpoints
"""

from flask import Blueprint, jsonify, request

from models import db, Camera

camera_bp = Blueprint("camera", __name__)


@camera_bp.route("/api/cameras", methods=["GET"])
def get_cameras():
    cameras = Camera.query.all()
    return jsonify([
        {
            "id": c.id,
            "name": c.name,
            "source": c.source,
            "location": c.location,
            "is_active": c.is_active,
        }
        for c in cameras
    ])


@camera_bp.route("/api/cameras", methods=["POST"])
def add_camera():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    source = data.get("source", "0").strip()
    location = data.get("location", "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400

    camera = Camera(name=name, source=source, location=location)
    db.session.add(camera)
    db.session.commit()
    return jsonify({"id": camera.id, "name": camera.name, "source": camera.source, "location": camera.location}), 201


@camera_bp.route("/api/cameras/<int:camera_id>", methods=["PUT"])
def update_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    data = request.get_json() or {}

    if "name" in data:
        camera.name = data["name"]
    if "source" in data:
        camera.source = data["source"]
    if "location" in data:
        camera.location = data["location"]
    if "is_active" in data:
        camera.is_active = data["is_active"]

    db.session.commit()
    return jsonify({"id": camera.id, "name": camera.name, "source": camera.source})


@camera_bp.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    db.session.delete(camera)
    db.session.commit()
    return jsonify({"ok": True})
