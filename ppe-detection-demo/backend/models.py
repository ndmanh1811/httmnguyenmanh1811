"""
models.py - SQLAlchemy models for PPE Detection System
"""

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Camera(db.Model):
    __tablename__ = "cameras"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    source = db.Column(db.String(255), nullable=False, default="0")
    location = db.Column(db.String(200), default="")
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Violation(db.Model):
    __tablename__ = "violations"

    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey("cameras.id"), nullable=True)
    violation_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, default=0.0)
    image_path = db.Column(db.String(500), default="")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class DailyStats(db.Model):
    __tablename__ = "daily_stats"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False)
    total_violations = db.Column(db.Integer, default=0)
    helmet_violations = db.Column(db.Integer, default=0)
    vest_violations = db.Column(db.Integer, default=0)
    mask_violations = db.Column(db.Integer, default=0)
    fall_count = db.Column(db.Integer, default=0)


class Setting(db.Model):
    __tablename__ = "settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(500), default="")
    description = db.Column(db.String(500), default="")


class ExclusionZone(db.Model):
    __tablename__ = "exclusion_zones"

    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey("cameras.id"), nullable=True, index=True)
    name = db.Column(db.String(100), nullable=False)
    zone_type = db.Column(db.String(50), default="welding")  # welding, kitchen, boiler, smoking, other
    polygon_points = db.Column(db.Text, nullable=False)      # JSON list: [[x, y], ...] normalized [0.0 - 1.0]
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        import json
        try:
            points = json.loads(self.polygon_points) if isinstance(self.polygon_points, str) else self.polygon_points
        except Exception:
            points = []
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "name": self.name,
            "zone_type": self.zone_type,
            "polygon_points": points,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

