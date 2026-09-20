"""
config.py - Application configuration
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    SQLITE_PATH = os.path.join(BASE_DIR, "ppe_detection.db")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'ppe_detection.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
    EVIDENCE_FOLDER = os.path.join(BASE_DIR, "static", "evidence")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB
