"""
app.py - Flask API Backend for PPE Detection System
"""

import logging
import os
import sys
import threading
import time
import uuid
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from config import Config
from models import db, Camera, Violation, Setting

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

socketio = SocketIO()
_active_uploads: dict[str, threading.Event] = {}


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins=["http://localhost:5173", "http://localhost:3000"])

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EVIDENCE_FOLDER"], exist_ok=True)

    from routes.camera_routes import camera_bp
    from routes.violation_routes import violation_bp
    from routes.stats_routes import stats_bp
    from routes.settings_routes import settings_bp
    from routes.exclusion_routes import exclusion_bp, register_zone_listener

    app.register_blueprint(camera_bp)
    app.register_blueprint(violation_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(exclusion_bp)

    _detector = None
    _fall_detector = None
    _fire_model = None
    _detector_lock = threading.Lock()
    _fall_detector_lock = threading.Lock()
    _fire_model_lock = threading.Lock()
    _cameras = {}  # camera_key -> CameraManager

    def get_detector():
        nonlocal _detector
        with _detector_lock:
            if _detector is None:
                from detector import PPEDetector

                logger.info("Loading YOLO model...")
                _detector = PPEDetector()
                logger.info("Model loaded.")
            return _detector

    def get_fall_detector():
        nonlocal _fall_detector
        with _fall_detector_lock:
            if _fall_detector is None:
                from fall_detector import FallDetector

                logger.info("Loading Fall Detection model...")
                _fall_detector = FallDetector()
                logger.info("Fall model loaded.")
            return _fall_detector

    def get_fire_model():
        nonlocal _fire_model
        with _fire_model_lock:
            if _fire_model is None:
                from fire_detector import FireSmokeModel

                logger.info("Loading shared Fire & Smoke YOLO model...")
                _fire_model = FireSmokeModel()
                logger.info("Fire & Smoke shared model loaded.")
            return _fire_model

    def _on_violation(alert_type, confidence, image_path, camera_id, count=1, person_id=None):
        try:
            with app.app_context():
                violation = Violation(
                    camera_id=camera_id if camera_id else None,
                    violation_type=alert_type,
                    confidence=confidence,
                    image_path=image_path,
                )
                db.session.add(violation)
                db.session.commit()

                vtype_labels = {
                    "no_helmet": "Thiếu mũ bảo hiểm",
                    "no_vest": "Thiếu áo bảo hộ",
                    "no_mask": "Thiếu khẩu trang",
                }
                base_label = vtype_labels.get(alert_type, alert_type)
                type_label = f"Công nhân #{person_id}: {base_label}" if person_id else base_label

                socketio.emit("violation_alert", {
                    "id": violation.id,
                    "type": alert_type,
                    "type_label": type_label,
                    "person_id": person_id,
                    "confidence": round(confidence * 100, 1),
                    "image": f"/api/violations/{violation.id}/image",
                    "timestamp": violation.timestamp.isoformat(),
                    "camera_id": camera_id,
                })
        except Exception as e:
            logger.error("Failed to save violation: %s", e)

    def _on_fall(confidence, image_path, camera_id):
        try:
            with app.app_context():
                violation = Violation(
                    camera_id=camera_id if camera_id else None,
                    violation_type="fall_detected",
                    confidence=confidence,
                    image_path=image_path,
                )
                db.session.add(violation)
                db.session.commit()

                socketio.emit("fall_alert", {
                    "id": violation.id,
                    "type": "fall_detected",
                    "type_label": "Phát hiện ngã",
                    "confidence": round(confidence * 100, 1),
                    "image": f"/api/violations/{violation.id}/image",
                    "timestamp": violation.timestamp.isoformat(),
                    "camera_id": camera_id,
                })
        except Exception as e:
            logger.error("Failed to save fall alert: %s", e)

    def _on_zones_updated(changed_cam_id):
        with app.app_context():
            from models import ExclusionZone
            zones = ExclusionZone.query.filter_by(camera_id=changed_cam_id, is_active=True).all()
            zones_data = [z.to_dict() for z in zones]
            for c_key, mgr in list(_cameras.items()):
                if mgr.camera_id == changed_cam_id or (changed_cam_id == 0 and mgr.camera_id in (0, None)):
                    mgr.update_exclusion_zones(zones_data)

    register_zone_listener(_on_zones_updated)

    def _on_fire(alert_type, confidence, image_path, camera_id, is_breakout=False, breakout_zone=None):
        try:
            with app.app_context():
                violation = Violation(
                    camera_id=camera_id if camera_id else None,
                    violation_type=alert_type,
                    confidence=confidence,
                    image_path=image_path,
                )
                db.session.add(violation)
                db.session.commit()

                if is_breakout:
                    type_label = f"NGUY HIỂM: CHÁY LAN RA NGOÀI {breakout_zone or 'VÙNG LOẠI TRỪ'}!"
                elif alert_type == "fire_detected":
                    type_label = "Phát hiện cháy (Hỏa hoạn)"
                else:
                    type_label = "Phát hiện khói (Nguy cơ cháy)"

                socketio.emit("fire_alert", {
                    "id": violation.id,
                    "type": alert_type,
                    "type_label": type_label,
                    "confidence": round(confidence * 100, 1),
                    "image": f"/api/violations/{violation.id}/image",
                    "timestamp": violation.timestamp.isoformat(),
                    "camera_id": camera_id,
                    "is_breakout": is_breakout,
                    "breakout_zone": breakout_zone,
                })
        except Exception as e:
            logger.error("Failed to save fire alert: %s", e)

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "ppe_model_loaded": _detector is not None,
            "fall_model_loaded": _fall_detector is not None,
            "fire_model_loaded": _fire_model is not None,
        })

    @app.route("/api/detect/upload", methods=["POST"])
    def detect_upload():
        if "video" not in request.files:
            return jsonify({"error": "No video file"}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("mp4", "avi", "mov", "mkv"):
            return jsonify({"error": "Unsupported format"}), 400

        filename = secure_filename(file.filename)
        input_name = f"{int(time.time())}_{filename}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_name)
        file.save(input_path)

        # Tự động hủy toàn bộ tiến trình phân tích video cũ còn đang chạy để giải phóng GPU/CPU
        for old_uid, old_event in list(_active_uploads.items()):
            old_event.set()
            logger.info("Auto-cancelled previous active upload before starting new one: %s", old_uid)
        _active_uploads.clear()

        frame_skip = int(request.form.get("frame_skip", 2))
        upload_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        _active_uploads[upload_id] = cancel_event

        def _get_form_bool(key, default=True):
            val = request.form.get(key)
            if val is not None:
                return str(val).lower() in ("true", "1", "yes", "on")
            s = Setting.query.filter_by(key=key).first()
            if s:
                return s.value.lower() in ("true", "1", "yes", "on")
            return default

        enable_ppe = _get_form_bool("enable_ppe", True)
        enable_fall = _get_form_bool("enable_fall", True)
        enable_fire = _get_form_bool("enable_fire", True)

        try:
            imgsz = int(request.form.get("imgsz", 960))
        except (ValueError, TypeError):
            imgsz = 960

        raw_zones = request.form.get("exclusion_zones")
        exclusion_zones = []
        if raw_zones:
            import json
            try:
                exclusion_zones = json.loads(raw_zones) if isinstance(raw_zones, str) else raw_zones
            except Exception as e:
                logger.warning("Could not parse upload exclusion_zones: %s", e)

        def _run_detection():
            try:
                from fire_detector import FireSmokeStreamAnalyzer

                detector = get_detector()
                fall_det = get_fall_detector() if enable_fall else None
                fire_stream_analyzer = None
                if enable_fire:
                    shared_fire_model = get_fire_model()
                    fire_stream_analyzer = FireSmokeStreamAnalyzer(
                        model=shared_fire_model,
                        exclusion_zones=exclusion_zones,
                    )

                output_name = f"result_{input_name.rsplit('.', 1)[0]}.mp4"
                output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)

                def on_progress(processed, total, elapsed):
                    remaining = (elapsed / processed) * (total - processed) if processed > 0 else 0
                    socketio.emit("upload_progress", {
                        "upload_id": upload_id,
                        "processed": processed,
                        "total": total,
                        "percent": round(processed / total * 100, 1) if total else 0,
                        "elapsed": round(elapsed, 1),
                        "remaining": round(remaining, 1),
                    })

                stats = detector.process_video(
                    input_path,
                    output_path,
                    frame_skip=frame_skip,
                    fall_detector=fall_det,
                    fire_detector=fire_stream_analyzer,
                    enable_ppe=enable_ppe,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                    imgsz=imgsz,
                )

                if cancel_event.is_set():
                    socketio.emit("upload_done", {"upload_id": upload_id, "cancelled": True})
                else:
                    socketio.emit("upload_done", {
                        "upload_id": upload_id,
                        "cancelled": False,
                        "video_url": f"/static/outputs/{output_name}",
                        "output_video": output_name,
                        "stats": stats,
                    })
            except Exception as e:
                logger.exception("Detection error")
                socketio.emit("upload_done", {"upload_id": upload_id, "error": str(e)})
            finally:
                _active_uploads.pop(upload_id, None)

        socketio.start_background_task(_run_detection)
        return jsonify({"upload_id": upload_id})

    @app.route("/api/detect/output/<path:filename>")
    def detect_output(filename):
        from flask import send_from_directory
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

    @app.route("/api/detect/cancel", methods=["POST"])
    def detect_cancel():
        data = request.get_json(silent=True) or {}
        upload_id = data.get("upload_id")
        cancelled_count = 0
        if upload_id and upload_id in _active_uploads:
            _active_uploads[upload_id].set()
            cancelled_count += 1
            logger.info("Cancelled specific upload: %s", upload_id)
        else:
            # Hủy tất cả các tiến trình upload video đang chạy nếu không truyền upload_id cụ thể
            for uid, ev in list(_active_uploads.items()):
                ev.set()
                cancelled_count += 1
                logger.info("Cancelled running upload: %s", uid)
        return jsonify({"ok": True, "cancelled_count": cancelled_count})

    @app.route("/api/webcam_stream")
    def webcam_stream():
        from camera_manager import CameraManager

        camera_id = request.args.get("camera_id", 0, type=int)
        frame_skip = int(request.args.get("frame_skip", 4))

        def _get_query_bool(key, default=True):
            val = request.args.get(key)
            if val is not None:
                return str(val).lower() in ("true", "1", "yes", "on")
            s = Setting.query.filter_by(key=key).first()
            if s:
                return s.value.lower() in ("true", "1", "yes", "on")
            return default

        enable_ppe = _get_query_bool("enable_ppe", True)
        enable_fall = _get_query_bool("enable_fall", True)
        enable_fire = _get_query_bool("enable_fire", True)

        if camera_id:
            camera = Camera.query.get(camera_id)
            source = camera.source if camera else "0"
        else:
            source = "0"

        cam_key = f"{source}_{frame_skip}"

        from models import ExclusionZone
        initial_zones = ExclusionZone.query.filter_by(camera_id=camera_id, is_active=True).all()
        zones_data = [z.to_dict() for z in initial_zones]

        if cam_key not in _cameras or not _cameras[cam_key].is_running():
            manager = CameraManager(
                source=source,
                detector=get_detector(),
                fall_detector=get_fall_detector(),
                fire_model=get_fire_model(),
                frame_skip=frame_skip,
                camera_id=camera_id,
                evidence_folder=app.config["EVIDENCE_FOLDER"],
                on_violation=_on_violation,
                on_fall=_on_fall,
                on_fire=_on_fire,
                enable_ppe=enable_ppe,
                enable_fall=enable_fall,
                enable_fire=enable_fire,
                exclusion_zones=zones_data,
            )
            manager.start()
            _cameras[cam_key] = manager
        else:
            _cameras[cam_key].update_toggles(
                enable_ppe=enable_ppe,
                enable_fall=enable_fall,
                enable_fire=enable_fire,
            )
            _cameras[cam_key].update_exclusion_zones(zones_data)

        def generate():
            manager = _cameras.get(cam_key)
            if not manager:
                return
            none_count = 0
            try:
                while manager.is_running():
                    frame = manager.get_frame()
                    if frame is None:
                        none_count += 1
                        if none_count >= 3:
                            break
                        continue
                    none_count = 0
                    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
            except (GeneratorExit, BrokenPipeError):
                pass

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/webcam_toggles", methods=["POST"])
    def webcam_toggles():
        data = request.get_json() or {}
        for manager in _cameras.values():
            manager.update_toggles(
                enable_ppe=data.get("enable_ppe"),
                enable_fall=data.get("enable_fall"),
                enable_fire=data.get("enable_fire"),
            )
        return jsonify({"ok": True})

    @app.route("/api/webcam_stop", methods=["POST"])
    def webcam_stop():
        from camera_manager import CameraManager

        data = request.get_json() or {}
        cam_key = data.get("cam_key")

        if cam_key and cam_key in _cameras:
            _cameras[cam_key].stop()
            del _cameras[cam_key]
            return jsonify({"ok": True, "message": f"Camera {cam_key} stopped"})

        for key, manager in list(_cameras.items()):
            manager.stop()
        _cameras.clear()
        return jsonify({"ok": True, "message": "All cameras stopped"})

    @app.route("/api/webcam_status")
    def webcam_status():
        return jsonify({
            "running": len(_cameras),
            "cameras": {k: v.is_running() for k, v in _cameras.items()},
        })

    def _get_stats_dict():
        today = date.today()
        total_today = Violation.query.filter(
            db.func.date(Violation.timestamp) == today
        ).count()
        return {"total_today": total_today}

    @app.route("/api/violations/<int:violation_id>/image")
    def violation_image(violation_id):
        from flask import send_file

        v = Violation.query.get_or_404(violation_id)
        if v.image_path and os.path.exists(v.image_path):
            return send_file(v.image_path, mimetype="image/jpeg")
        return jsonify({"error": "Image not found"}), 404

    return app


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully.")

    logger.info("Starting server at http://localhost:5000")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
