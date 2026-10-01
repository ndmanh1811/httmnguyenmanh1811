"""
iot_service.py - Background service connecting MQTT Broker to Flask & Socket.IO
"""

import json
import logging
import threading
import time
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

TOPIC_SENSOR = "ppe/iot/sensor"
TOPIC_ALARM = "ppe/iot/alarm"

_state = {
    "temp": None,
    "hum": None,
    "temp_limit": 40.0,
    "is_over_limit": False,
    "alarm_state": False,
    "last_update": None,
    "connected": False,
}

_mqtt_client = None
_app = None
_socketio = None
_lock = threading.Lock()


def get_iot_status():
    with _lock:
        return dict(_state)


def set_alarm_state(state: bool):
    """Publish ON/OFF to MQTT topic ppe/iot/alarm and notify clients."""
    global _mqtt_client, _socketio
    cmd = "ON" if state else "OFF"
    with _lock:
        _state["alarm_state"] = state

    if _mqtt_client and _mqtt_client.is_connected():
        try:
            _mqtt_client.publish(TOPIC_ALARM, cmd)
            logger.info("Published alarm command to MQTT: %s", cmd)
        except Exception as e:
            logger.error("Failed to publish alarm to MQTT: %s", e)

    if _socketio:
        try:
            _socketio.emit("iot_alarm_state", {"alarm_state": state, "cmd": cmd})
        except Exception as e:
            logger.error("Failed to emit iot_alarm_state: %s", e)

    return state


def _on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT Broker successfully (127.0.0.1:1883).")
        with _lock:
            _state["connected"] = True
        client.subscribe(TOPIC_SENSOR)
        logger.info("Subscribed to MQTT topic: %s", TOPIC_SENSOR)
    else:
        logger.error("MQTT connection failed with code: %s", rc)
        with _lock:
            _state["connected"] = False


def _on_disconnect(client, userdata, flags, rc, properties=None):
    logger.warning("Disconnected from MQTT Broker (rc=%s)", rc)
    with _lock:
        _state["connected"] = False
        _state["temp"] = None
        _state["hum"] = None
    if _socketio:
        try:
            _socketio.emit(
                "iot_sensor_data",
                {
                    "temp": None,
                    "hum": None,
                    "connected": False,
                    "is_over_limit": False,
                    "alarm_state": _state["alarm_state"],
                },
            )
        except Exception:
            pass


def _on_message(client, userdata, msg):
    global _app, _socketio
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
        temp = float(data.get("temp", 0.0))
        hum = float(data.get("hum", 0.0))

        temp_limit = 40.0
        auto_alarm = False

        if _app:
            with _app.app_context():
                try:
                    from models import Setting
                    s_limit = Setting.query.filter_by(key="temp_limit").first()
                    if s_limit and s_limit.value:
                        temp_limit = float(s_limit.value)

                    s_auto = Setting.query.filter_by(key="auto_alarm_on_overheat").first()
                    if s_auto and s_auto.value:
                        auto_alarm = s_auto.value.lower() in ("true", "1", "yes", "on")
                except Exception as ex:
                    logger.debug("Could not read temp_limit setting from DB: %s", ex)

        is_over = temp > temp_limit

        with _lock:
            _state["temp"] = temp
            _state["hum"] = hum
            _state["temp_limit"] = temp_limit
            _state["is_over_limit"] = is_over
            _state["last_update"] = time.time()
            current_alarm = _state["alarm_state"]

        # If auto alarm is enabled, toggle alarm if over limit
        if auto_alarm and is_over and not current_alarm:
            set_alarm_state(True)

        if _socketio:
            _socketio.emit(
                "iot_sensor_data",
                {
                    "temp": temp,
                    "hum": hum,
                    "temp_limit": temp_limit,
                    "is_over_limit": is_over,
                    "alarm_state": _state["alarm_state"],
                    "connected": True,
                    "timestamp": time.time(),
                },
            )

    except Exception as e:
        logger.error("Error processing MQTT message on %s: %s", msg.topic, e)


def _watchdog_loop():
    """Check every 4s if sensor data stopped arriving for > 10s."""
    while True:
        time.sleep(4)
        with _lock:
            last = _state["last_update"]
            temp = _state["temp"]
            if temp is not None and last and (time.time() - last > 10):
                _state["temp"] = None
                _state["hum"] = None
                _state["is_over_limit"] = False
                if _socketio:
                    try:
                        _socketio.emit(
                            "iot_sensor_data",
                            {
                                "temp": None,
                                "hum": None,
                                "connected": False,
                                "is_over_limit": False,
                                "alarm_state": _state["alarm_state"],
                            },
                        )
                    except Exception:
                        pass


def init_iot_service(app, socketio):
    """Initialize MQTT background client for the Flask app."""
    global _mqtt_client, _app, _socketio
    _app = app
    _socketio = socketio

    try:
        # Use CallbackAPIVersion.VERSION2 for paho-mqtt 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect
        client.on_message = _on_message

        client.connect_async("127.0.0.1", 1883, 60)
        client.loop_start()
        _mqtt_client = client
        logger.info("IoT MQTT background loop started.")

        watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
        watchdog.start()
    except Exception as e:
        logger.error("Failed to initialize IoT MQTT service: %s", e)
