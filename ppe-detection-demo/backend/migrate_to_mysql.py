"""
migrate_to_mysql.py
-------------------
Migrates schema and data from SQLite (ppe_detection.db) to MySQL XAMPP (ppe_detection),
with full schema reconciliation to ensure 100% compatibility with models.py.
"""

import os
import sys
import sqlite3
import pymysql
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(BASE_DIR, "ppe_detection.db")

MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "ppe_detection"


def reconcile_schema(conn):
    """Align legacy MySQL schema to match current models.py definitions."""
    with conn.cursor() as cur:
        # 1. Cameras: ensure is_active exists
        cur.execute("SHOW COLUMNS FROM `cameras` LIKE 'is_active'")
        if not cur.fetchone():
            print("[*] Adding `is_active` to `cameras` table...")
            cur.execute("ALTER TABLE `cameras` ADD COLUMN `is_active` TINYINT(1) DEFAULT 1")
            cur.execute("SHOW COLUMNS FROM `cameras` LIKE 'status'")
            if cur.fetchone():
                cur.execute("UPDATE `cameras` SET `is_active` = (status = 'active')")
        conn.commit()

        # 2. DailyStats: check if legacy stat_date column is present
        cur.execute("SHOW TABLES LIKE 'daily_stats'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM `daily_stats` LIKE 'stat_date'")
            if cur.fetchone():
                print("[*] Recreating `daily_stats` to match modern models.py schema...")
                cur.execute("DROP TABLE `daily_stats`")
                conn.commit()

        # 3. Settings: rename setting_key -> key, setting_value -> value
        cur.execute("SHOW TABLES LIKE 'settings'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM `settings` LIKE 'setting_key'")
            if cur.fetchone():
                print("[*] Updating `settings` column names to `key` and `value`...")
                cur.execute("ALTER TABLE `settings` CHANGE `setting_key` `key` VARCHAR(100) NOT NULL")
                cur.execute("ALTER TABLE `settings` CHANGE `setting_value` `value` VARCHAR(500) DEFAULT ''")
                cur.execute("ALTER TABLE `settings` MODIFY `description` VARCHAR(500) DEFAULT ''")
                conn.commit()

        # 4. Violations: widen violation_type from enum to VARCHAR(50)
        cur.execute("SHOW TABLES LIKE 'violations'")
        if cur.fetchone():
            cur.execute("SHOW COLUMNS FROM `violations` LIKE 'violation_type'")
            col = cur.fetchone()
            if col and "enum" in col["Type"].lower():
                print("[*] Widening `violations.violation_type` enum to VARCHAR(50)...")
                cur.execute("ALTER TABLE `violations` MODIFY `violation_type` VARCHAR(50) NOT NULL")
                conn.commit()


def migrate():
    print(f"[*] Checking MySQL connection on {MYSQL_HOST}:{MYSQL_PORT}...")
    # 1. Connect to MySQL server (without DB first) to ensure DB exists
    conn_root = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
    )
    with conn_root.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn_root.commit()
    conn_root.close()
    print(f"[+] MySQL Database `{MYSQL_DB}` verified/created.")

    # 2. Connect to MySQL with DB
    mysql_conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor,
    )

    # 3. Reconcile existing legacy schema
    reconcile_schema(mysql_conn)

    # 4. Create/verify tables using SQLAlchemy metadata
    sys.path.insert(0, BASE_DIR)
    from app import create_app
    from models import db

    app = create_app()
    app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    with app.app_context():
        db.create_all()
        print("[+] All tables created/verified in MySQL via SQLAlchemy.")

    # 5. Migrate data from SQLite to MySQL
    if os.path.exists(SQLITE_PATH):
        print(f"[*] Reading data from SQLite ({SQLITE_PATH})...")
        sqlite_conn = sqlite3.connect(SQLITE_PATH)
        sqlite_conn.row_factory = sqlite3.Row

        # A. Migrate Cameras
        cur_sq = sqlite_conn.cursor()
        cur_sq.execute("SELECT * FROM cameras")
        cam_rows = cur_sq.fetchall()
        with mysql_conn.cursor() as cur_my:
            for r in cam_rows:
                ts = r["created_at"]
                if ts and isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        pass
                cur_my.execute(
                    "INSERT INTO cameras (id, name, source, location, is_active, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE name=VALUES(name), source=VALUES(source), location=VALUES(location), is_active=VALUES(is_active)",
                    (r["id"], r["name"], r["source"], r["location"], bool(r["is_active"]), ts),
                )
        mysql_conn.commit()
        print(f"[+] Migrated/synchronized {len(cam_rows)} cameras.")

        # B. Migrate Settings
        cur_sq.execute("SELECT * FROM settings")
        setting_rows = cur_sq.fetchall()
        with mysql_conn.cursor() as cur_my:
            for r in setting_rows:
                cur_my.execute(
                    "INSERT INTO settings (id, `key`, value, description) VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE value=VALUES(value), description=VALUES(description)",
                    (r["id"], r["key"], r["value"], r["description"]),
                )
        mysql_conn.commit()
        print(f"[+] Migrated/synchronized {len(setting_rows)} settings.")

        # C. Migrate Exclusion Zones
        cur_sq.execute("SELECT * FROM exclusion_zones")
        ez_rows = cur_sq.fetchall()
        with mysql_conn.cursor() as cur_my:
            for r in ez_rows:
                created_at = r["created_at"]
                updated_at = r["updated_at"]
                if created_at and isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except Exception:
                        pass
                if updated_at and isinstance(updated_at, str):
                    try:
                        updated_at = datetime.fromisoformat(updated_at)
                    except Exception:
                        pass
                cur_my.execute(
                    "INSERT INTO exclusion_zones (id, camera_id, name, zone_type, polygon_points, is_active, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE name=VALUES(name), zone_type=VALUES(zone_type), polygon_points=VALUES(polygon_points), is_active=VALUES(is_active)",
                    (r["id"], r["camera_id"], r["name"], r["zone_type"], r["polygon_points"], bool(r["is_active"]), created_at, updated_at),
                )
        mysql_conn.commit()
        print(f"[+] Migrated/synchronized {len(ez_rows)} exclusion zones.")

        # D. Migrate Violations
        cur_sq.execute("SELECT * FROM violations")
        viol_rows = cur_sq.fetchall()
        migrated_viol = 0
        with mysql_conn.cursor() as cur_my:
            for r in viol_rows:
                ts = r["timestamp"]
                if ts and isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        pass
                cur_my.execute(
                    "INSERT INTO violations (id, camera_id, violation_type, confidence, image_path, timestamp) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE confidence=VALUES(confidence), image_path=VALUES(image_path), timestamp=VALUES(timestamp)",
                    (r["id"], r["camera_id"], r["violation_type"], r["confidence"], r["image_path"], ts),
                )
                migrated_viol += 1
        mysql_conn.commit()
        print(f"[+] Migrated/synchronized {migrated_viol} violations to MySQL.")

        sqlite_conn.close()

    mysql_conn.close()
    print("[SUCCESS] Migration from SQLite to MySQL on XAMPP completed successfully!")


if __name__ == "__main__":
    migrate()
