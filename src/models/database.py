"""
SQLite database setup for the Yunnan Enterprise Employment Data Collection System.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'system.db')


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
    -- Users and roles
    CREATE TABLE IF NOT EXISTS roles (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT    NOT NULL UNIQUE,
        desc TEXT
    );

    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT    NOT NULL UNIQUE,
        password    TEXT    NOT NULL,
        role_id     INTEGER NOT NULL REFERENCES roles(id),
        region_code TEXT,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    -- Enterprise filing
    CREATE TABLE IF NOT EXISTS enterprises (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        org_code        TEXT    NOT NULL UNIQUE,    -- 组织机构代码
        name            TEXT    NOT NULL,
        nature          TEXT,                       -- 企业性质
        industry        TEXT,                       -- 所属行业
        region_code     TEXT    NOT NULL,
        contact_person  TEXT,
        contact_phone   TEXT,
        email           TEXT,
        address         TEXT,
        main_business   TEXT,
        status          TEXT    NOT NULL DEFAULT 'pending',  -- pending/approved/rejected
        created_by      INTEGER REFERENCES users(id),
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    -- Survey periods (上报期)
    CREATE TABLE IF NOT EXISTS survey_periods (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        period_key TEXT    NOT NULL UNIQUE,  -- e.g. '2025-01'
        start_date TEXT    NOT NULL,
        end_date   TEXT    NOT NULL,
        created_by INTEGER REFERENCES users(id)
    );

    -- Employment data submissions (数据填报)
    CREATE TABLE IF NOT EXISTS submissions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        enterprise_id           INTEGER NOT NULL REFERENCES enterprises(id),
        period_id               INTEGER NOT NULL REFERENCES survey_periods(id),
        baseline_employees      INTEGER NOT NULL,   -- 建档期就业人数
        survey_employees        INTEGER NOT NULL,   -- 调查期就业人数
        decrease_type           TEXT,               -- 就业人数减少类型
        main_reason             TEXT,
        main_reason_desc        TEXT,
        secondary_reason        TEXT,
        secondary_reason_desc   TEXT,
        third_reason            TEXT,
        third_reason_desc       TEXT,
        status                  TEXT NOT NULL DEFAULT 'draft',  -- draft/submitted/city_approved/province_approved/returned
        submitted_by            INTEGER REFERENCES users(id),
        submitted_at            TEXT,
        city_reviewed_by        INTEGER REFERENCES users(id),
        city_reviewed_at        TEXT,
        province_reviewed_by    INTEGER REFERENCES users(id),
        province_reviewed_at    TEXT,
        return_reason           TEXT,
        created_at              TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );

    -- Notifications (通知)
    CREATE TABLE IF NOT EXISTS notifications (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        title        TEXT    NOT NULL,
        content      TEXT    NOT NULL,
        published_by INTEGER NOT NULL REFERENCES users(id),
        published_at TEXT    NOT NULL DEFAULT (datetime('now')),
        is_deleted   INTEGER NOT NULL DEFAULT 0
    );

    -- Agent messages (消息总线)
    CREATE TABLE IF NOT EXISTS agent_messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        sender       TEXT    NOT NULL,
        receiver     TEXT    NOT NULL,
        msg_type     TEXT    NOT NULL,
        payload      TEXT    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'pending',  -- pending/processed
        created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        processed_at TEXT
    );
    """)

    # Seed roles
    cur.executemany(
        "INSERT OR IGNORE INTO roles(name, desc) VALUES (?, ?)",
        [
            ('province_admin', '省级管理员'),
            ('city_admin', '市级管理员'),
            ('enterprise', '企业用户'),
        ]
    )

    # Seed a default province admin
    cur.execute(
        "INSERT OR IGNORE INTO users(username, password, role_id, region_code) "
        "VALUES ('admin', 'admin123', (SELECT id FROM roles WHERE name='province_admin'), '53')"
    )

    conn.commit()
    conn.close()
