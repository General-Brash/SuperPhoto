import sqlite3
import secrets
from datetime import datetime, timezone

from .config import ADMIN_PASSWORD, ADMIN_USERNAME, DB_PATH, ensure_directories
from .security import hash_password


SCHEMA = '''
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    user_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    original_name TEXT NOT NULL,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS jobs_status_sequence_idx ON jobs(status, sequence);
CREATE INDEX IF NOT EXISTS jobs_lease_idx ON jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS jobs_batch_idx ON jobs(batch_id, sequence);

CREATE TABLE IF NOT EXISTS service_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    daily_quota INTEGER NOT NULL DEFAULT 30,
    active_quota INTEGER NOT NULL DEFAULT 10,
    image_quotas TEXT NOT NULL DEFAULT '{"2k":30,"4k":20,"6k":10,"8k":0}',
    video_quotas TEXT NOT NULL DEFAULT '{"1k":10,"2k":5,"4k":2}',
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    csrf_token TEXT NOT NULL,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS invites (
    id TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL UNIQUE,
    created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    used_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    expires_at TEXT,
    used_at TEXT,
    revoked_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_by TEXT REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS shares_job_idx ON shares(job_id);
CREATE INDEX IF NOT EXISTS shares_expiry_idx ON shares(expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    owner_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    owner_user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL UNIQUE,
    extension TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    has_alpha INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS uploads_owner_session_idx ON uploads(owner_session_id);
CREATE INDEX IF NOT EXISTS uploads_owner_user_idx ON uploads(owner_user_id);
CREATE INDEX IF NOT EXISTS uploads_expiry_idx ON uploads(expires_at);

CREATE TABLE IF NOT EXISTS job_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    worker_slot INTEGER,
    metrics_version INTEGER NOT NULL DEFAULT 1,
    model_cache_hit INTEGER NOT NULL DEFAULT 1,
    decode_ms REAL NOT NULL DEFAULT 0,
    sr_ms REAL NOT NULL DEFAULT 0,
    resize_ms REAL NOT NULL DEFAULT 0,
    face_ms REAL NOT NULL DEFAULT 0,
    encode_ms REAL NOT NULL DEFAULT 0,
    verify_ms REAL NOT NULL DEFAULT 0,
    total_ms REAL NOT NULL DEFAULT 0,
    output_bytes INTEGER,
    concurrent_jobs INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS job_metrics_created_idx ON job_metrics(created_at);
'''


JOB_COLUMNS = {
    'owner_session_id': 'TEXT REFERENCES sessions(id) ON DELETE SET NULL',
    'owner_user_id': 'TEXT REFERENCES users(id) ON DELETE SET NULL',
    'model_name': "TEXT NOT NULL DEFAULT 'general'",
    'target_resolution': "TEXT NOT NULL DEFAULT '4k'",
    'aspect_ratio': "TEXT NOT NULL DEFAULT 'original'",
    'crop_enabled': 'INTEGER NOT NULL DEFAULT 0',
    'face_enhance': 'INTEGER NOT NULL DEFAULT 0',
    'output_format': "TEXT NOT NULL DEFAULT 'png'",
    'quality_preset': "TEXT NOT NULL DEFAULT 'high'",
    'compression_level': 'INTEGER NOT NULL DEFAULT 5',
    'tile_size': 'INTEGER NOT NULL DEFAULT 256',
    'output_width': 'INTEGER',
    'output_height': 'INTEGER',
    'expires_at': 'TEXT',
    'deleted_at': 'TEXT',
    'progress': 'INTEGER NOT NULL DEFAULT 0',
    'estimated_seconds': 'INTEGER',
    'upload_type': "TEXT NOT NULL DEFAULT 'photo'",
    'duration_seconds': 'REAL',
    'fps': 'REAL',
    'frame_count': 'INTEGER',
    'video_codec': 'TEXT',
    'audio_codec': 'TEXT',
}

BATCH_COLUMNS = {
    'owner_session_id': 'TEXT REFERENCES sessions(id) ON DELETE SET NULL',
    'owner_user_id': 'TEXT REFERENCES users(id) ON DELETE SET NULL',
    'expires_at': 'TEXT',
}

UPLOAD_COLUMNS = {
    'upload_type': "TEXT NOT NULL DEFAULT 'photo'",
    'duration_seconds': 'REAL',
    'fps': 'REAL',
    'frame_count': 'INTEGER',
    'video_codec': 'TEXT',
    'audio_codec': 'TEXT',
}

USER_COLUMNS = {
    'image_quotas': "TEXT NOT NULL DEFAULT '{\"2k\":30,\"4k\":20,\"6k\":10,\"8k\":0}'",
    'video_quotas': "TEXT NOT NULL DEFAULT '{\"1k\":10,\"2k\":5,\"4k\":2}'",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('PRAGMA busy_timeout=30000')
    return connection


def init_db():
    ensure_directories()
    with connect() as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=NORMAL')
        connection.executescript(SCHEMA)
        _add_columns(connection, 'batches', BATCH_COLUMNS)
        _add_columns(connection, 'jobs', JOB_COLUMNS)
        _add_columns(connection, 'uploads', UPLOAD_COLUMNS)
        _add_columns(connection, 'users', USER_COLUMNS)
        connection.execute(
            '''UPDATE jobs SET estimated_seconds=MAX(1, ROUND(width * height * 12.0 / 1000000))
               WHERE estimated_seconds IS NULL'''
        )
        connection.execute("UPDATE jobs SET progress=100 WHERE status='succeeded' AND progress < 100")
        _bootstrap_admin(connection)


def _add_columns(connection, table, columns):
    existing = {row['name'] for row in connection.execute(f'PRAGMA table_info({table})')}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')


def _bootstrap_admin(connection):
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        return
    exists = connection.execute('SELECT 1 FROM users WHERE role=? LIMIT 1', ('admin',)).fetchone()
    if exists:
        return
    now = utc_now()
    connection.execute(
        '''INSERT INTO users(id, username, password_hash, role, daily_quota, active_quota,
           created_at, updated_at) VALUES (?, ?, ?, 'admin', 1000, 100, ?, ?)''',
        (secrets.token_hex(16), ADMIN_USERNAME, hash_password(ADMIN_PASSWORD), now, now),
    )
