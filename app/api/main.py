import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.common.auth import (
    clear_session,
    get_session,
    owner_clause,
    require_admin,
    require_csrf,
    require_user,
    rotate_session,
)
from app.common.config import (
    ANIME_MODEL_PATH,
    GFPGAN_MODEL_PATH,
    INPUT_DIR,
    MAX_BATCH_FILES,
    MAX_FILE_BYTES,
    MAX_GLOBAL_JOBS,
    MAX_OUTPUT_PIXELS,
    GUEST_DAILY_QUOTA,
    DEFAULT_IMAGE_QUOTAS,
    DEFAULT_VIDEO_QUOTAS,
    MAX_SIDE,
    MIN_FREE_DISK_BYTES,
    MODEL_PATH,
    MODEL_REGISTRY,
    OUTPUT_DIR,
    ROLE_ADMIN,
    ROLE_ADVANCED,
    ROLE_GUEST,
    ROLE_USER,
    SHARE_TTL_HOURS,
    STATIC_DIR,
    SESSION_SECRET,
    TMP_DIR,
    TURNSTILE_REQUIRED,
    TURNSTILE_SECRET_KEY,
    TURNSTILE_SITE_KEY,
    VALID_ROLES,
    FFMPEG_BIN,
    FFPROBE_BIN,
    VIDEO_INPUT_CODECS,
    VIDEO_INPUT_EXTENSIONS,
    VIDEO_MAX_FILE_BYTES,
    VIDEO_MAX_HEIGHT,
    VIDEO_MAX_WIDTH,
    ensure_directories,
    face_models_available,
)
from app.common.db import connect, init_db, utc_now
from app.common.jobs import (
    estimate_output_bytes,
    estimate_processing_seconds,
    estimate_video_output_bytes,
    expiry_iso,
    output_extension,
    target_dimensions,
    validate_settings,
)
from app.common.security import hash_password, random_token, token_hash, verify_password


app = FastAPI(title='SuperPhoto API', version='1.0.4', docs_url=None, redoc_url=None, openapi_url=None)
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE
USERNAME_PATTERN = re.compile(r'^[A-Za-z0-9_]{3,24}$')
_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()


@app.middleware('http')
async def disable_stale_web_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith('/assets/') and request.query_params.get('v'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    elif request.url.path == '/' or request.url.path.startswith('/assets/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


class RegisterPayload(BaseModel):
    username: str
    password: str
    invite_code: str


class LoginPayload(BaseModel):
    username: str
    password: str


class PasswordPayload(BaseModel):
    current_password: str
    new_password: str


class InvitePayload(BaseModel):
    expires_hours: int | None = 168


class UserUpdatePayload(BaseModel):
    role: str | None = None
    daily_quota: int | None = None
    active_quota: int | None = None
    disabled: bool | None = None
    image_quotas: dict[str, int] | None = None
    video_quotas: dict[str, int] | None = None
    daily_image_quota: int | None = None
    daily_video_quota: int | None = None
    resolution_1k: int | None = None
    resolution_2k: int | None = None
    resolution_4k: int | None = None


class ResetPasswordPayload(BaseModel):
    new_password: str


class EstimateImagePayload(BaseModel):
    width: int
    height: int
    has_alpha: bool = False
    size_bytes: int = 0
    duration_seconds: float = 0
    fps: float = 0
    frame_count: int = 0


class EstimatePayload(BaseModel):
    images: list[EstimateImagePayload]
    settings: dict = Field(default_factory=dict)


def client_key(request):
    return request.headers.get('CF-Connecting-IP') or (request.client.host if request.client else 'local')


def rate_limit(request, name, limit, window_seconds):
    key = f'{name}:{client_key(request)}'
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and bucket[0] <= now - window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(429, 'Too many requests')
        bucket.append(now)


def validate_image(content):
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, 'File exceeds the 10 MiB limit')
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        expected_format, extension = 'PNG', '.png'
    elif content.startswith(b'\xff\xd8\xff'):
        expected_format, extension = 'JPEG', '.jpg'
    else:
        raise HTTPException(400, 'Only static JPEG and PNG files are accepted')
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != expected_format or getattr(image, 'is_animated', False):
                raise HTTPException(400, 'Only static JPEG and PNG files are accepted')
            width, height = image.size
            has_alpha = 'A' in image.getbands() or image.info.get('transparency') is not None
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(400, f'Invalid image: {error}') from error
    if width < 1 or height < 1 or width > MAX_SIDE or height > MAX_SIDE:
        raise HTTPException(400, f'Image dimensions must be between 1 and {MAX_SIDE} pixels per side')
    if width * height * 16 > MAX_OUTPUT_PIXELS:
        raise HTTPException(400, 'Estimated x4 intermediate image exceeds the 64 MP limit')
    return extension, width, height, has_alpha


VIDEO_FORMAT_NAMES = {'mov', 'mp4', 'm4v', 'matroska', 'webm', 'avi', 'flv', 'mpeg'}


def upload_looks_video(upload):
    suffix = Path(upload.filename or '').suffix.lower()
    content_type = (upload.content_type or '').lower()
    return suffix in VIDEO_INPUT_EXTENSIONS or content_type.startswith('video/')


def store_upload_stream(upload, max_bytes):
    path = TMP_DIR / f'.incoming-{uuid.uuid4().hex}.part'
    size = 0
    try:
        with path.open('wb') as target:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, 'Video file exceeds the 1 GiB limit')
                target.write(chunk)
        return path, size
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except OSError as error:
        path.unlink(missing_ok=True)
        raise HTTPException(500, f'Failed to store upload: {error}') from error


def probe_video(path):
    try:
        result = subprocess.run(
            [FFPROBE_BIN, '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        probe = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise HTTPException(400, 'Unable to inspect video; supported video codecs are required') from error
    streams = probe.get('streams') or []
    video = next((item for item in streams if item.get('codec_type') == 'video'), None)
    if not video:
        raise HTTPException(400, 'Video stream is missing')
    format_name = str((probe.get('format') or {}).get('format_name') or '').split(',')[0].lower()
    codec = str(video.get('codec_name') or '').lower()
    if format_name not in VIDEO_FORMAT_NAMES:
        raise HTTPException(400, 'Unsupported video container format')
    if codec not in VIDEO_INPUT_CODECS:
        raise HTTPException(400, 'Unsupported video codec')
    try:
        width, height = int(video.get('width') or 0), int(video.get('height') or 0)
        duration = float(video.get('duration') or (probe.get('format') or {}).get('duration') or 0)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid duration')
        rotation = 0
        rotation = int((video.get('tags') or {}).get('rotate') or 0)
    except (TypeError, ValueError, OverflowError) as error:
        raise HTTPException(400, 'Video metadata is invalid') from error
    for side_data in video.get('side_data_list') or []:
        if side_data.get('rotation') is not None:
            try:
                rotation = int(side_data['rotation'])
            except (TypeError, ValueError):
                pass
    if abs(rotation) % 180 == 90:
        width, height = height, width
    if (width < 1 or height < 1 or max(width, height) > VIDEO_MAX_WIDTH
            or min(width, height) > VIDEO_MAX_HEIGHT):
        raise HTTPException(400, 'Video resolution must not exceed 1920x1080 (portrait videos may be 1080x1920)')
    try:
        fps_text = str(video.get('avg_frame_rate') or video.get('r_frame_rate') or '0/1')
        numerator, denominator = fps_text.split('/', 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if not math.isfinite(fps) or fps < 0 or fps > 240:
        raise HTTPException(400, 'Video frame rate is invalid or unsupported')
    try:
        frame_count = int(video.get('nb_frames') or 0) or None
    except (TypeError, ValueError, OverflowError):
        frame_count = None
    if frame_count is None and duration > 0 and fps > 0:
        frame_count = max(1, round(duration * fps))
    audio = next((item for item in streams if item.get('codec_type') == 'audio'), None)
    return {
        'extension': Path(path).suffix.lower() or '.mp4',
        'width': width,
        'height': height,
        'duration_seconds': max(0.0, duration),
        'fps': fps if fps > 0 else 24.0,
        'frame_count': frame_count,
        'video_codec': codec,
        'audio_codec': str(audio.get('codec_name') or '') if audio else None,
    }


def serialize_user(row):
    result = {
        'id': row['id'],
        'username': row['username'],
        'role': row['role'],
        'daily_quota': row['daily_quota'],
        'active_quota': row['active_quota'],
        'disabled': bool(row['disabled']),
        'created_at': row['created_at'],
    }
    # Optional per-user limits are added by newer schema migrations. Keep the
    # admin API compatible with existing databases while exposing them when
    # available.
    for key in ('daily_image_quota', 'daily_video_quota', 'resolution_1k', 'resolution_2k', 'resolution_4k'):
        if key in row.keys():
            result[key] = row[key]
    for key, defaults in (('image_quotas', DEFAULT_IMAGE_QUOTAS), ('video_quotas', DEFAULT_VIDEO_QUOTAS)):
        if key in row.keys():
            try:
                value = json.loads(row[key] or '{}')
                result[key] = {resolution: max(0, int(value.get(resolution, 0))) for resolution in defaults}
            except (TypeError, ValueError, json.JSONDecodeError):
                result[key] = dict(defaults)
        else:
            result[key] = dict(defaults)
    return result


def serialize_job(connection, row):
    position = None
    if row['status'] == 'queued':
        position = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND deleted_at IS NULL AND sequence <= ?",
            (row['sequence'],),
        ).fetchone()[0]
    progress = int(row['progress'] or 0)
    return {
        'id': row['id'],
        'batch_id': row['batch_id'],
        'original_name': row['original_name'],
        'upload_type': row['upload_type'] if 'upload_type' in row.keys() else 'photo',
        'width': row['width'],
        'height': row['height'],
        'output_width': row['output_width'],
        'output_height': row['output_height'],
        'status': row['status'],
        'position': position,
        'attempts': row['attempts'],
        'error': row['error'],
        'model': row['model_name'],
        'target_resolution': row['target_resolution'],
        'aspect_ratio': row['aspect_ratio'],
        'crop_enabled': bool(row['crop_enabled']),
        'face_enhance': bool(row['face_enhance']),
        'output_format': row['output_format'],
        'quality_preset': row['quality_preset'],
        'compression_level': row['compression_level'],
        'tile_size': row['tile_size'],
        'progress': progress,
        'estimated_seconds': row['estimated_seconds'],
        'remaining_seconds': job_remaining_seconds(connection, row, progress),
        'created_at': row['created_at'],
        'started_at': row['started_at'],
        'finished_at': row['finished_at'],
        'expires_at': row['expires_at'],
        'duration_seconds': row['duration_seconds'] if 'duration_seconds' in row.keys() else None,
        'fps': row['fps'] if 'fps' in row.keys() else None,
    }


def worker_count(connection):
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
    count = connection.execute(
        "SELECT COUNT(*) FROM service_state WHERE key LIKE 'worker:%' AND updated_at >= ?",
        (cutoff,),
    ).fetchone()[0]
    return max(1, count)


def simulate_queue_seconds(connection, estimates):
    """Assign active and incoming jobs to the earliest-free worker."""
    workers = worker_count(connection)
    loads = [0.0] * workers
    rows = connection.execute(
        '''SELECT estimated_seconds, progress, status FROM jobs
           WHERE deleted_at IS NULL AND status IN ('queued', 'processing')
           ORDER BY sequence'''
    ).fetchall()
    processing_index = 0
    for row in rows:
        remaining = float(row['estimated_seconds'] or 1) * max(0, 100 - int(row['progress'] or 0)) / 100.0
        if row['status'] == 'processing' and processing_index < workers:
            loads[processing_index] += remaining
            processing_index += 1
        else:
            slot = min(range(workers), key=lambda index: loads[index])
            loads[slot] += remaining
    for estimate in estimates:
        slot = min(range(workers), key=lambda index: loads[index])
        loads[slot] += max(1, float(estimate))
    return round(max(loads)) if loads else 0


def job_remaining_seconds(connection, row, progress=None):
    estimate = int(row['estimated_seconds'] or estimate_processing_seconds(row['width'], row['height'], row['face_enhance']))
    if row['status'] == 'processing':
        current_progress = int(row['progress'] or 0) if progress is None else progress
        return max(1, round(estimate * (100 - current_progress) / 100)) if current_progress < 100 else 0
    if row['status'] != 'queued':
        return 0 if row['status'] == 'succeeded' else None
    workers = worker_count(connection)
    loads = [0.0] * workers
    ahead_rows = connection.execute(
        '''SELECT estimated_seconds, progress, status FROM jobs
           WHERE deleted_at IS NULL AND (
               status='processing' OR (status='queued' AND sequence < ?)
           ) ORDER BY sequence''',
        (row['sequence'],),
    ).fetchall()
    processing_index = 0
    for item in ahead_rows:
        remaining = float(item['estimated_seconds'] or 1) * max(0, 100 - int(item['progress'] or 0)) / 100.0
        if item['status'] == 'processing' and processing_index < workers:
            loads[processing_index] += remaining
            processing_index += 1
        else:
            slot = min(range(workers), key=lambda index: loads[index])
            loads[slot] += remaining
    slot = min(range(workers), key=lambda index: loads[index])
    loads[slot] += estimate
    return max(1, round(max(loads)))


def usage_summary(connection, session):
    clause, values = owner_clause(session)
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    used_today = connection.execute(
        f'SELECT COUNT(*) FROM jobs WHERE created_at >= ? AND deleted_at IS NULL AND {clause}',
        (midnight, *values),
    ).fetchone()[0]
    active = connection.execute(
        f"SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing') AND deleted_at IS NULL AND {clause}",
        values,
    ).fetchone()[0]
    role = session.get('role') or ROLE_GUEST
    daily_quota = GUEST_DAILY_QUOTA if role == ROLE_GUEST else int(session['daily_quota'])
    active_quota = 2 if role == ROLE_GUEST else int(session['active_quota'])
    return {
        'daily_quota': daily_quota,
        'used_today': used_today,
        'remaining_today': max(0, daily_quota - used_today),
        'active_quota': active_quota,
        'active_jobs': active,
        'remaining_active': max(0, active_quota - active),
    }


def processing_observations(connection):
    rows = connection.execute(
        '''SELECT j.width, j.height, j.output_width, j.output_height, j.model_name, j.tile_size,
                  j.face_enhance, j.output_format, j.quality_preset, j.compression_level,
                  j.started_at, j.finished_at, m.*
           FROM jobs j LEFT JOIN job_metrics m ON m.job_id=j.id
           WHERE j.status='succeeded' AND j.started_at IS NOT NULL AND j.finished_at IS NOT NULL
           ORDER BY j.finished_at DESC LIMIT 200'''
    ).fetchall()
    observations = []
    for row in rows:
        try:
            elapsed = (datetime.fromisoformat(row['finished_at']) - datetime.fromisoformat(row['started_at'])).total_seconds()
        except (TypeError, ValueError):
            continue
        observations.append({
            'width': row['width'], 'height': row['height'],
            'input_mp': row['width'] * row['height'] / 1_000_000,
            'output_mp': (row['output_width'] or row['width'] * 4) * (row['output_height'] or row['height'] * 4) / 1_000_000,
            'model_name': row['model_name'], 'tile_size': row['tile_size'],
            'face_enhance': row['face_enhance'], 'output_format': row['output_format'],
            'quality_preset': row['quality_preset'], 'compression_level': row['compression_level'],
            'total_seconds': elapsed,
            'sr_seconds': (row['sr_ms'] or 0) / 1000,
            'resize_seconds': (row['resize_ms'] or 0) / 1000,
            'face_seconds': (row['face_ms'] or 0) / 1000,
            'encode_seconds': (row['encode_ms'] or 0) / 1000,
            'work_mp': row['width'] * row['height'] / 1_000_000 * 16,
            'fixed_seconds': ((row['decode_ms'] or 0) + (row['verify_ms'] or 0)) / 1000,
        })
    return observations


def owned_job(connection, job_id, session):
    clause, values = owner_clause(session)
    row = connection.execute(
        f'SELECT * FROM jobs WHERE id=? AND deleted_at IS NULL AND {clause}', (job_id, *values)
    ).fetchone()
    if not row:
        raise HTTPException(404, 'Job not found')
    return row


def owned_batch(connection, batch_id, session):
    clause, values = owner_clause(session)
    row = connection.execute(f'SELECT * FROM batches WHERE id=? AND {clause}', (batch_id, *values)).fetchone()
    if not row:
        raise HTTPException(404, 'Batch not found')
    return row


def delete_job_files(row):
    for root, name in ((INPUT_DIR, row['input_path']), (OUTPUT_DIR, row['output_path'])):
        if not name:
            continue
        path = (root / name).resolve()
        if path.parent == root.resolve() and not path.is_symlink():
            path.unlink(missing_ok=True)


def upload_file_path(row):
    path = (TMP_DIR / row['stored_path']).resolve()
    if path.parent != TMP_DIR.resolve() or path.is_symlink():
        raise HTTPException(410, 'Uploaded file is missing')
    return path


def owned_upload(connection, upload_id, session):
    clause, values = owner_clause(session)
    row = connection.execute(
        f'SELECT * FROM uploads WHERE id=? AND expires_at > ? AND {clause}',
        (upload_id, utc_now(), *values),
    ).fetchone()
    if not row:
        raise HTTPException(404, 'Upload not found')
    path = upload_file_path(row)
    if not path.is_file():
        raise HTTPException(410, 'Uploaded file is missing')
    return row


def delete_upload_file(row):
    try:
        upload_file_path(row).unlink(missing_ok=True)
    except HTTPException:
        pass


def restore_prepared_uploads(prepared, created_paths, transient_paths):
    for item in prepared:
        if item['input_path'] in created_paths and item['upload_id']:
            try:
                if item['input_path'].is_file():
                    os.replace(item['input_path'], item['temp_path'])
            except OSError:
                pass
        elif not item['upload_id']:
            item['input_path'].unlink(missing_ok=True)
            item['temp_path'].unlink(missing_ok=True)
    for path in transient_paths:
        path.unlink(missing_ok=True)


def verify_turnstile(token, request):
    if not TURNSTILE_REQUIRED:
        return
    if not TURNSTILE_SECRET_KEY:
        raise HTTPException(503, 'Turnstile is required but not configured')
    data = urllib.parse.urlencode(
        {'secret': TURNSTILE_SECRET_KEY, 'response': token, 'remoteip': client_key(request)}
    ).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request('https://challenges.cloudflare.com/turnstile/v0/siteverify', data=data),
            timeout=8,
        ) as response:
            result = json.load(response)
    except (OSError, ValueError) as error:
        raise HTTPException(503, 'Turnstile verification is unavailable') from error
    if not result.get('success'):
        raise HTTPException(403, 'Turnstile verification failed')


def cleanup_expired():
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM jobs WHERE expires_at < ? AND status != 'processing'", (utc_now(),)
        ).fetchall()
        for row in rows:
            delete_job_files(row)
        connection.executemany('DELETE FROM jobs WHERE id=?', [(row['id'],) for row in rows])
        connection.execute('DELETE FROM batches WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.batch_id=batches.id)')
        connection.execute('DELETE FROM shares WHERE expires_at < ? OR revoked_at IS NOT NULL', (utc_now(),))
        connection.execute('DELETE FROM sessions WHERE expires_at < ?', (utc_now(),))
        uploads = connection.execute('SELECT * FROM uploads WHERE expires_at < ?', (utc_now(),)).fetchall()
        for upload in uploads:
            delete_upload_file(upload)
        connection.executemany('DELETE FROM uploads WHERE id=?', [(upload['id'],) for upload in uploads])
    for pattern in ('*.zip', '*.partial.*', '.incoming-*.part'):
        for path in TMP_DIR.glob(pattern):
            try:
                if path.stat().st_mtime < time.time() - 3600:
                    path.unlink(missing_ok=True)
            except OSError:
                pass


def cleanup_loop():
    while True:
        try:
            cleanup_expired()
        except Exception:
            pass
        time.sleep(3600)


@app.on_event('startup')
def startup():
    if len(SESSION_SECRET) < 32:
        raise RuntimeError('SUPERPHOTO_SESSION_SECRET must contain at least 32 characters')
    ensure_directories()
    init_db()
    threading.Thread(target=cleanup_loop, daemon=True, name='superphoto-cleanup').start()


@app.get('/')
def website():
    index = STATIC_DIR / 'index.html'
    if not index.is_file():
        raise HTTPException(503, 'Web application is not installed')
    return FileResponse(index)


@app.get('/health')
def legacy_health():
    return service_health()


@app.get('/api/health')
def service_health():
    with connect() as connection:
        workers = connection.execute(
            "SELECT key, value, updated_at FROM service_state WHERE key LIKE 'worker:%' ORDER BY key"
        ).fetchall()
    return {
        'status': 'ok',
        'workers': [dict(worker) for worker in workers],
        'models': {
            'general': MODEL_PATH.is_file(),
            'anime': ANIME_MODEL_PATH.is_file(),
            'face_enhance': face_models_available(),
            'video': shutil.which(FFMPEG_BIN) is not None and shutil.which(FFPROBE_BIN) is not None,
        },
    }


@app.get('/api/auth/me')
def auth_me(request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response)
        user = None
        if session.get('user_id'):
            row = connection.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
            user = serialize_user(row)
        usage = usage_summary(connection, session)
    return {
        'authenticated': bool(user),
        'user': user,
        'role': user['role'] if user else ROLE_GUEST,
        'csrf_token': session['csrf_token'],
        'turnstile_site_key': TURNSTILE_SITE_KEY,
        'turnstile_required': TURNSTILE_REQUIRED,
        'capabilities': service_health()['models'],
        'usage': usage,
    }


@app.get('/api/usage')
def api_usage(request: Request, response: Response):
    with connect() as connection:
        return usage_summary(connection, get_session(connection, request, response))


@app.post('/api/estimate')
def api_estimate(payload: EstimatePayload, request: Request, response: Response):
    if not payload.images or len(payload.images) > MAX_BATCH_FILES:
        raise HTTPException(400, f'An estimate must contain 1 to {MAX_BATCH_FILES} images')
    with connect() as connection:
        session = get_session(connection, request, response)
        role = session.get('role') or ROLE_GUEST
        batch_limit = 2 if role == ROLE_GUEST else MAX_BATCH_FILES
        if len(payload.images) > batch_limit:
            raise HTTPException(400, f'An estimate must contain 1 to {batch_limit} images')
        timing_samples = processing_observations(connection)
        size_rows = connection.execute(
            '''SELECT output_width, output_height, output_format, quality_preset, compression_level, output_path
               FROM jobs WHERE status='succeeded' AND deleted_at IS NULL
               ORDER BY finished_at DESC LIMIT 200'''
        ).fetchall()
        size_samples = []
        for row in size_rows:
            path = OUTPUT_DIR / row['output_path']
            if path.is_file() and row['output_width'] and row['output_height']:
                size_samples.append((row['output_width'], row['output_height'], row['output_format'], row['quality_preset'], row['compression_level'], path.stat().st_size))
        estimates = []
        size_estimates = []
        for image in payload.images:
            settings = validate_settings(payload.settings, role, image.has_alpha)
            if settings['upload_type'] == 'video':
                if (image.width < 1 or image.height < 1 or max(image.width, image.height) > VIDEO_MAX_WIDTH
                        or min(image.width, image.height) > VIDEO_MAX_HEIGHT):
                    raise HTTPException(400, 'Video resolution must not exceed 1920x1080')
                if image.size_bytes < 0 or image.size_bytes > VIDEO_MAX_FILE_BYTES:
                    raise HTTPException(400, 'Video size metadata is invalid')
                if any(not math.isfinite(value) or value < 0 for value in (image.duration_seconds, image.fps)):
                    raise HTTPException(400, 'Video timing metadata is invalid')
            elif image.width < 1 or image.height < 1 or image.width > MAX_SIDE or image.height > MAX_SIDE:
                raise HTTPException(400, f'Image dimensions must be between 1 and {MAX_SIDE} pixels per side')
            if settings['upload_type'] == 'photo' and image.width * image.height * 16 > MAX_OUTPUT_PIXELS:
                raise HTTPException(400, 'Estimated x4 intermediate image exceeds the 64 MP limit')
            output_width, output_height = target_dimensions(
                image.width, image.height, settings['target_resolution'], settings['aspect_ratio'], bool(settings['crop_enabled'])
            )
            if settings['upload_type'] == 'video':
                output_width -= output_width % 2
                output_height -= output_height % 2
            estimates.append(estimate_processing_seconds(
                image.width, image.height, bool(settings['face_enhance']), observations=timing_samples,
                settings={**settings, 'output_width': output_width, 'output_height': output_height,
                          'has_alpha': image.has_alpha, 'duration_seconds': image.duration_seconds,
                          'fps': image.fps, 'frame_count': image.frame_count},
            ))
            if settings['upload_type'] == 'video':
                size_estimates.append(estimate_video_output_bytes(
                    output_width, output_height, image.fps, image.duration_seconds, settings['output_format']
                ))
            else:
                size_estimates.append(estimate_output_bytes(
                    output_width, output_height, settings['output_format'], settings['quality_preset'],
                    settings['compression_level'], size_samples,
                ))
        usage = usage_summary(connection, session)
        processing_seconds = sum(estimates)
        estimated_queue = simulate_queue_seconds(connection, [])
        estimated_remaining = simulate_queue_seconds(connection, estimates)
        workers = worker_count(connection)
        return {
            'image_estimates_seconds': estimates,
            'image_estimates_bytes': size_estimates,
            'estimated_output_bytes': sum(size_estimates),
            'calibration_samples': len(timing_samples),
            'size_calibration_samples': len(size_samples),
            'estimated_processing_seconds': processing_seconds,
            'estimated_queue_seconds': estimated_queue,
            'estimated_remaining_seconds': max(1, estimated_remaining),
            'remaining_today': usage['remaining_today'],
            'remaining_today_after': max(0, usage['remaining_today'] - len(payload.images)),
            'can_submit': len(payload.images) <= usage['remaining_today'] and len(payload.images) <= usage['remaining_active'],
        }


@app.post('/api/auth/register', status_code=201)
def register(payload: RegisterPayload, request: Request, response: Response):
    rate_limit(request, 'register', 10, 3600)
    if not USERNAME_PATTERN.fullmatch(payload.username):
        raise HTTPException(400, 'Username must contain 3 to 24 letters, numbers, or underscores')
    try:
        password_hash = hash_password(payload.password)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    connection = connect()
    try:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        connection.commit()
        connection.execute('BEGIN IMMEDIATE')
        invite = connection.execute(
            '''SELECT * FROM invites WHERE code_hash=? AND used_at IS NULL AND revoked_at IS NULL
               AND (expires_at IS NULL OR expires_at > ?)''',
            (token_hash(payload.invite_code.strip()), utc_now()),
        ).fetchone()
        if not invite:
            raise HTTPException(400, 'Invitation code is invalid or expired')
        user_id = uuid.uuid4().hex
        now = utc_now()
        connection.execute(
            '''INSERT INTO users(id, username, password_hash, role, daily_quota, active_quota,
               created_at, updated_at) VALUES (?, ?, ?, 'user', 30, 10, ?, ?)''',
            (user_id, payload.username, password_hash, now, now),
        )
        connection.execute('UPDATE invites SET used_by=?, used_at=? WHERE id=?', (user_id, now, invite['id']))
        connection.execute('UPDATE uploads SET owner_user_id=? WHERE owner_session_id=?', (user_id, session['id']))
        rotate_session(connection, request, response, user_id)
        connection.commit()
    except HTTPException:
        connection.rollback()
        raise
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise HTTPException(409, 'Username is already registered') from error
    finally:
        connection.close()
    return {'status': 'registered'}


@app.post('/api/auth/login')
def login(payload: LoginPayload, request: Request, response: Response):
    rate_limit(request, 'login', 20, 900)
    connection = connect()
    try:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        user = connection.execute('SELECT * FROM users WHERE username=? COLLATE NOCASE', (payload.username,)).fetchone()
        if not user or user['disabled'] or not verify_password(payload.password, user['password_hash']):
            raise HTTPException(401, 'Invalid username or password')
        connection.execute('UPDATE uploads SET owner_user_id=? WHERE owner_session_id=?', (user['id'], session['id']))
        rotate_session(connection, request, response, user['id'])
        connection.commit()
        return {'status': 'authenticated', 'user': serialize_user(user)}
    finally:
        connection.close()


@app.post('/api/auth/logout')
def logout(request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response, create=False)
        if session:
            require_csrf(request, session)
        clear_session(connection, request, response)
    return {'status': 'logged_out'}


@app.post('/api/auth/password')
def change_password(payload: PasswordPayload, request: Request, response: Response):
    with connect() as connection:
        session = require_user(get_session(connection, request, response))
        require_csrf(request, session)
        user = connection.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
        if not verify_password(payload.current_password, user['password_hash']):
            raise HTTPException(400, 'Current password is incorrect')
        try:
            new_hash = hash_password(payload.new_password)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        connection.execute('UPDATE users SET password_hash=?, updated_at=? WHERE id=?', (new_hash, utc_now(), user['id']))
    return {'status': 'password_changed'}


@app.delete('/api/auth/delete')
def delete_account(request: Request, response: Response):
    with connect() as connection:
        session = require_user(get_session(connection, request, response))
        require_csrf(request, session)
        if session['role'] == ROLE_ADMIN:
            raise HTTPException(409, 'Administrator accounts cannot delete themselves')
        rows = connection.execute('SELECT * FROM jobs WHERE owner_user_id=?', (session['user_id'],)).fetchall()
        if any(row['status'] == 'processing' for row in rows):
            raise HTTPException(409, 'Wait for active processing jobs before deleting the account')
        for row in rows:
            delete_job_files(row)
        uploads = connection.execute('SELECT * FROM uploads WHERE owner_user_id=?', (session['user_id'],)).fetchall()
        for upload in uploads:
            delete_upload_file(upload)
        connection.execute('DELETE FROM uploads WHERE owner_user_id=?', (session['user_id'],))
        connection.execute('DELETE FROM jobs WHERE owner_user_id=?', (session['user_id'],))
        connection.execute('DELETE FROM batches WHERE owner_user_id=?', (session['user_id'],))
        connection.execute('DELETE FROM users WHERE id=?', (session['user_id'],))
        response.delete_cookie('superphoto_session', path='/')
    return {'status': 'account_deleted'}


@app.post('/api/uploads', status_code=201)
def create_uploads(request: Request, response: Response, files: list[UploadFile] = File(...)):
    rate_limit(request, 'upload', 60, 3600)
    connection = connect()
    created_paths = []
    try:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        role = session.get('role') or ROLE_GUEST
        upload_limit = 2 if role == ROLE_GUEST else MAX_BATCH_FILES
        if not files or len(files) > upload_limit:
            raise HTTPException(400, f'An upload must contain 1 to {upload_limit} files')
        clause, values = owner_clause(session)
        existing = connection.execute(
            f'SELECT COUNT(*) FROM uploads WHERE expires_at > ? AND {clause}',
            (utc_now(), *values),
        ).fetchone()[0]
        if existing + len(files) > upload_limit:
            raise HTTPException(429, f'At most {upload_limit} unsubmitted files are allowed')
        if shutil.disk_usage(TMP_DIR).free < MIN_FREE_DISK_BYTES:
            raise HTTPException(503, 'Insufficient free disk space')

        result = []
        for upload in files:
            is_video = upload_looks_video(upload)
            if is_video and role != ROLE_ADMIN:
                raise HTTPException(403, 'Video processing is restricted to administrators')
            if is_video:
                incoming_path, size_bytes = store_upload_stream(upload, VIDEO_MAX_FILE_BYTES)
                try:
                    metadata = probe_video(incoming_path)
                except Exception:
                    incoming_path.unlink(missing_ok=True)
                    raise
                extension = Path(upload.filename or '').suffix.lower()
                if extension not in VIDEO_INPUT_EXTENSIONS:
                    extension = '.mp4'
                width, height, has_alpha = metadata['width'], metadata['height'], False
                content = None
            else:
                content = upload.file.read(MAX_FILE_BYTES + 1)
                extension, width, height, has_alpha = validate_image(content)
                size_bytes = len(content)
            upload_id = uuid.uuid4().hex
            stored_path = f'upload-{upload_id}{extension}'
            path = TMP_DIR / stored_path
            if content is not None:
                path.write_bytes(content)
            else:
                os.replace(incoming_path, path)
            created_paths.append(path)
            expires_at = expiry_iso(session.get('user_id'))
            connection.execute(
                '''INSERT INTO uploads(
                       id, owner_session_id, owner_user_id, original_name, stored_path,
                       extension, width, height, has_alpha, size_bytes, created_at, expires_at,
                       upload_type, duration_seconds, fps, frame_count, video_codec, audio_codec
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    upload_id, session['id'], session.get('user_id'), upload.filename or 'image', stored_path,
                    extension, width, height, int(has_alpha), size_bytes, utc_now(), expires_at,
                    'video' if is_video else 'photo',
                    metadata['duration_seconds'] if is_video else None,
                    metadata['fps'] if is_video else None,
                    metadata['frame_count'] if is_video else None,
                    metadata['video_codec'] if is_video else None,
                    metadata['audio_codec'] if is_video else None,
                ),
            )
            result.append({
                'id': upload_id,
                'original_name': upload.filename or 'image',
                'width': width,
                'height': height,
                'has_alpha': has_alpha,
                'size_bytes': size_bytes,
                'upload_type': 'video' if is_video else 'photo',
                'duration_seconds': metadata['duration_seconds'] if is_video else None,
                'fps': metadata['fps'] if is_video else None,
                'frame_count': metadata['frame_count'] if is_video else None,
                'expires_at': expires_at,
            })
        connection.commit()
        return {'uploads': result}
    except HTTPException:
        connection.rollback()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    except (OSError, sqlite3.Error) as error:
        connection.rollback()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise HTTPException(500, f'Failed to store upload: {error}') from error
    finally:
        connection.close()


@app.delete('/api/uploads/{upload_id}')
def delete_upload(upload_id: str, request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        row = owned_upload(connection, upload_id, session)
        delete_upload_file(row)
        connection.execute('DELETE FROM uploads WHERE id=?', (upload_id,))
    return {'status': 'deleted'}


def create_modern_batch(request, response, files, upload_ids_text, settings_text, overrides_text, turnstile_token):
    rate_limit(request, 'batch', 30, 3600)
    try:
        base_settings = json.loads(settings_text or '{}')
        overrides = json.loads(overrides_text or '{}')
        upload_ids = json.loads(upload_ids_text or '[]')
    except json.JSONDecodeError as error:
        raise HTTPException(400, 'Invalid settings JSON') from error
    if not isinstance(overrides, dict):
        raise HTTPException(400, 'Overrides must be a JSON object')
    if not isinstance(upload_ids, list) or any(not isinstance(item, str) for item in upload_ids):
        raise HTTPException(400, 'Upload IDs must be a JSON array of strings')
    if len(set(upload_ids)) != len(upload_ids):
        raise HTTPException(400, 'Duplicate upload IDs are not allowed')
    files = files or []
    connection = connect()
    prepared = []
    created_paths = []
    transient_paths = []
    try:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        role = session.get('role') or ROLE_GUEST
        requested_upload_type = base_settings.get('upload_type', 'photo')
        if requested_upload_type == 'video' and role != ROLE_ADMIN:
            raise HTTPException(403, 'Video processing is restricted to administrators')
        batch_limit = 2 if role == ROLE_GUEST else MAX_BATCH_FILES
        item_count = len(files) + len(upload_ids)
        if item_count < 1 or item_count > batch_limit:
            raise HTTPException(400, f'A batch must contain 1 to {batch_limit} files')
        if role == ROLE_GUEST:
            verify_turnstile(turnstile_token, request)
        if shutil.disk_usage(INPUT_DIR).free < MIN_FREE_DISK_BYTES:
            raise HTTPException(503, 'Insufficient free disk space')

        timing_samples = processing_observations(connection)
        source_items = []
        for upload_id in upload_ids:
            row = owned_upload(connection, upload_id, session)
            source_items.append({
                'upload_id': row['id'],
                'original_name': row['original_name'],
                'temp_path': upload_file_path(row),
                'extension': row['extension'],
                'width': row['width'],
                'height': row['height'],
                'has_alpha': bool(row['has_alpha']),
                'upload_type': row['upload_type'] if 'upload_type' in row.keys() else 'photo',
                'duration_seconds': row['duration_seconds'] if 'duration_seconds' in row.keys() else None,
                'fps': row['fps'] if 'fps' in row.keys() else None,
                'frame_count': row['frame_count'] if 'frame_count' in row.keys() else None,
                'video_codec': row['video_codec'] if 'video_codec' in row.keys() else None,
                'audio_codec': row['audio_codec'] if 'audio_codec' in row.keys() else None,
            })
        for upload in files:
            is_video = upload_looks_video(upload)
            if is_video:
                if role != ROLE_ADMIN:
                    raise HTTPException(403, 'Video processing is restricted to administrators')
                incoming_path, size_bytes = store_upload_stream(upload, VIDEO_MAX_FILE_BYTES)
                metadata = probe_video(incoming_path)
                extension = Path(upload.filename or '').suffix.lower()
                if extension not in VIDEO_INPUT_EXTENSIONS:
                    extension = '.mp4'
                transient_paths.append(incoming_path)
                content = None
                width, height, has_alpha = metadata['width'], metadata['height'], False
            else:
                content = upload.file.read(MAX_FILE_BYTES + 1)
                extension, width, height, has_alpha = validate_image(content)
                metadata, size_bytes = {}, len(content)
            if content is None:
                temp_path = incoming_path
            else:
                temp_path = TMP_DIR / f'{uuid.uuid4().hex}.upload'
                temp_path.write_bytes(content)
                transient_paths.append(temp_path)
            source_items.append({
                'upload_id': None,
                'original_name': upload.filename or 'image',
                'temp_path': temp_path,
                'extension': extension,
                'width': width,
                'height': height,
                'has_alpha': has_alpha,
                'upload_type': 'video' if is_video else 'photo',
                **metadata,
            })

        for index, source in enumerate(source_items):
            extension = source['extension']
            width = source['width']
            height = source['height']
            has_alpha = source['has_alpha']
            merged = dict(base_settings)
            item_override = overrides.get(str(index), {})
            if isinstance(item_override, dict):
                merged.update(item_override)
            validated_settings = validate_settings(merged, role, has_alpha)
            if source.get('upload_type', 'photo') != validated_settings['upload_type']:
                raise HTTPException(400, 'Upload type does not match the selected media')
            selected_model = MODEL_REGISTRY[validated_settings['model_name']]
            if not selected_model['path'].is_file():
                raise HTTPException(503, f'{selected_model["label"]} model is not available')
            if validated_settings['face_enhance'] and not face_models_available():
                raise HTTPException(503, 'Face enhancement model is not available')
            output_width, output_height = target_dimensions(
                width,
                height,
                validated_settings['target_resolution'],
                validated_settings['aspect_ratio'],
                bool(validated_settings['crop_enabled']),
            )
            if validated_settings['upload_type'] == 'video':
                output_width -= output_width % 2
                output_height -= output_height % 2
            job_id = uuid.uuid4().hex
            output_name = f'{job_id}.{output_extension(validated_settings["output_format"])}'
            prepared.append(
                {
                    'id': job_id,
                    'upload_id': source['upload_id'],
                    'original_name': source['original_name'],
                    'temp_path': source['temp_path'],
                    'input_path': INPUT_DIR / f'{job_id}{extension}',
                    'output_path': OUTPUT_DIR / output_name,
                    'width': width,
                    'height': height,
                    'upload_type': source.get('upload_type', 'photo'),
                    'duration_seconds': source.get('duration_seconds'),
                    'fps': source.get('fps'),
                    'frame_count': source.get('frame_count'),
                    'video_codec': source.get('video_codec'),
                    'audio_codec': source.get('audio_codec'),
                    'output_width': output_width,
                    'output_height': output_height,
                    'estimated_seconds': estimate_processing_seconds(
                        width, height, bool(validated_settings['face_enhance']), observations=timing_samples,
                        settings={**validated_settings, 'output_width': output_width, 'output_height': output_height,
                                  'has_alpha': has_alpha, 'duration_seconds': source.get('duration_seconds'),
                                  'fps': source.get('fps'), 'frame_count': source.get('frame_count')},
                    ),
                    'estimated_output_bytes': estimate_video_output_bytes(
                        output_width, output_height, source.get('fps'), source.get('duration_seconds'),
                        validated_settings['output_format'],
                    ) if validated_settings['upload_type'] == 'video' else estimate_output_bytes(
                        output_width, output_height, validated_settings['output_format'],
                        validated_settings['quality_preset'], validated_settings['compression_level'],
                    ),
                    **validated_settings,
                }
            )

        connection.commit()
        estimated_output_bytes = sum(item['estimated_output_bytes'] for item in prepared)
        if shutil.disk_usage(OUTPUT_DIR).free < MIN_FREE_DISK_BYTES + estimated_output_bytes:
            raise HTTPException(503, 'Insufficient free disk space for the estimated output')
        connection.execute('BEGIN IMMEDIATE')
        active = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing') AND deleted_at IS NULL"
        ).fetchone()[0]
        clause, values = owner_clause(session)
        owner_active = connection.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing') AND deleted_at IS NULL AND {clause}",
            values,
        ).fetchone()[0]
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        owner_today = connection.execute(
            f'SELECT COUNT(*) FROM jobs WHERE created_at >= ? AND deleted_at IS NULL AND {clause}',
            (midnight, *values),
        ).fetchone()[0]
        daily_quota = GUEST_DAILY_QUOTA if role == ROLE_GUEST else int(session['daily_quota'])
        active_quota = 2 if role == ROLE_GUEST else int(session['active_quota'])
        if role != ROLE_GUEST and session.get('user_id'):
            user_row = connection.execute('SELECT image_quotas, video_quotas FROM users WHERE id=?', (session['user_id'],)).fetchone()
            for media, defaults, column in (('photo', DEFAULT_IMAGE_QUOTAS, 'image_quotas'), ('video', DEFAULT_VIDEO_QUOTAS, 'video_quotas')):
                try:
                    limits = {key: int(json.loads(user_row[column] or '{}').get(key, 0)) for key in defaults}
                except (TypeError, ValueError, json.JSONDecodeError):
                    limits = dict(defaults)
                for resolution, requested in __import__('collections').Counter(item['target_resolution'] for item in prepared if item['upload_type'] == media).items():
                    used = connection.execute(
                        f'SELECT COUNT(*) FROM jobs WHERE created_at >= ? AND deleted_at IS NULL AND target_resolution=? AND upload_type=? AND {clause}',
                        (midnight, resolution, media, *values),
                    ).fetchone()[0]
                    if limits.get(resolution, 0) <= 0 or used + requested > limits.get(resolution, 0):
                        raise HTTPException(429, f'{media} {resolution} quota exceeded')
        if active + len(prepared) > MAX_GLOBAL_JOBS:
            raise HTTPException(429, 'Global queue is full')
        if owner_active + len(prepared) > active_quota:
            raise HTTPException(429, 'Active job quota exceeded')
        if owner_today + len(prepared) > daily_quota:
            raise HTTPException(429, 'Daily job quota exceeded')

        batch_id = uuid.uuid4().hex
        expires_at = expiry_iso(session.get('user_id'))
        connection.execute(
            '''INSERT INTO batches(id, user_key, owner_session_id, owner_user_id, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (batch_id, session['id'], session['id'], session.get('user_id'), utc_now(), expires_at),
        )
        for item in prepared:
            os.replace(item['temp_path'], item['input_path'])
            created_paths.append(item['input_path'])
            connection.execute(
                '''INSERT INTO jobs(
                    id, batch_id, user_key, owner_session_id, owner_user_id, original_name,
                    input_path, output_path, width, height, status, created_at, expires_at,
                    model_name, target_resolution, aspect_ratio, crop_enabled, face_enhance,
                    output_format, quality_preset, compression_level, tile_size, output_width,
                    output_height, estimated_seconds, upload_type, duration_seconds, fps,
                    frame_count, video_codec, audio_codec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    item['id'], batch_id, session['id'], session['id'], session.get('user_id'),
                    item['original_name'], item['input_path'].name, item['output_path'].name,
                    item['width'], item['height'], utc_now(), expires_at, item['model_name'],
                    item['target_resolution'], item['aspect_ratio'], item['crop_enabled'],
                    item['face_enhance'], item['output_format'], item['quality_preset'],
                    item['compression_level'], item['tile_size'], item['output_width'],
                    item['output_height'], item['estimated_seconds'], item['upload_type'],
                    item.get('duration_seconds'), item.get('fps'), item.get('frame_count'),
                    item.get('video_codec'), item.get('audio_codec'),
                ),
            )
            if item['upload_id']:
                connection.execute('DELETE FROM uploads WHERE id=?', (item['upload_id'],))
        connection.commit()
        return {
            'batch_id': batch_id,
            'job_ids': [item['id'] for item in prepared],
            'estimated_processing_seconds': sum(item['estimated_seconds'] for item in prepared),
            'remaining_today': max(0, daily_quota - owner_today - len(prepared)),
        }
    except HTTPException:
        connection.rollback()
        restore_prepared_uploads(prepared, created_paths, transient_paths)
        raise
    except (OSError, sqlite3.Error) as error:
        connection.rollback()
        restore_prepared_uploads(prepared, created_paths, transient_paths)
        raise HTTPException(500, f'Failed to create batch: {error}') from error
    finally:
        connection.close()


@app.post('/api/batches', status_code=202)
def api_create_batch(
    request: Request,
    response: Response,
    files: list[UploadFile] | None = File(None),
    upload_ids: str = Form('[]'),
    settings: str = Form('{}'),
    overrides: str = Form('{}'),
    turnstile_token: str = Form(''),
):
    return create_modern_batch(request, response, files, upload_ids, settings, overrides, turnstile_token)


@app.get('/api/batches')
def list_batches(request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response)
        clause, values = owner_clause(session, 'b.')
        try:
            page = max(1, int(request.query_params.get('page', '1')))
            page_size = min(50, max(1, int(request.query_params.get('page_size', '5'))))
        except ValueError as error:
            raise HTTPException(400, 'Invalid pagination parameters') from error
        paged = 'page' in request.query_params or 'page_size' in request.query_params
        total = connection.execute(
            f'''SELECT COUNT(*) FROM batches b WHERE {clause}
                AND EXISTS (SELECT 1 FROM jobs j WHERE j.batch_id=b.id AND j.deleted_at IS NULL)''',
            values,
        ).fetchone()[0]
        batches = connection.execute(
            f'''SELECT b.* FROM batches b WHERE {clause}
                AND EXISTS (SELECT 1 FROM jobs j WHERE j.batch_id=b.id AND j.deleted_at IS NULL)
                ORDER BY b.created_at DESC LIMIT ? OFFSET ?''',
            (*values, page_size if paged else 100, (page - 1) * page_size if paged else 0),
        ).fetchall()
        result = []
        for batch in batches:
            jobs = connection.execute(
                'SELECT * FROM jobs WHERE batch_id=? AND deleted_at IS NULL ORDER BY sequence', (batch['id'],)
            ).fetchall()
            result.append({'id': batch['id'], 'created_at': batch['created_at'], 'jobs': [serialize_job(connection, row) for row in jobs]})
        if paged:
            return {'items': result, 'total': total, 'page': page, 'page_size': page_size,
                    'pages': max(1, math.ceil(total / page_size))}
        return result


@app.get('/api/batches/{batch_id}')
def api_get_batch(batch_id: str, request: Request, response: Response):
    with connect() as connection:
        batch = owned_batch(connection, batch_id, get_session(connection, request, response))
        jobs = connection.execute(
            'SELECT * FROM jobs WHERE batch_id=? AND deleted_at IS NULL ORDER BY sequence', (batch_id,)
        ).fetchall()
        return {'id': batch['id'], 'created_at': batch['created_at'], 'jobs': [serialize_job(connection, job) for job in jobs]}


@app.get('/api/jobs/{job_id}')
def api_get_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        return serialize_job(connection, row)


@app.get('/api/jobs/{job_id}/progress')
def api_get_job_progress(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        progress = int(row['progress'] or 0)
        return {
            'id': row['id'],
            'status': row['status'],
            'progress': progress,
            'estimated_seconds': row['estimated_seconds'],
            'remaining_seconds': job_remaining_seconds(connection, row, progress),
            'updated_at': row['heartbeat_at'] or row['created_at'],
        }


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        row = owned_job(connection, job_id, session)
        if row['status'] not in ('queued', 'processing'):
            raise HTTPException(409, 'Only queued or processing jobs can be cancelled')
        connection.execute(
            "UPDATE jobs SET status='cancelled', finished_at=?, lease_owner=NULL, lease_expires_at=NULL WHERE id=?",
            (utc_now(), job_id),
        )
    return {'status': 'cancelled'}


@app.post('/api/jobs/{job_id}/retry')
def retry_job(job_id: str, request: Request, response: Response):
    connection = connect()
    try:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        connection.commit()
        connection.execute('BEGIN IMMEDIATE')
        row = owned_job(connection, job_id, session)
        if row['status'] not in ('failed', 'cancelled'):
            raise HTTPException(409, 'Only failed or cancelled jobs can be retried')
        if row['upload_type'] == 'video' and session.get('role') != ROLE_ADMIN:
            raise HTTPException(403, 'Video processing is restricted to administrators')
        input_path = (INPUT_DIR / row['input_path']).resolve()
        if input_path.parent != INPUT_DIR.resolve() or input_path.is_symlink() or not input_path.is_file():
            raise HTTPException(410, 'Original input file is missing and cannot be retried')
        global_active = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing') AND deleted_at IS NULL"
        ).fetchone()[0]
        clause, values = owner_clause(session)
        owner_active = connection.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing') AND deleted_at IS NULL AND {clause}",
            values,
        ).fetchone()[0]
        active_quota = 2 if not session.get('user_id') else int(session['active_quota'])
        if global_active >= MAX_GLOBAL_JOBS:
            raise HTTPException(429, 'Global queue is full')
        if owner_active >= active_quota:
            raise HTTPException(429, 'Active job quota exceeded')
        connection.execute(
            "UPDATE jobs SET status='queued', error=NULL, attempts=0, progress=0, started_at=NULL, finished_at=NULL WHERE id=?",
            (job_id,),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {'status': 'queued'}


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        session = get_session(connection, request, response)
        require_csrf(request, session)
        if not session.get('user_id'):
            raise HTTPException(403, 'Guest tasks cannot be deleted')
        row = owned_job(connection, job_id, session)
        if row['status'] == 'processing':
            raise HTTPException(409, 'Processing jobs cannot be deleted')
        delete_job_files(row)
        connection.execute('DELETE FROM jobs WHERE id=?', (job_id,))
        connection.execute('DELETE FROM batches WHERE id=? AND NOT EXISTS (SELECT 1 FROM jobs WHERE batch_id=?)', (row['batch_id'], row['batch_id']))
    return {'status': 'deleted'}


def output_file(row):
    if row['status'] != 'succeeded':
        raise HTTPException(409, 'Job is not complete')
    path = (OUTPUT_DIR / row['output_path']).resolve()
    if path.parent != OUTPUT_DIR.resolve() or path.is_symlink() or not path.is_file():
        raise HTTPException(410, 'Output file is missing')
    return path


def preview_file(path, cache_key):
    """Create or reuse a bounded preview so compare views never stream originals."""
    try:
        source_stat = path.stat()
    except OSError as error:
        raise HTTPException(410, 'Preview source is missing') from error
    preview_dir = TMP_DIR / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    cache_path = preview_dir / f'{cache_key}.webp'
    if path.suffix.lower() in VIDEO_INPUT_EXTENSIONS | {'.mp4', '.webm'}:
        jpeg_path = preview_dir / f'{cache_key}.jpg'
        if jpeg_path.is_file() and jpeg_path.stat().st_mtime_ns >= source_stat.st_mtime_ns:
            return jpeg_path, 'image/jpeg'
        try:
            frame = subprocess.run(
                [FFMPEG_BIN, '-v', 'error', '-i', str(path), '-map', '0:v:0', '-frames:v', '1', '-vf', 'scale=1600:-2',
                 '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1'],
                check=True, capture_output=True, timeout=60,
            ).stdout
            if not frame:
                raise ValueError('empty video frame')
            temporary = preview_dir / f'.{cache_key}.{uuid.uuid4().hex}.tmp'
            temporary.write_bytes(frame)
            os.replace(temporary, jpeg_path)
            return jpeg_path, 'image/jpeg'
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise HTTPException(410, 'Preview is unavailable') from error
    try:
        if cache_path.is_file() and cache_path.stat().st_mtime_ns >= source_stat.st_mtime_ns:
            return cache_path, 'image/webp'
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            temporary = preview_dir / f'.{cache_key}.{uuid.uuid4().hex}.tmp'
            try:
                image.save(temporary, format='WEBP', quality=84, method=4)
                os.replace(temporary, cache_path)
                return cache_path, 'image/webp'
            except (OSError, ValueError):
                temporary.unlink(missing_ok=True)
                jpeg_path = preview_dir / f'{cache_key}.jpg'
                temporary = preview_dir / f'.{cache_key}.{uuid.uuid4().hex}.tmp'
                image.save(temporary, format='JPEG', quality=84, optimize=True)
                os.replace(temporary, jpeg_path)
                return jpeg_path, 'image/jpeg'
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(410, 'Preview is unavailable') from error


@app.get('/api/jobs/{job_id}/download')
def api_download_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        path = output_file(row)
        filename = f'{Path(row["original_name"]).stem[:80] or "image"}_superphoto.{output_extension(row["output_format"])}'
    return FileResponse(path, filename=filename)


@app.get('/api/jobs/{job_id}/preview')
def api_preview_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        path = output_file(row)
    preview, media_type = preview_file(path, f'{job_id}-result')
    return FileResponse(preview, media_type=media_type, headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/api/jobs/{job_id}/original')
def api_original_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        path = (INPUT_DIR / row['input_path']).resolve()
        if path.parent != INPUT_DIR.resolve() or path.is_symlink() or not path.is_file():
            raise HTTPException(410, 'Input file is missing')
        return FileResponse(path)


@app.get('/api/jobs/{job_id}/original-preview')
def api_original_preview(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        path = (INPUT_DIR / row['input_path']).resolve()
        if path.parent != INPUT_DIR.resolve() or path.is_symlink() or not path.is_file():
            raise HTTPException(410, 'Input file is missing')
    preview, media_type = preview_file(path, f'{job_id}-original')
    return FileResponse(preview, media_type=media_type, headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/api/jobs/{job_id}/thumbnail')
def api_job_thumbnail(job_id: str, request: Request, response: Response):
    with connect() as connection:
        row = owned_job(connection, job_id, get_session(connection, request, response))
        path = (INPUT_DIR / row['input_path']).resolve()
        if path.parent != INPUT_DIR.resolve() or path.is_symlink() or not path.is_file():
            raise HTTPException(410, 'Input file is missing')
    if path.suffix.lower() in {'.mp4', '.webm'}:
        thumbnail, media_type = preview_file(path, f'{job_id}-thumbnail')
        return FileResponse(thumbnail, media_type=media_type, headers={'Cache-Control': 'private, max-age=3600'})
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((160, 120), Image.Resampling.LANCZOS)
            thumbnail = io.BytesIO()
            image.save(thumbnail, format='JPEG', quality=78, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(410, 'Input thumbnail is unavailable') from error
    response.headers['Cache-Control'] = 'private, max-age=3600'
    return Response(content=thumbnail.getvalue(), media_type='image/jpeg', headers=dict(response.headers))


@app.get('/api/batches/{batch_id}/download')
def download_batch(batch_id: str, request: Request, response: Response, background_tasks: BackgroundTasks):
    rate_limit(request, 'batch_zip', 10, 3600)
    if shutil.disk_usage(TMP_DIR).free < MIN_FREE_DISK_BYTES:
        raise HTTPException(503, 'Insufficient free disk space for batch download')
    with connect() as connection:
        owned_batch(connection, batch_id, get_session(connection, request, response))
        rows = connection.execute(
            "SELECT * FROM jobs WHERE batch_id=? AND status='succeeded' AND deleted_at IS NULL ORDER BY sequence",
            (batch_id,),
        ).fetchall()
        if not rows:
            raise HTTPException(409, 'Batch has no completed files')
        zip_path = TMP_DIR / f'{batch_id}-{uuid.uuid4().hex}.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            used = set()
            for index, row in enumerate(rows, 1):
                path = output_file(row)
                name = f'{Path(row["original_name"]).stem[:70] or "image"}_superphoto.{output_extension(row["output_format"])}'
                if name in used:
                    name = f'{index}_{name}'
                used.add(name)
                archive.write(path, name)
    background_tasks.add_task(zip_path.unlink, missing_ok=True)
    return FileResponse(zip_path, media_type='application/zip', filename=f'superphoto-{batch_id[:8]}.zip')


@app.post('/api/jobs/{job_id}/share')
def create_share(job_id: str, request: Request, response: Response):
    with connect() as connection:
        session = require_user(get_session(connection, request, response))
        require_csrf(request, session)
        row = owned_job(connection, job_id, session)
        output_file(row)
        connection.execute('UPDATE shares SET revoked_at=? WHERE job_id=? AND revoked_at IS NULL', (utc_now(), job_id))
        token = random_token()
        expires = (datetime.now(timezone.utc) + timedelta(hours=SHARE_TTL_HOURS)).isoformat()
        connection.execute(
            'INSERT INTO shares(id, job_id, token_hash, created_by, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)',
            (uuid.uuid4().hex, job_id, token_hash(token), session['user_id'], utc_now(), expires),
        )
    return {'url': f'/share/{token}', 'expires_at': expires}


@app.delete('/api/jobs/{job_id}/share')
def revoke_share(job_id: str, request: Request, response: Response):
    with connect() as connection:
        session = require_user(get_session(connection, request, response))
        require_csrf(request, session)
        owned_job(connection, job_id, session)
        connection.execute('UPDATE shares SET revoked_at=? WHERE job_id=? AND revoked_at IS NULL', (utc_now(), job_id))
    return {'status': 'revoked'}


def public_share(connection, token):
    row = connection.execute(
        '''SELECT shares.*, jobs.original_name, jobs.output_path, jobs.output_format,
                  jobs.output_width, jobs.output_height, jobs.model_name, jobs.target_resolution,
                  jobs.status
           FROM shares JOIN jobs ON jobs.id=shares.job_id
           WHERE shares.token_hash=? AND shares.revoked_at IS NULL AND shares.expires_at > ?''',
        (token_hash(token), utc_now()),
    ).fetchone()
    if not row or row['status'] != 'succeeded':
        raise HTTPException(404, 'Share not found')
    return row


@app.get('/api/shares/{token}')
def share_metadata(token: str):
    with connect() as connection:
        row = public_share(connection, token)
        return {
            'original_name': row['original_name'],
            'width': row['output_width'],
            'height': row['output_height'],
            'model': row['model_name'],
            'target_resolution': row['target_resolution'],
            'output_format': row['output_format'],
            'expires_at': row['expires_at'],
            'download_url': f'/api/shares/{token}/download',
            'preview_url': f'/api/shares/{token}/preview',
        }


@app.get('/api/shares/{token}/download')
def share_download(token: str):
    with connect() as connection:
        row = public_share(connection, token)
        path = output_file(row)
        filename = f'{Path(row["original_name"]).stem[:80] or "image"}_superphoto.{output_extension(row["output_format"])}'
    return FileResponse(path, filename=filename)


@app.get('/api/shares/{token}/preview')
def share_preview(token: str):
    with connect() as connection:
        row = public_share(connection, token)
        path = output_file(row)
    preview, media_type = preview_file(path, f'share-{token_hash(token)[:24]}')
    return FileResponse(preview, media_type=media_type, headers={'Cache-Control': 'public, max-age=3600'})


@app.get('/share/{token}')
def share_page(token: str):
    with connect() as connection:
        public_share(connection, token)
    return FileResponse(STATIC_DIR / 'share.html')


def admin_session(connection, request, response, csrf=False):
    session = require_admin(get_session(connection, request, response))
    if csrf:
        require_csrf(request, session)
    return session


@app.get('/api/admin/users')
def admin_users(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
        rows = connection.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
        return [serialize_user(row) for row in rows]


@app.patch('/api/admin/users/{user_id}')
def admin_update_user(user_id: str, payload: UserUpdatePayload, request: Request, response: Response):
    with connect() as connection:
        session = admin_session(connection, request, response, csrf=True)
        row = connection.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'User not found')
        role = payload.role if payload.role is not None else row['role']
        if role not in VALID_ROLES:
            raise HTTPException(400, 'Invalid role')
        daily = payload.daily_quota if payload.daily_quota is not None else row['daily_quota']
        active = payload.active_quota if payload.active_quota is not None else row['active_quota']
        disabled = int(payload.disabled) if payload.disabled is not None else row['disabled']
        if not 1 <= daily <= 1000 or not 1 <= active <= 100:
            raise HTTPException(400, 'Quota is outside the allowed range')
        if user_id == session['user_id'] and (role != ROLE_ADMIN or disabled):
            raise HTTPException(409, 'Administrator cannot remove their own access')
        updates = {'role': role, 'daily_quota': daily, 'active_quota': active, 'disabled': disabled, 'updated_at': utc_now()}
        def normalize_quotas(value, defaults, field):
            if value is None:
                try:
                    current = json.loads(row[field] or '{}') if field in row.keys() else defaults
                except (TypeError, ValueError, json.JSONDecodeError):
                    current = defaults
            else:
                current = value
            if set(current) - set(defaults) or any(not isinstance(v, int) or v < 0 or v > 10000 for v in current.values()):
                raise HTTPException(400, f'Invalid {field}')
            return json.dumps({key: int(current.get(key, 0)) for key in defaults}, separators=(',', ':'))
        updates['image_quotas'] = normalize_quotas(payload.image_quotas, DEFAULT_IMAGE_QUOTAS, 'image_quotas')
        updates['video_quotas'] = normalize_quotas(payload.video_quotas, DEFAULT_VIDEO_QUOTAS, 'video_quotas')
        # Legacy admin controls map to the canonical per-resolution JSON fields.
        if payload.daily_image_quota is not None or any(getattr(payload, f'resolution_{key}') is not None for key in ('1k', '2k', '4k')):
            image = json.loads(updates['image_quotas'])
            if payload.daily_image_quota is not None:
                image = {key: payload.daily_image_quota for key in image}
            for key in ('1k', '2k', '4k'):
                flag = getattr(payload, f'resolution_{key}')
                if flag is not None and key in image and flag == 0:
                    image[key] = 0
            updates['image_quotas'] = json.dumps(image, separators=(',', ':'))
        if payload.daily_video_quota is not None or any(getattr(payload, f'resolution_{key}') is not None for key in ('1k', '2k', '4k')):
            video = json.loads(updates['video_quotas'])
            if payload.daily_video_quota is not None:
                video = {key: payload.daily_video_quota for key in video}
            for key in ('1k', '2k', '4k'):
                flag = getattr(payload, f'resolution_{key}')
                if flag is not None and flag == 0:
                    video[key] = 0
            updates['video_quotas'] = json.dumps(video, separators=(',', ':'))
        optional = ('daily_image_quota', 'daily_video_quota', 'resolution_1k', 'resolution_2k', 'resolution_4k')
        columns = {row[1] for row in connection.execute('PRAGMA table_info(users)').fetchall()}
        for name in optional:
            value = getattr(payload, name)
            if value is not None:
                if value < 0 or value > 10000:
                    raise HTTPException(400, f'{name} is outside the allowed range')
                if name.startswith('resolution_') and value not in (0, 1):
                    raise HTTPException(400, f'{name} must be 0 or 1')
                if name not in columns:
                    raise HTTPException(409, 'User permission fields require database migration')
                updates[name] = value
        assignments = ', '.join(f'{key}=?' for key in updates)
        connection.execute(f'UPDATE users SET {assignments} WHERE id=?', (*updates.values(), user_id))
        if disabled:
            connection.execute('DELETE FROM sessions WHERE user_id=?', (user_id,))
    return {'status': 'updated'}


@app.post('/api/admin/users/{user_id}/password')
def admin_reset_password(user_id: str, payload: ResetPasswordPayload, request: Request, response: Response):
    try:
        encoded = hash_password(payload.new_password)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        cursor = connection.execute(
            'UPDATE users SET password_hash=?, updated_at=? WHERE id=?', (encoded, utc_now(), user_id)
        )
        if cursor.rowcount != 1:
            raise HTTPException(404, 'User not found')
        connection.execute('DELETE FROM sessions WHERE user_id=?', (user_id,))
    return {'status': 'password_reset'}


@app.get('/api/invites')
@app.get('/api/admin/invites')
def list_invites(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
        rows = connection.execute(
            '''SELECT invites.id, invites.expires_at, invites.used_at, invites.revoked_at,
                      invites.created_at, users.username AS used_by_username
               FROM invites LEFT JOIN users ON users.id=invites.used_by ORDER BY invites.created_at DESC'''
        ).fetchall()
        return [dict(row) for row in rows]


@app.post('/api/invites', status_code=201)
@app.post('/api/admin/invites', status_code=201)
def create_invite(payload: InvitePayload, request: Request, response: Response):
    with connect() as connection:
        session = admin_session(connection, request, response, csrf=True)
        if payload.expires_hours is not None and not 1 <= payload.expires_hours <= 24 * 365:
            raise HTTPException(400, 'Invitation expiry is outside the allowed range')
        code = random_token(18)
        expires = None
        if payload.expires_hours is not None:
            expires = (datetime.now(timezone.utc) + timedelta(hours=payload.expires_hours)).isoformat()
        invite_id = uuid.uuid4().hex
        connection.execute(
            'INSERT INTO invites(id, code_hash, created_by, expires_at, created_at) VALUES (?, ?, ?, ?, ?)',
            (invite_id, token_hash(code), session['user_id'], expires, utc_now()),
        )
    return {'id': invite_id, 'code': code, 'expires_at': expires}


@app.delete('/api/invites/{invite_id}')
@app.delete('/api/admin/invites/{invite_id}')
def revoke_invite(invite_id: str, request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        cursor = connection.execute(
            'UPDATE invites SET revoked_at=? WHERE id=? AND used_at IS NULL', (utc_now(), invite_id)
        )
        if cursor.rowcount != 1:
            raise HTTPException(404, 'Active invitation not found')
    return {'status': 'revoked'}


@app.get('/api/admin/jobs')
def admin_jobs(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
        try:
            page = max(1, int(request.query_params.get('page', '1')))
            page_size = min(50, max(1, int(request.query_params.get('page_size', '5'))))
        except ValueError as error:
            raise HTTPException(400, 'Invalid pagination parameters') from error
        paged = 'page' in request.query_params or 'page_size' in request.query_params
        total = connection.execute(
            'SELECT COUNT(*) FROM jobs WHERE deleted_at IS NULL'
        ).fetchone()[0]
        rows = connection.execute(
            '''SELECT jobs.*, users.username FROM jobs
               LEFT JOIN users ON users.id=jobs.owner_user_id
               WHERE jobs.deleted_at IS NULL ORDER BY jobs.sequence DESC LIMIT ? OFFSET ?''',
            (page_size if paged else 200, (page - 1) * page_size if paged else 0),
        ).fetchall()
        items = [{**serialize_job(connection, row), 'username': row['username'] or '访客'} for row in rows]
        if paged:
            return {'items': items, 'total': total, 'page': page, 'page_size': page_size,
                    'pages': max(1, math.ceil(total / page_size))}
        return items


@app.delete('/api/admin/jobs')
def admin_clear_jobs(request: Request, response: Response):
    """Clear finished task history; queued/processing jobs are preserved."""
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        rows = connection.execute("SELECT id, input_path, output_path FROM jobs WHERE status IN ('succeeded','failed','cancelled') AND deleted_at IS NULL").fetchall()
        now = utc_now()
        connection.execute("UPDATE jobs SET deleted_at=?, finished_at=COALESCE(finished_at, ?) WHERE status IN ('succeeded','failed','cancelled') AND deleted_at IS NULL", (now, now))
    return {'status': 'cleared', 'count': len(rows)}


@app.delete('/api/admin/audit-log')
def admin_clear_audit_log(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        cursor = connection.execute('DELETE FROM audit_log')
    return {'status': 'cleared', 'count': cursor.rowcount}


@app.delete('/api/admin/invites')
def admin_clear_invites(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        cursor = connection.execute('DELETE FROM invites WHERE used_at IS NOT NULL OR revoked_at IS NOT NULL OR (expires_at IS NOT NULL AND expires_at < ?)', (utc_now(),))
    return {'status': 'cleared', 'count': cursor.rowcount}


@app.post('/api/admin/jobs/{job_id}/cancel')
def admin_cancel_job(job_id: str, request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response, csrf=True)
        row = connection.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Job not found')
        if row['status'] != 'queued':
            raise HTTPException(409, 'Only queued jobs can be cancelled safely')
        connection.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?", (utc_now(), job_id))
    return {'status': 'cancelled'}


@app.get('/api/admin/health')
def admin_health(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
        counts = {
            row['status']: row['count']
            for row in connection.execute(
                'SELECT status, COUNT(*) AS count FROM jobs WHERE deleted_at IS NULL GROUP BY status'
            )
        }
        users = connection.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    return {**service_health(), 'job_counts': counts, 'users': users, 'disk_free': shutil.disk_usage(INPUT_DIR).free}


@app.get('/api/openapi.json')
def protected_openapi(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
    return app.openapi()


@app.get('/api/docs')
def protected_docs(request: Request, response: Response):
    with connect() as connection:
        admin_session(connection, request, response)
    return get_swagger_ui_html(openapi_url='/api/openapi.json', title='SuperPhoto API')


# Legacy API compatibility. These routes preserve the original local/IP-scoped contract.
def require_local_legacy(request):
    if request.headers.get('CF-Ray') or request.headers.get('CF-Connecting-IP'):
        raise HTTPException(404, 'Not found')


@app.post('/batches', status_code=202)
async def legacy_create_batch(request: Request, files: list[UploadFile] = File(...)):
    require_local_legacy(request)
    rate_limit(request, 'legacy_batch', 10, 3600)
    if not files or len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f'A batch must contain 1 to {MAX_BATCH_FILES} files')
    validated = []
    for upload in files:
        content = await upload.read(MAX_FILE_BYTES + 1)
        extension, width, height, _ = validate_image(content)
        validated.append((upload.filename or 'image', content, extension, width, height))
    user_key = request.client.host if request.client else 'local'
    batch_id = uuid.uuid4().hex
    created = []
    with connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        if connection.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0] + len(validated) > MAX_GLOBAL_JOBS:
            raise HTTPException(429, 'Global queue is full')
        connection.execute('INSERT INTO batches(id, user_key, created_at) VALUES (?, ?, ?)', (batch_id, user_key, utc_now()))
        job_ids = []
        try:
            for name, content, extension, width, height in validated:
                job_id = uuid.uuid4().hex
                input_path = INPUT_DIR / f'{job_id}{extension}'
                output_path = OUTPUT_DIR / f'{job_id}.png'
                input_path.write_bytes(content)
                created.append(input_path)
                connection.execute(
                    '''INSERT INTO jobs(id,batch_id,user_key,original_name,input_path,output_path,width,height,status,created_at,
                       output_width,output_height,output_format,expires_at)
                       VALUES (?,?,?,?,?,?,?,?,'queued',?,?,?,'png',?)''',
                    (job_id,batch_id,user_key,name,input_path.name,output_path.name,width,height,utc_now(),width*4,height*4,expiry_iso(None)),
                )
                job_ids.append(job_id)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return {'batch_id': batch_id, 'job_ids': job_ids}


@app.get('/batches/{batch_id}')
def legacy_get_batch(batch_id: str, request: Request):
    require_local_legacy(request)
    with connect() as connection:
        batch = connection.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
        if not batch:
            raise HTTPException(404, 'Batch not found')
        jobs = connection.execute('SELECT * FROM jobs WHERE batch_id=? ORDER BY sequence', (batch_id,)).fetchall()
        return {'id': batch['id'], 'created_at': batch['created_at'], 'jobs': [serialize_job(connection, row) for row in jobs]}


@app.get('/jobs/{job_id}')
def legacy_get_job(job_id: str, request: Request):
    require_local_legacy(request)
    with connect() as connection:
        row = connection.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Job not found')
        return serialize_job(connection, row)


@app.get('/jobs/{job_id}/download')
def legacy_download_job(job_id: str, request: Request):
    require_local_legacy(request)
    with connect() as connection:
        row = connection.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row or row['user_key'] != (request.client.host if request.client else 'local'):
            raise HTTPException(404, 'Job not found')
        path = output_file(row)
        filename = f'{Path(row["original_name"]).stem[:80] or "image"}_x4.{output_extension(row["output_format"])}'
    return FileResponse(path, filename=filename)


if STATIC_DIR.is_dir():
    app.mount('/assets', StaticFiles(directory=STATIC_DIR), name='assets')
