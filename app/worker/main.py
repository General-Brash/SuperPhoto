import os
import multiprocessing
import signal
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import cv2

from app.common.config import GFPGAN_MODEL_PATH, INPUT_DIR, MODEL_REGISTRY, MODEL_PATH, OUTPUT_DIR, TMP_DIR, ensure_directories, face_models_available
from app.common.db import connect, init_db, utc_now
from app.common.jobs import QUALITY_VALUES, crop_box
from app.inference import OpenVINORealESRGANer


def set_worker_state(slot, value):
    with connect() as connection:
        connection.execute(
            '''INSERT INTO service_state(key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at''',
            (f'worker:{slot}', value, utc_now()),
        )


LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15


class JobLeaseLost(Exception):
    """The job was cancelled, reclaimed, or its lease could not be renewed."""


def lease_deadline():
    return (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)).isoformat()


def claim_job(lease_owner):
    connection = connect()
    try:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(
            '''UPDATE jobs SET status='failed', error='Worker stopped repeatedly', finished_at=?,
               lease_owner=NULL, lease_expires_at=NULL
               WHERE status='processing' AND lease_expires_at < ? AND attempts >= 2''',
            (utc_now(), utc_now()),
        )
        job = connection.execute(
            '''SELECT * FROM jobs
               WHERE (status='queued' AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at > ?))
                  OR (status='processing' AND lease_expires_at < ? AND attempts < 2)
               ORDER BY sequence LIMIT 1''',
            (utc_now(), utc_now()),
        ).fetchone()
        if not job:
            connection.commit()
            return None
        connection.execute(
            '''UPDATE jobs SET status='processing', attempts=attempts+1, progress=0,
               started_at=COALESCE(started_at, ?), error=NULL, lease_owner=?,
               lease_expires_at=?, heartbeat_at=? WHERE id=?''',
            (utc_now(), lease_owner, lease_deadline(), utc_now(), job['id']),
        )
        connection.commit()
        return dict(job)
    finally:
        connection.close()


def finish_job(job_id, lease_owner, status, error=None):
    with connect() as connection:
        cursor = connection.execute(
            '''UPDATE jobs SET status=?, error=?, finished_at=?, progress=?, lease_owner=NULL,
               lease_expires_at=NULL WHERE id=? AND lease_owner=? AND status='processing' ''',
            (status, error, utc_now(), 100 if status == 'succeeded' else 0, job_id, lease_owner),
        )
        return cursor.rowcount == 1


def renew_lease(job_id, lease_owner):
    with connect() as connection:
        cursor = connection.execute(
            '''UPDATE jobs SET lease_expires_at=?, heartbeat_at=?
               WHERE id=? AND lease_owner=? AND status='processing' ''',
            (lease_deadline(), utc_now(), job_id, lease_owner),
        )
        return cursor.rowcount == 1


def update_progress(job_id, lease_owner, progress):
    with connect() as connection:
        cursor = connection.execute(
            '''UPDATE jobs SET progress=?, heartbeat_at=?
               WHERE id=? AND lease_owner=? AND status='processing' ''',
            (max(0, min(99, int(progress))), utc_now(), job_id, lease_owner),
        )
        return cursor.rowcount == 1


def heartbeat_loop(stop_event, lost_event, job_id, lease_owner):
    while not stop_event.wait(HEARTBEAT_SECONDS):
        try:
            renewed = renew_lease(job_id, lease_owner)
        except Exception:
            # If SQLite cannot confirm ownership, abandon work instead of publishing it.
            lost_event.set()
            return
        if not renewed:
            lost_event.set()
            return


def expected_dimensions(job):
    return (
        int(job.get('output_height') or job['height'] * 4),
        int(job.get('output_width') or job['width'] * 4),
    )


def valid_output(path, job):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.shape[0:2] == expected_dimensions(job)


def create_upsampler(model_name, tile, threads):
    model = MODEL_REGISTRY.get(model_name)
    if not model:
        raise ValueError(f'Unknown model: {model_name}')
    if not model['path'].is_file():
        raise FileNotFoundError(f'OpenVINO model is missing: {model["path"]}')
    return OpenVINORealESRGANer(
        scale=4,
        model_path=str(model['path']),
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        threads=threads,
    )


def create_face_enhancer():
    if not face_models_available():
        raise FileNotFoundError('GFPGAN auxiliary face models are missing')
    try:
        from gfpgan import GFPGANer
    except ImportError as error:
        raise RuntimeError('GFPGAN runtime dependency is not installed') from error
    original_cwd = os.getcwd()
    try:
        os.chdir(GFPGAN_MODEL_PATH.parent)
        return GFPGANer(
            model_path=str(GFPGAN_MODEL_PATH),
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
    finally:
        os.chdir(original_cwd)


def enhance_faces(image, enhancer):
    alpha = None
    source = image
    if image.ndim == 3 and image.shape[2] == 4:
        source = image[:, :, :3]
        alpha = image[:, :, 3]
    _, _, restored = enhancer.enhance(source, has_aligned=False, only_center_face=False, paste_back=True)
    if restored is None:
        raise RuntimeError('Face enhancement did not produce an output')
    if alpha is not None:
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2BGRA)
        restored[:, :, 3] = alpha
    return restored


def encode_output(path, image, output_format, quality, compression_level=5):
    parameters = []
    if output_format == 'jpeg':
        parameters = [cv2.IMWRITE_JPEG_QUALITY, QUALITY_VALUES['jpeg'][quality]]
    elif output_format == 'webp':
        parameters = [cv2.IMWRITE_WEBP_QUALITY, QUALITY_VALUES['webp'][quality]]
    elif output_format == 'png':
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError('Failed to encode output image')


def process_job(upsampler, face_enhancer, job, lease_owner, lost_event):
    started_clock = time.perf_counter()
    metrics = {'decode_ms': 0, 'sr_ms': 0, 'resize_ms': 0, 'face_ms': 0, 'encode_ms': 0, 'verify_ms': 0}
    input_path = INPUT_DIR / job['input_path']
    output_path = OUTPUT_DIR / job['output_path']
    if output_path.is_file() and valid_output(output_path, job):
        if lost_event.is_set():
            raise JobLeaseLost()
        return None, None

    stage_started = time.perf_counter()
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    metrics['decode_ms'] = (time.perf_counter() - stage_started) * 1000
    if image is None:
        raise ValueError('Input image could not be decoded')
    if job.get('crop_enabled') and job.get('aspect_ratio') != 'original':
        left, top, right, bottom = crop_box(image.shape[1], image.shape[0], job['aspect_ratio'])
        image = image[top:bottom, left:right]
    tile_passes = 2 if image.ndim == 3 and image.shape[2] == 4 else 1
    completed_tiles = 0
    last_progress = 4

    def check_progress(progress):
        if lost_event.is_set() or not update_progress(job['id'], lease_owner, progress):
            raise JobLeaseLost()

    def report_tile_progress(_tile_index, total_tiles):
        nonlocal completed_tiles, last_progress
        if lost_event.is_set():
            raise JobLeaseLost()
        completed_tiles += 1
        progress = int(5 + 80 * completed_tiles / (total_tiles * tile_passes))
        if progress > last_progress:
            check_progress(progress)
            last_progress = progress

    check_progress(5)
    upsampler.progress_callback = report_tile_progress
    stage_started = time.perf_counter()
    try:
        output, _ = upsampler.enhance(image, outscale=4)
    finally:
        upsampler.progress_callback = None
    metrics['sr_ms'] = (time.perf_counter() - stage_started) * 1000
    if lost_event.is_set():
        raise JobLeaseLost()

    expected_height, expected_width = expected_dimensions(job)
    stage_started = time.perf_counter()
    if output.shape[0:2] != (expected_height, expected_width):
        interpolation = cv2.INTER_AREA if output.shape[0] > expected_height else cv2.INTER_LANCZOS4
        output = cv2.resize(output, (expected_width, expected_height), interpolation=interpolation)
    metrics['resize_ms'] = (time.perf_counter() - stage_started) * 1000
    if job.get('face_enhance'):
        stage_started = time.perf_counter()
        check_progress(88)
        output = enhance_faces(output, face_enhancer)
        if output.shape[0:2] != (expected_height, expected_width):
            output = cv2.resize(output, (expected_width, expected_height), interpolation=cv2.INTER_LANCZOS4)
        metrics['face_ms'] = (time.perf_counter() - stage_started) * 1000

    suffix = 'jpg' if job.get('output_format') == 'jpeg' else job.get('output_format', 'png')
    temporary_path = TMP_DIR / f"{job['id']}.{lease_owner}.partial.{suffix}"
    check_progress(95)
    stage_started = time.perf_counter()
    encode_output(
        temporary_path,
        output,
        job.get('output_format', 'png'),
        job.get('quality_preset', 'high'),
        int(job.get('compression_level') or 5),
    )
    metrics['encode_ms'] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    verified = cv2.imread(str(temporary_path), cv2.IMREAD_UNCHANGED)
    if verified is None or verified.shape[0:2] != (expected_height, expected_width):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError('Output image validation failed')
    metrics['verify_ms'] = (time.perf_counter() - stage_started) * 1000
    check_progress(99)
    metrics['total_ms'] = (time.perf_counter() - started_clock) * 1000
    metrics['output_bytes'] = temporary_path.stat().st_size
    return temporary_path, metrics


def publish_success(job, lease_owner, temporary_path, metrics, worker_slot):
    output_path = OUTPUT_DIR / job['output_path']
    connection = connect()
    try:
        connection.execute('BEGIN IMMEDIATE')
        owned = connection.execute(
            "SELECT 1 FROM jobs WHERE id=? AND lease_owner=? AND status='processing'",
            (job['id'], lease_owner),
        ).fetchone()
        if not owned:
            connection.rollback()
            if temporary_path:
                temporary_path.unlink(missing_ok=True)
            return False
        if temporary_path:
            os.replace(temporary_path, output_path)
        connection.execute(
            '''UPDATE jobs SET status='succeeded', error=NULL, finished_at=?, progress=100,
               lease_owner=NULL, lease_expires_at=NULL WHERE id=? AND lease_owner=?''',
            (utc_now(), job['id'], lease_owner),
        )
        if metrics:
            connection.execute(
                '''INSERT INTO job_metrics(job_id, worker_slot, decode_ms, sr_ms, resize_ms, face_ms,
                   encode_ms, verify_ms, total_ms, output_bytes, concurrent_jobs, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     (SELECT COUNT(*) FROM jobs WHERE status='processing'), ?)''',
                (job['id'], worker_slot, metrics.get('decode_ms', 0), metrics.get('sr_ms', 0),
                 metrics.get('resize_ms', 0), metrics.get('face_ms', 0), metrics.get('encode_ms', 0),
                 metrics.get('verify_ms', 0), metrics.get('total_ms', 0), metrics.get('output_bytes'), utc_now()),
            )
        connection.commit()
        return True
    finally:
        connection.close()


def run_claimed_job(job, lease_owner, slot, threads, upsamplers, face_enhancer):
    set_worker_state(slot, f"processing:{job['id']}")
    stop_event = threading.Event()
    lost_event = threading.Event()
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(stop_event, lost_event, job['id'], lease_owner),
        daemon=True,
    )
    heartbeat.start()
    temporary_path = None
    metrics = None
    failure = None
    lease_lost = False
    try:
        model_key = (job.get('model_name') or 'general', int(job.get('tile_size') or 256))
        if model_key not in upsamplers:
            upsamplers[model_key] = create_upsampler(*model_key, threads)
            while len(upsamplers) > 4:
                upsamplers.popitem(last=False)
        else:
            upsamplers.move_to_end(model_key)
        upsampler = upsamplers[model_key]
        if job.get('face_enhance') and face_enhancer is None:
            face_enhancer = create_face_enhancer()
        temporary_path, metrics = process_job(upsampler, face_enhancer, job, lease_owner, lost_event)
    except JobLeaseLost:
        lease_lost = True  # Cancellation and lease transfer are owned by the API/reclaimer.
    except Exception as error:
        failure = error
    finally:
        stop_event.set()
        heartbeat.join()  # Never publish while a renewal can still be in flight.

    try:
        if not lost_event.is_set() and failure is not None:
            finish_job(job['id'], lease_owner, 'failed', str(failure)[:1000])
        elif not lost_event.is_set() and not lease_lost and failure is None:
            # publish_success verifies ownership inside the same transaction as os.replace.
            publish_success(job, lease_owner, temporary_path, metrics, slot)
    finally:
        for partial in TMP_DIR.glob(f"{job['id']}.{lease_owner}.partial.*"):
            partial.unlink(missing_ok=True)
    return face_enhancer


def worker_loop(slot):
    ensure_directories()
    init_db()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'OpenVINO model is missing: {MODEL_PATH}')

    threads = int(os.getenv('OPENVINO_NUM_THREADS', '4'))
    upsamplers = OrderedDict({('general', 256): create_upsampler('general', 256, threads)})
    face_enhancer = None
    set_worker_state(slot, 'idle')
    last_heartbeat = 0.0
    lease_owner = uuid.uuid4().hex

    while True:
        job = claim_job(lease_owner)
        if not job:
            if time.monotonic() - last_heartbeat >= 5:
                set_worker_state(slot, 'idle')
                last_heartbeat = time.monotonic()
            time.sleep(1)
            continue

        face_enhancer = run_claimed_job(job, lease_owner, slot, threads, upsamplers, face_enhancer)
        set_worker_state(slot, 'idle')


def main():
    concurrency = int(os.getenv('WORKER_CONCURRENCY', '1'))
    if concurrency < 1:
        raise ValueError('WORKER_CONCURRENCY must be at least 1')
    if concurrency == 1:
        worker_loop(1)
        return

    context = multiprocessing.get_context('spawn')
    processes = [
        context.Process(target=worker_loop, args=(slot,), name=f'realesrgan-worker-{slot}')
        for slot in range(1, concurrency + 1)
    ]

    def stop_workers(*_):
        for process in processes:
            if process.is_alive():
                process.terminate()

    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process for process in processes if process.exitcode not in (0, -signal.SIGTERM)]
    if failed:
        raise RuntimeError('One or more worker processes exited unexpectedly')


if __name__ == '__main__':
    main()
