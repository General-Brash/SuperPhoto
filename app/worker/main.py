import os
import multiprocessing
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np

from app.common.config import FFMPEG_BIN, FFPROBE_BIN, GFPGAN_MODEL_PATH, INPUT_DIR, MIN_FREE_DISK_BYTES, MODEL_REGISTRY, MODEL_PATH, OUTPUT_DIR, TMP_DIR, ensure_directories, face_models_available
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


LEASE_SECONDS = 30


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
               WHERE ((status='queued' AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at > ?))
                  OR (status='processing' AND lease_expires_at < ? AND attempts < 2))
                 AND (upload_type!='video' OR NOT EXISTS (
                     SELECT 1 FROM jobs active_video
                     WHERE active_video.status='processing' AND active_video.upload_type='video'
                       AND active_video.lease_expires_at >= ?
                 ))
               ORDER BY sequence LIMIT 1''',
            (utc_now(), utc_now(), utc_now()),
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


def heartbeat_loop(stop_event, job_id, lease_owner):
    while not stop_event.wait(10):
        if not renew_lease(job_id, lease_owner):
            return


def expected_dimensions(job):
    return (
        int(job.get('output_height') or job['height'] * 4),
        int(job.get('output_width') or job['width'] * 4),
    )


def valid_output(path, job):
    if job.get('upload_type') == 'video':
        try:
            result = subprocess.run(
                [FFPROBE_BIN, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                 'stream=width,height', '-of', 'csv=p=0:s=x', str(path)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            return result.stdout.strip() == f"{job.get('output_width')}x{job.get('output_height')}"
        except (OSError, subprocess.SubprocessError):
            return False
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


def process_video_job(upsampler, face_enhancer, job, lease_owner):
    input_path = INPUT_DIR / job['input_path']
    output_width, output_height = expected_dimensions(job)[1], expected_dimensions(job)[0]
    fps = float(job.get('fps') or 24.0)
    frame_count = int(job.get('frame_count') or 0)
    output_format = job.get('output_format') or 'mp4'
    suffix = 'mp4' if output_format == 'mp4' else 'webm'
    temporary_path = TMP_DIR / f"{job['id']}.{lease_owner}.partial.{suffix}"
    source_width, source_height = int(job['width']), int(job['height'])
    decode_command = [FFMPEG_BIN, '-v', 'error', '-i', str(input_path), '-f', 'rawvideo',
                      '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1']
    decode = subprocess.Popen(
        decode_command,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if output_format == 'webm':
        video_args = ['-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0', '-pix_fmt', 'yuv420p',
                      '-c:a', 'libopus', '-b:a', '128k']
    else:
        video_args = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                      '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart']
    encode = subprocess.Popen(
        [FFMPEG_BIN, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-s', f'{output_width}x{output_height}', '-r', str(fps), '-i', 'pipe:0',
         '-i', str(input_path), '-map', '0:v:0', '-map', '1:a:0?', '-shortest', *video_args,
         str(temporary_path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frame_bytes = source_width * source_height * 3
    processed = 0
    started = time.perf_counter()
    try:
        while True:
            data = decode.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise RuntimeError('Video decoder returned a partial frame')
            image = np.frombuffer(data, dtype=np.uint8).reshape((source_height, source_width, 3))
            if job.get('crop_enabled') and job.get('aspect_ratio') != 'original':
                left, top, right, bottom = crop_box(image.shape[1], image.shape[0], job['aspect_ratio'])
                image = image[top:bottom, left:right]
            output, _ = upsampler.enhance(image, outscale=4)
            if output.shape[0:2] != (output_height, output_width):
                output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
            if job.get('face_enhance'):
                output = enhance_faces(output, face_enhancer)
            encode.stdin.write(output.astype('uint8').tobytes())
            processed += 1
            if processed % 30 == 0 and shutil.disk_usage(TMP_DIR).free < MIN_FREE_DISK_BYTES:
                raise RuntimeError('Video processing stopped because free disk space is below the safety reserve')
            if frame_count:
                if not update_progress(job['id'], lease_owner, 5 + int(90 * processed / frame_count)):
                    raise RuntimeError('Video job was cancelled')
    except Exception:
        decode.kill()
        encode.kill()
        raise
    finally:
        if decode.stdout:
            decode.stdout.close()
        try:
            decode.wait(timeout=30)
        except subprocess.TimeoutExpired:
            decode.kill()
            decode.wait()
        if encode.stdin:
            encode.stdin.close()
        try:
            encode.wait(timeout=120)
        except subprocess.TimeoutExpired:
            encode.kill()
            encode.wait()
    if decode.returncode != 0:
        raise RuntimeError('Video decode failed')
    if encode.returncode != 0:
        raise RuntimeError('Video encode failed')
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise RuntimeError('Video output validation failed')
    return temporary_path, {
        'decode_ms': 0,
        'sr_ms': (time.perf_counter() - started) * 1000,
        'resize_ms': 0,
        'face_ms': 0,
        'encode_ms': 0,
        'verify_ms': 0,
        'total_ms': (time.perf_counter() - started) * 1000,
        'output_bytes': temporary_path.stat().st_size,
    }


def process_job(upsampler, face_enhancer, job, lease_owner):
    if job.get('upload_type') == 'video':
        return process_video_job(upsampler, face_enhancer, job, lease_owner)
    started_clock = time.perf_counter()
    metrics = {'decode_ms': 0, 'sr_ms': 0, 'resize_ms': 0, 'face_ms': 0, 'encode_ms': 0, 'verify_ms': 0}
    input_path = INPUT_DIR / job['input_path']
    output_path = OUTPUT_DIR / job['output_path']
    if output_path.is_file() and valid_output(output_path, job):
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

    def report_tile_progress(_tile_index, total_tiles):
        nonlocal completed_tiles, last_progress
        completed_tiles += 1
        progress = int(5 + 80 * completed_tiles / (total_tiles * tile_passes))
        if progress > last_progress:
            update_progress(job['id'], lease_owner, progress)
            last_progress = progress

    update_progress(job['id'], lease_owner, 5)
    upsampler.progress_callback = report_tile_progress
    stage_started = time.perf_counter()
    try:
        output, _ = upsampler.enhance(image, outscale=4)
    finally:
        upsampler.progress_callback = None
    metrics['sr_ms'] = (time.perf_counter() - stage_started) * 1000

    expected_height, expected_width = expected_dimensions(job)
    stage_started = time.perf_counter()
    if output.shape[0:2] != (expected_height, expected_width):
        interpolation = cv2.INTER_AREA if output.shape[0] > expected_height else cv2.INTER_LANCZOS4
        output = cv2.resize(output, (expected_width, expected_height), interpolation=interpolation)
    metrics['resize_ms'] = (time.perf_counter() - stage_started) * 1000
    if job.get('face_enhance'):
        stage_started = time.perf_counter()
        update_progress(job['id'], lease_owner, 88)
        output = enhance_faces(output, face_enhancer)
        if output.shape[0:2] != (expected_height, expected_width):
            output = cv2.resize(output, (expected_width, expected_height), interpolation=cv2.INTER_LANCZOS4)
        metrics['face_ms'] = (time.perf_counter() - stage_started) * 1000

    suffix = 'jpg' if job.get('output_format') == 'jpeg' else job.get('output_format', 'png')
    temporary_path = TMP_DIR / f"{job['id']}.{lease_owner}.partial.{suffix}"
    update_progress(job['id'], lease_owner, 95)
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

        set_worker_state(slot, f"processing:{job['id']}")
        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=heartbeat_loop,
            args=(stop_event, job['id'], lease_owner),
            daemon=True,
        )
        heartbeat.start()
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
            temporary_path, metrics = process_job(upsampler, face_enhancer, job, lease_owner)
        except Exception as error:
            for partial in TMP_DIR.glob(f"{job['id']}.{lease_owner}.partial.*"):
                partial.unlink(missing_ok=True)
            finish_job(job['id'], lease_owner, 'failed', str(error)[:1000])
        else:
            publish_success(job, lease_owner, temporary_path, metrics, slot)
        finally:
            stop_event.set()
            heartbeat.join(timeout=2)
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
