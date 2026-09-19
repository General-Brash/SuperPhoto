import math
from statistics import median
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from .config import (
    ADMIN_TILES,
    ASPECT_RATIOS,
    GUEST_FILE_TTL_HOURS,
    OUTPUT_FORMATS,
    QUALITY_PRESETS,
    ROLE_ADMIN,
    ROLE_ADVANCED,
    SAFE_TILES,
    TARGET_RESOLUTIONS,
    USER_FILE_TTL_DAYS,
)


RATIO_VALUES = {
    '16:9': 16 / 9,
    '9:16': 9 / 16,
    '4:3': 4 / 3,
    '3:4': 3 / 4,
    '1:1': 1,
}

TARGETS = {
    ('16:9', '2k'): (2560, 1440),
    ('16:9', '4k'): (3840, 2160),
    ('9:16', '2k'): (1440, 2560),
    ('9:16', '4k'): (2160, 3840),
    ('4:3', '2k'): (2048, 1536),
    ('4:3', '4k'): (3840, 2880),
    ('3:4', '2k'): (1536, 2048),
    ('3:4', '4k'): (2880, 3840),
    ('1:1', '2k'): (2048, 2048),
    ('1:1', '4k'): (3840, 3840),
    ('16:9', '6k'): (5760, 3240),
    ('16:9', '8k'): (7680, 4320),
    ('9:16', '6k'): (3240, 5760),
    ('9:16', '8k'): (4320, 7680),
    ('4:3', '6k'): (5760, 4320),
    ('4:3', '8k'): (7680, 5760),
    ('3:4', '6k'): (4320, 5760),
    ('3:4', '8k'): (5760, 7680),
    ('1:1', '6k'): (5760, 5760),
    ('1:1', '8k'): (7680, 7680),
}

QUALITY_VALUES = {
    'jpeg': {'standard': 86, 'high': 93, 'maximum': 98},
    'webp': {'standard': 82, 'high': 92, 'maximum': 100},
}

COMPRESSION_LEVELS = {1, 3, 5, 7, 9}
DEFAULT_SECONDS_PER_MEGAPIXEL = 12.0
MAX_CALIBRATION_SAMPLES = 200


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def validate_settings(raw, role, has_alpha=False):
    raw = raw or {}
    model = raw.get('model', 'general')
    target = raw.get('target_resolution', '4k')
    aspect = raw.get('aspect_ratio', 'original')
    crop = parse_bool(raw.get('crop_enabled', False))
    face = parse_bool(raw.get('face_enhance', False))
    output_format = raw.get('output_format', 'png' if has_alpha else 'jpeg')
    quality = raw.get('quality_preset', 'high')
    compression = int(raw.get('compression_level', 5))
    tile = int(raw.get('tile_size', 256))

    if model not in ('general', 'anime'):
        raise HTTPException(400, 'Unsupported model')
    if aspect not in ASPECT_RATIOS:
        raise HTTPException(400, 'Unsupported target resolution or aspect ratio')
    if target not in TARGET_RESOLUTIONS:
        raise HTTPException(400, 'Unsupported target resolution or aspect ratio')
    allowed_targets = {
        'guest': {'2k', '4k'},
        'user': {'2k', '4k', '6k'},
        ROLE_ADVANCED: {'2k', '4k', '6k', '8k'},
        ROLE_ADMIN: {'2k', '4k', '6k', '8k'},
    }.get(role, {'2k', '4k'})
    if target not in allowed_targets:
        raise HTTPException(403, 'Target resolution is not allowed for this account')
    if output_format not in OUTPUT_FORMATS or quality not in QUALITY_PRESETS:
        raise HTTPException(400, 'Unsupported output format or quality')
    if compression not in COMPRESSION_LEVELS:
        raise HTTPException(400, 'Compression level must be one of 1, 3, 5, 7, or 9')
    if has_alpha and output_format == 'jpeg':
        raise HTTPException(400, 'JPEG cannot preserve image transparency')
    if crop and aspect == 'original':
        raise HTTPException(400, 'Cropping requires a target aspect ratio')
    if role in (ROLE_ADVANCED, ROLE_ADMIN):
        allowed_tiles = ADMIN_TILES if role == ROLE_ADMIN else SAFE_TILES
    else:
        allowed_tiles = {256}
    if tile not in allowed_tiles:
        raise HTTPException(403, 'Tile size is not allowed for this account')
    return {
        'model_name': model,
        'target_resolution': target,
        'aspect_ratio': aspect,
        'crop_enabled': int(crop),
        'face_enhance': int(face),
        'output_format': output_format,
        'quality_preset': quality,
        'compression_level': compression,
        'tile_size': tile,
    }


def fit_processing_model(observations):
    samples = []
    for observation in observations:
        if isinstance(observation, dict):
            mp = observation.get('input_mp') or observation.get('width', 0) * observation.get('height', 0) / 1_000_000
            seconds = observation.get('total_seconds') or observation.get('seconds')
        else:
            mp, seconds = observation[:2]
        if mp and seconds and float(mp) > 0 and float(seconds) > 0:
            samples.append((float(mp), float(seconds)))
    if len(samples) < 3:
        return 0.0, DEFAULT_SECONDS_PER_MEGAPIXEL
    mean_x = sum(item[0] for item in samples) / len(samples)
    mean_y = sum(item[1] for item in samples) / len(samples)
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    if denominator < 1e-9:
        return 0.0, median(y / x for x, y in samples)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / denominator
    intercept = mean_y - slope * mean_x
    return max(0.0, intercept), max(0.1, slope)


def _median_rate(observations, value_key, denominator_key):
    rates = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        value = float(item.get(value_key) or 0)
        denominator = float(item.get(denominator_key) or 0)
        if value > 0 and denominator > 0:
            rates.append(value / denominator)
    return median(rates) if rates else None


def _stage_estimate(width, height, settings, observations):
    input_mp = width * height / 1_000_000
    output_width = int(settings.get('output_width') or width * 4)
    output_height = int(settings.get('output_height') or height * 4)
    output_mp = output_width * output_height / 1_000_000
    tile = max(64, int(settings.get('tile_size') or 256))
    tile_count = max(1, math.ceil(width / tile) * math.ceil(height / tile))
    tile_work_mp = input_mp * (1 + 2 * 10 / tile) ** 2
    alpha_passes = 2 if settings.get('has_alpha') else 1

    sr_rate = _median_rate(observations, 'sr_seconds', 'work_mp') or 0
    if not sr_rate:
        _, sr_rate = fit_processing_model(observations)
        sr_rate /= 16
    sr_seconds = (sr_rate * tile_work_mp * 16 + 0.03 * tile_count) * alpha_passes

    resize_rate = _median_rate(observations, 'resize_seconds', 'output_mp') or 0.015
    resize_seconds = resize_rate * max(output_mp, input_mp * 16)

    face_seconds = 0
    if settings.get('face_enhance'):
        face_rate = _median_rate(observations, 'face_seconds', 'output_mp') or 0.04
        face_seconds = face_rate * output_mp

    output_format = settings.get('output_format', 'png')
    quality = settings.get('quality_preset', 'high')
    compression = int(settings.get('compression_level') or 5)
    encode_rates = [
        float(item.get('encode_seconds') or 0) / float(item.get('output_mp') or 1)
        for item in observations if isinstance(item, dict)
        and item.get('output_format') == output_format
        and item.get('quality_preset') == quality
        and (output_format != 'png' or int(item.get('compression_level') or 5) == compression)
        and float(item.get('encode_seconds') or 0) > 0 and float(item.get('output_mp') or 0) > 0
    ]
    if encode_rates:
        encode_rate = median(encode_rates)
    elif output_format == 'png':
        encode_rate = 0.018 + 0.006 * compression
    elif output_format == 'webp':
        encode_rate = {'standard': 0.012, 'high': 0.018, 'maximum': 0.028}.get(quality, 0.018)
    else:
        encode_rate = {'standard': 0.008, 'high': 0.012, 'maximum': 0.018}.get(quality, 0.012)
    encode_seconds = encode_rate * output_mp

    fixed = median([
        float(item.get('fixed_seconds') or 0) for item in observations
        if isinstance(item, dict) and float(item.get('fixed_seconds') or 0) > 0
    ]) if observations else 0
    return max(1, round(fixed + sr_seconds + resize_seconds + face_seconds + encode_seconds))


def estimate_processing_seconds(width, height, face_enhance=False, seconds_per_megapixel=None, observations=None, settings=None):
    if seconds_per_megapixel:
        intercept, rate = 0.0, seconds_per_megapixel
    else:
        observations = observations or []
        if settings and any(isinstance(item, dict) and item.get('sr_seconds') for item in observations):
            return _stage_estimate(width, height, {**settings, 'face_enhance': face_enhance}, observations)
        intercept, rate = fit_processing_model(observations)
    multiplier = 1.35 if face_enhance else 1.0
    return max(1, round((intercept + (width * height / 1_000_000) * rate) * multiplier))


def estimate_output_bytes(width, height, output_format, quality='high', compression_level=5, observations=None):
    pixels = max(1, int(width) * int(height))
    matching = [float(size) / max(1, int(w) * int(h)) for w, h, fmt, quality_value, level, size in (observations or [])
                if fmt == output_format and quality_value == quality and int(level or 5) == int(compression_level or 5) and size > 0]
    if matching:
        bytes_per_pixel = max(0.03, min(8.0, median(matching)))
    elif output_format == 'png':
        bytes_per_pixel = 0.55 - 0.035 * int(compression_level or 5)
    elif output_format == 'webp':
        bytes_per_pixel = {'standard': 0.28, 'high': 0.38, 'maximum': 0.52}.get(quality, 0.38)
    else:
        bytes_per_pixel = {'standard': 0.32, 'high': 0.45, 'maximum': 0.62}.get(quality, 0.45)
    return max(1024, round(pixels * max(0.03, bytes_per_pixel)))


def detect_aspect(width, height, tolerance=0.001):
    ratio = width / height
    matches = [(name, abs(ratio - value) / value) for name, value in RATIO_VALUES.items()]
    name, difference = min(matches, key=lambda item: item[1])
    return name if difference <= tolerance else 'original'


def target_dimensions(width, height, target, aspect='original', crop=False):
    effective_aspect = aspect if crop and aspect != 'original' else detect_aspect(width, height)
    if effective_aspect in RATIO_VALUES:
        return TARGETS[(effective_aspect, target)]
    long_side = {'2k': 2560, '4k': 3840, '6k': 5760, '8k': 7680}[target]
    if width >= height:
        return long_side, max(1, round(height * long_side / width))
    return max(1, round(width * long_side / height)), long_side


def crop_box(width, height, aspect):
    target_ratio = RATIO_VALUES[aspect]
    current = width / height
    if current > target_ratio:
        new_width = round(height * target_ratio)
        left = (width - new_width) // 2
        return left, 0, left + new_width, height
    new_height = round(width / target_ratio)
    top = (height - new_height) // 2
    return 0, top, width, top + new_height


def expiry_iso(user_id):
    now = datetime.now(timezone.utc)
    if user_id:
        return (now + timedelta(days=USER_FILE_TTL_DAYS)).isoformat()
    return (now + timedelta(hours=GUEST_FILE_TTL_HOURS)).isoformat()


def output_extension(output_format):
    return 'jpg' if output_format == 'jpeg' else output_format
