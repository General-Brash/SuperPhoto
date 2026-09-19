import os
from pathlib import Path


STATE_DIR = Path(os.getenv('REALESRGAN_STATE_DIR', '/state'))
INPUT_DIR = Path(os.getenv('REALESRGAN_INPUT_DIR', '/data/input'))
OUTPUT_DIR = Path(os.getenv('REALESRGAN_OUTPUT_DIR', '/data/output'))
TMP_DIR = Path(os.getenv('REALESRGAN_TMP_DIR', '/data/tmp'))
DB_PATH = STATE_DIR / 'jobs.db'
STATIC_DIR = Path(os.getenv('SUPERPHOTO_STATIC_DIR', '/opt/realesrgan/app/web'))
MODEL_DIR = Path(os.getenv('REALESRGAN_MODEL_DIR', '/models'))
MODEL_PATH = Path(os.getenv('REALESRGAN_MODEL_PATH', str(MODEL_DIR / 'RealESRGAN_x4plus.xml')))
ANIME_MODEL_PATH = Path(
    os.getenv('REALESRGAN_ANIME_MODEL_PATH', str(MODEL_DIR / 'RealESRGAN_x4plus_anime_6B.xml'))
)
GFPGAN_MODEL_PATH = Path(os.getenv('GFPGAN_MODEL_PATH', str(MODEL_DIR / 'GFPGANv1.3.pth')))
FACE_DETECTION_MODEL_PATH = MODEL_DIR / 'gfpgan/weights/detection_Resnet50_Final.pth'
FACE_PARSING_MODEL_PATH = MODEL_DIR / 'gfpgan/weights/parsing_parsenet.pth'

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 10
MAX_SIDE = 4096
MAX_OUTPUT_PIXELS = 64_000_000
MAX_USER_JOBS = 20
MAX_GLOBAL_JOBS = 50
GUEST_DAILY_QUOTA = 3
GUEST_ACTIVE_QUOTA = 2
DEFAULT_IMAGE_QUOTAS = {'2k': 30, '4k': 20, '6k': 10, '8k': 0}

SESSION_COOKIE = 'superphoto_session'
SESSION_TTL_HOURS = 24
USER_FILE_TTL_DAYS = 7
GUEST_FILE_TTL_HOURS = 24
SHARE_TTL_HOURS = 24
MIN_FREE_DISK_BYTES = 5 * 1024 * 1024 * 1024

SESSION_SECRET = os.getenv('SUPERPHOTO_SESSION_SECRET', '')
ADMIN_USERNAME = os.getenv('SUPERPHOTO_ADMIN_USERNAME', '')
ADMIN_PASSWORD = os.getenv('SUPERPHOTO_ADMIN_PASSWORD', '')
TURNSTILE_SITE_KEY = os.getenv('TURNSTILE_SITE_KEY', '')
TURNSTILE_SECRET_KEY = os.getenv('TURNSTILE_SECRET_KEY', '')
TURNSTILE_REQUIRED = os.getenv('TURNSTILE_REQUIRED', 'false').lower() in ('1', 'true', 'yes')
COOKIE_SECURE = os.getenv('SUPERPHOTO_COOKIE_SECURE', 'true').lower() in ('1', 'true', 'yes')

# OIDC / OAuth2：SuperPhoto 作为被认证方(RP)，认证方为 Personal_Sub2。
OIDC_ISSUER = os.getenv('OIDC_ISSUER', '')
OIDC_DISCOVERY_URL = os.getenv('OIDC_DISCOVERY_URL', '')
OIDC_CLIENT_ID = os.getenv('OIDC_CLIENT_ID', '')
OIDC_CLIENT_SECRET = os.getenv('OIDC_CLIENT_SECRET', '')
OIDC_REDIRECT_URI = os.getenv('OIDC_REDIRECT_URI', '')
OIDC_SCOPES = os.getenv('OIDC_SCOPES', 'openid profile')
# 缺失任一关键项时，即便 OIDC_ENABLED=true 也视为关闭（fail-closed）。
OIDC_ENABLED = (
    os.getenv('OIDC_ENABLED', 'false').lower() in ('1', 'true', 'yes')
    and bool(OIDC_CLIENT_ID and OIDC_CLIENT_SECRET and OIDC_REDIRECT_URI and OIDC_ISSUER)
)

ROLE_GUEST = 'guest'
ROLE_USER = 'user'
ROLE_ADVANCED = 'advanced'
ROLE_ADMIN = 'admin'
VALID_ROLES = {ROLE_USER, ROLE_ADVANCED, ROLE_ADMIN}

MODEL_REGISTRY = {
    'general': {
        'label': '通用照片',
        'path': MODEL_PATH,
        'architecture': 'rrdb',
    },
    'anime': {
        'label': '动漫插画',
        'path': ANIME_MODEL_PATH,
        'architecture': 'rrdb6',
    },
}

SAFE_TILES = {128, 256, 512}
ADMIN_TILES = {64, 128, 192, 256, 384, 512, 768, 1024}
OUTPUT_FORMATS = {'png', 'jpeg', 'webp'}
QUALITY_PRESETS = {'standard', 'high', 'maximum'}
TARGET_RESOLUTIONS = {'2k', '4k', '6k', '8k'}
ASPECT_RATIOS = {'original', '16:9', '9:16', '4:3', '3:4', '1:1'}


def face_models_available():
    minimum_sizes = {
        GFPGAN_MODEL_PATH: 300 * 1024 * 1024,
        FACE_DETECTION_MODEL_PATH: 90 * 1024 * 1024,
        FACE_PARSING_MODEL_PATH: 70 * 1024 * 1024,
    }
    return all(path.is_file() and path.stat().st_size >= size for path, size in minimum_sizes.items())


def ensure_directories():
    for path in (STATE_DIR, INPUT_DIR, OUTPUT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
