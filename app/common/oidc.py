"""Self-contained OpenID Connect (Authorization Code + PKCE) helpers.

This module is intentionally decoupled from ``app.common.config``: every value it
needs is supplied through :class:`OIDCConfig` so it can be unit tested without any
application state or network access.

Runtime constraint: the deployment image (requirements.lock) ships neither
``cryptography`` nor ``PyJWT``/``python-jose``. Only the standard library and
``requests`` are guaranteed. Therefore all HTTP calls use ``urllib.request`` and
id_token signature verification is treated as *soft-optional*
(see :func:`verify_signature_if_possible`).
"""

import base64
import hashlib
import json
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass


class OIDCError(Exception):
    """Raised for any recoverable OIDC failure (network, parsing, validation)."""


@dataclass
class OIDCConfig:
    enabled: bool = False
    issuer: str = ''
    discovery_url: str = ''
    client_id: str = ''
    client_secret: str = ''
    redirect_uri: str = ''
    scopes: str = 'openid profile'


DISCOVERY_TTL_SECONDS = 3600
_DISCOVERY_CACHE = {}


def clear_discovery_cache():
    """Drop every cached discovery document (used by tests and config reloads)."""
    _DISCOVERY_CACHE.clear()


def discovery_url_for(config):
    """Return the explicit discovery_url or derive it from the issuer."""
    if config.discovery_url:
        return config.discovery_url
    return config.issuer.rstrip('/') + '/.well-known/openid-configuration'


def _b64url_decode(segment):
    padding = '=' * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _b64url_nopad(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def _request_json(request, timeout):
    """Perform a urllib request and parse the JSON body, wrapping errors as OIDCError."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (OSError, ValueError) as error:
        raise OIDCError('OIDC request failed: {}'.format(error)) from error
    try:
        return json.loads(payload)
    except (ValueError, TypeError) as error:
        raise OIDCError('OIDC endpoint returned invalid JSON') from error


def discover(config, *, timeout=5.0):
    """Fetch (and process-cache with a TTL) the OIDC discovery document."""
    url = discovery_url_for(config)
    now = time.monotonic()
    cached = _DISCOVERY_CACHE.get(url)
    if cached and cached[0] > now:
        return cached[1]
    request = urllib.request.Request(url, method='GET')
    request.add_header('Accept', 'application/json')
    document = _request_json(request, timeout)
    _DISCOVERY_CACHE[url] = (now + DISCOVERY_TTL_SECONDS, document)
    return document


def _code_challenge(verifier):
    """S256 transform: base64url(sha256(verifier)) without '=' padding."""
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    return _b64url_nopad(digest)


def generate_pkce(verifier=None):
    """Return a (code_verifier, code_challenge) pair using the S256 method.

    ``secrets.token_urlsafe(32)`` yields a 43-character URL-safe string, which sits
    within the RFC 7636 length window (43-128) and uses only allowed characters.
    A verifier may be supplied to make the transform deterministic (tests).
    """
    if verifier is None:
        verifier = secrets.token_urlsafe(32)
    return verifier, _code_challenge(verifier)


def generate_state():
    return secrets.token_urlsafe(32)


def generate_nonce():
    return secrets.token_urlsafe(32)


def build_authorization_url(config, discovery, *, state, nonce, code_challenge):
    """Build the authorization_endpoint redirect URL for the code flow."""
    endpoint = discovery.get('authorization_endpoint')
    if not endpoint:
        raise OIDCError('discovery document missing authorization_endpoint')
    query = urllib.parse.urlencode({
        'response_type': 'code',
        'client_id': config.client_id,
        'redirect_uri': config.redirect_uri,
        'scope': config.scopes,
        'state': state,
        'nonce': nonce,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
    })
    separator = '&' if '?' in endpoint else '?'
    return '{}{}{}'.format(endpoint, separator, query)


def exchange_code(config, discovery, *, code, code_verifier, timeout=5.0):
    """Exchange an authorization code for tokens at the token_endpoint.

    Client authentication uses HTTP Basic (client_secret_basic).
    """
    endpoint = discovery.get('token_endpoint')
    if not endpoint:
        raise OIDCError('discovery document missing token_endpoint')
    body = urllib.parse.urlencode({
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': config.redirect_uri,
        'code_verifier': code_verifier,
        'client_id': config.client_id,
    }).encode('utf-8')
    raw_credentials = '{}:{}'.format(config.client_id, config.client_secret).encode('utf-8')
    credentials = base64.b64encode(raw_credentials).decode('ascii')
    request = urllib.request.Request(endpoint, data=body, method='POST')
    request.add_header('Authorization', 'Basic {}'.format(credentials))
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    request.add_header('Accept', 'application/json')
    return _request_json(request, timeout)


def decode_jwt_unverified(token):
    """Base64url-decode the three JWT segments WITHOUT verifying the signature.

    Returns ``(header, claims)``. Parsing failures raise OIDCError.
    """
    try:
        header_b64, claims_b64, _signature = token.split('.')
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(claims_b64))
    except (ValueError, TypeError) as error:
        raise OIDCError('malformed JWT: {}'.format(error)) from error
    return header, claims


def validate_id_token(config, claims, *, nonce, issuer, now=None):
    """Validate id_token claims: iss, aud, exp (60s skew) and nonce.

    Raises OIDCError on any mismatch. ``now`` may be supplied for deterministic tests.
    """
    now = time.time() if now is None else now
    if claims.get('iss') != issuer:
        raise OIDCError('id_token issuer mismatch')
    audience = claims.get('aud')
    audiences = audience if isinstance(audience, list) else [audience]
    if config.client_id not in audiences:
        raise OIDCError('id_token audience mismatch')
    exp = claims.get('exp')
    if not isinstance(exp, (int, float)) or now > exp + 60:
        raise OIDCError('id_token expired')
    if claims.get('nonce') != nonce:
        raise OIDCError('id_token nonce mismatch')


def _load_jwt_library():
    """Return the PyJWT module if importable, otherwise None (kept small for tests)."""
    try:
        import jwt
    except ImportError:
        return None
    return jwt


def verify_signature_if_possible(discovery, token, *, timeout=5.0):
    """Best-effort RS256 signature check against the discovery ``jwks_uri``.

    Returns ``True``/``False`` when a crypto library (PyJWT) is available and the
    verification runs, or ``None`` when no such library is installed, signalling that
    local signature verification was skipped.

    Security note: in the Authorization Code flow the id_token is obtained directly
    from the token endpoint over a server-to-server TLS connection. OIDC Core
    section 3.1.3.7 permits relying on that TLS-protected direct fetch in place of
    local signature verification. When PyJWT/cryptography are absent (as in the
    pinned runtime image), trust therefore rests on TLS + the direct token exchange,
    complemented by the userinfo endpoint call, rather than on a local JWS check.
    """
    jwt = _load_jwt_library()
    if jwt is None:
        return None
    jwks_uri = discovery.get('jwks_uri')
    if not jwks_uri:
        return False
    try:
        client = jwt.PyJWKClient(jwks_uri, timeout=timeout)
        signing_key = client.get_signing_key_from_jwt(token)
        jwt.decode(
            token,
            signing_key.key,
            algorithms=['RS256'],
            options={'verify_aud': False, 'verify_exp': False},
        )
        return True
    except Exception:
        return False


def fetch_userinfo(config, discovery, *, access_token, timeout=5.0):
    """GET the userinfo_endpoint with a Bearer token and return the claims dict."""
    endpoint = discovery.get('userinfo_endpoint')
    if not endpoint:
        raise OIDCError('discovery document missing userinfo_endpoint')
    request = urllib.request.Request(endpoint, method='GET')
    request.add_header('Authorization', 'Bearer {}'.format(access_token))
    request.add_header('Accept', 'application/json')
    return _request_json(request, timeout)
