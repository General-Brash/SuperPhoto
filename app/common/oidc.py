"""Self-contained OpenID Connect (Authorization Code + PKCE) helpers.

This module is intentionally decoupled from ``app.common.config``: every value it
needs is supplied through :class:`OIDCConfig` so it can be unit tested without any
application state or network access.

The deployment image pins PyJWT and cryptography. ID Tokens must pass RS256
JWKS signature and claim checks before any identity is accepted.
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
        document = json.loads(payload)
    except (ValueError, TypeError) as error:
        raise OIDCError('OIDC endpoint returned invalid JSON') from error
    if not isinstance(document, dict):
        raise OIDCError('OIDC endpoint returned invalid JSON object')
    return document


def validate_discovery(config, document):
    """Never trust an issuer asserted solely by provider-supplied metadata."""
    if not isinstance(document, dict):
        raise OIDCError('OIDC discovery returned an invalid document')
    if not config.issuer or document.get('issuer') != config.issuer:
        raise OIDCError('OIDC discovery issuer mismatch')
    for field in ('authorization_endpoint', 'token_endpoint', 'jwks_uri', 'userinfo_endpoint'):
        url = document.get(field)
        if not isinstance(url, str) or urllib.parse.urlparse(url).scheme != 'https':
            raise OIDCError('OIDC discovery missing secure ' + field)
    return document


def discover(config, *, timeout=5.0):
    """Fetch (and process-cache with a TTL) the OIDC discovery document."""
    url = discovery_url_for(config)
    if urllib.parse.urlparse(url).scheme != 'https':
        raise OIDCError('OIDC discovery must use HTTPS')
    now = time.monotonic()
    cached = _DISCOVERY_CACHE.get(url)
    if cached and cached[0] > now:
        return validate_discovery(config, cached[1])
    request = urllib.request.Request(url, method='GET')
    request.add_header('Accept', 'application/json')
    document = validate_discovery(config, _request_json(request, timeout))
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
    if not isinstance(claims, dict):
        raise OIDCError('invalid ID Token claims')
    if not config.issuer or issuer != config.issuer or claims.get('iss') != config.issuer:
        raise OIDCError('id_token issuer mismatch')
    audience = claims.get('aud')
    audiences = audience if isinstance(audience, list) else [audience]
    if config.client_id not in audiences:
        raise OIDCError('id_token audience mismatch')
    exp = claims.get('exp')
    if isinstance(exp, bool) or not isinstance(exp, (int, float)) or now > exp + 60:
        raise OIDCError('id_token expired')
    if claims.get('nonce') != nonce:
        raise OIDCError('id_token nonce mismatch')


def _load_jwt_library():
    try:
        import jwt
    except ImportError as error:
        raise OIDCError('OIDC signature verification is unavailable') from error
    return jwt


def verify_signature_if_possible(discovery, token, *, config, timeout=5.0):
    """Require RS256 JWKS verification; return authenticated claims or fail closed.

    The historical name is retained for callers, but verification is never optional.
    """
    jwks_uri = discovery.get('jwks_uri')
    if not isinstance(jwks_uri, str) or urllib.parse.urlparse(jwks_uri).scheme != 'https':
        raise OIDCError('OIDC signing keys unavailable')
    try:
        jwt = _load_jwt_library()
    except ImportError as error:
        raise OIDCError('OIDC signature verification is unavailable') from error
    try:
        client = jwt.PyJWKClient(jwks_uri, timeout=timeout)
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, signing_key.key, algorithms=['RS256'],
            audience=config.client_id, issuer=config.issuer,
            options={'require': ['iss', 'sub', 'aud', 'exp']}, leeway=60,
        )
        if not isinstance(claims, dict):
            raise OIDCError('invalid ID Token claims')
        return claims
    except OIDCError:
        raise
    except Exception as error:
        raise OIDCError('OIDC ID Token signature or claims invalid') from error


def fetch_userinfo(config, discovery, *, access_token, timeout=5.0):
    """GET the userinfo_endpoint with a Bearer token and return the claims dict."""
    endpoint = discovery.get('userinfo_endpoint')
    if not endpoint:
        raise OIDCError('discovery document missing userinfo_endpoint')
    request = urllib.request.Request(endpoint, method='GET')
    request.add_header('Authorization', 'Bearer {}'.format(access_token))
    request.add_header('Accept', 'application/json')
    return _request_json(request, timeout)
