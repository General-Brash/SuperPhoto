import base64
import json
import time
import urllib.error
import urllib.parse
from unittest import mock

import pytest

from app.common import oidc
from app.common.oidc import OIDCConfig, OIDCError


def _b64url(data):
    return base64.urlsafe_b64encode(json.dumps(data).encode('utf-8')).rstrip(b'=').decode('ascii')


def _make_jwt(header, claims, signature='sig'):
    return '{}.{}.{}'.format(_b64url(header), _b64url(claims), signature)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _json_response(document):
    return FakeResponse(json.dumps(document).encode('utf-8'))


def _config(**overrides):
    defaults = dict(
        enabled=True,
        issuer='https://issuer.example.com',
        discovery_url='',
        client_id='client-abc',
        client_secret='secret-xyz',
        redirect_uri='https://app.example.com/callback',
    )
    defaults.update(overrides)
    return OIDCConfig(**defaults)


@pytest.fixture(autouse=True)
def _reset_cache():
    oidc.clear_discovery_cache()
    yield
    oidc.clear_discovery_cache()


# --- PKCE -------------------------------------------------------------------

def test_generate_pkce_matches_rfc7636_vector():
    verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
    _, challenge = oidc.generate_pkce(verifier=verifier)
    assert challenge == 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM'


def test_generate_pkce_random_verifier_in_length_window():
    verifier, challenge = oidc.generate_pkce()
    assert 43 <= len(verifier) <= 128
    assert '=' not in challenge
    # challenge is a pure S256 transform of the verifier
    assert oidc.generate_pkce(verifier=verifier)[1] == challenge


# --- state / nonce ----------------------------------------------------------

def test_generate_state_and_nonce_are_random_and_distinct():
    assert oidc.generate_state() != oidc.generate_state()
    assert oidc.generate_nonce() != oidc.generate_nonce()
    assert len(oidc.generate_state()) >= 20
    assert len(oidc.generate_nonce()) >= 20


# --- authorization url ------------------------------------------------------

def test_build_authorization_url_contains_required_params():
    config = _config()
    discovery = {'authorization_endpoint': 'https://issuer.example.com/authorize'}
    url = oidc.build_authorization_url(
        config, discovery, state='state123', nonce='nonce456', code_challenge='chal789'
    )
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    assert parsed.path == '/authorize'
    assert query['response_type'] == 'code'
    assert query['client_id'] == 'client-abc'
    assert query['redirect_uri'] == 'https://app.example.com/callback'
    assert query['scope'] == 'openid profile'
    assert query['state'] == 'state123'
    assert query['nonce'] == 'nonce456'
    assert query['code_challenge'] == 'chal789'
    assert query['code_challenge_method'] == 'S256'


def test_build_authorization_url_missing_endpoint():
    with pytest.raises(OIDCError):
        oidc.build_authorization_url(_config(), {}, state='s', nonce='n', code_challenge='c')


# --- id_token validation ----------------------------------------------------

def _valid_claims():
    return {
        'iss': 'https://issuer.example.com',
        'aud': 'client-abc',
        'exp': time.time() + 300,
        'nonce': 'expected-nonce',
    }


def test_validate_id_token_accepts_valid_claims():
    oidc.validate_id_token(
        _config(), _valid_claims(), nonce='expected-nonce', issuer='https://issuer.example.com'
    )


def test_validate_id_token_accepts_audience_list():
    claims = _valid_claims()
    claims['aud'] = ['other', 'client-abc']
    oidc.validate_id_token(
        _config(), claims, nonce='expected-nonce', issuer='https://issuer.example.com'
    )


def test_validate_id_token_rejects_bad_issuer():
    claims = _valid_claims()
    claims['iss'] = 'https://evil.example.com'
    with pytest.raises(OIDCError, match='issuer'):
        oidc.validate_id_token(
            _config(), claims, nonce='expected-nonce', issuer='https://issuer.example.com'
        )


def test_validate_id_token_rejects_bad_audience():
    claims = _valid_claims()
    claims['aud'] = 'someone-else'
    with pytest.raises(OIDCError, match='audience'):
        oidc.validate_id_token(
            _config(), claims, nonce='expected-nonce', issuer='https://issuer.example.com'
        )


def test_validate_id_token_rejects_expired():
    claims = _valid_claims()
    claims['exp'] = time.time() - 3600
    with pytest.raises(OIDCError, match='expired'):
        oidc.validate_id_token(
            _config(), claims, nonce='expected-nonce', issuer='https://issuer.example.com'
        )


def test_validate_id_token_rejects_bad_nonce():
    with pytest.raises(OIDCError, match='nonce'):
        oidc.validate_id_token(
            _config(), _valid_claims(), nonce='different-nonce', issuer='https://issuer.example.com'
        )


# --- unverified decode ------------------------------------------------------

def test_decode_jwt_unverified_parses_segments():
    token = _make_jwt(
        {'alg': 'none', 'typ': 'JWT'},
        {'sub': 'user-42', 'preferred_username': 'alice'},
    )
    header, claims = oidc.decode_jwt_unverified(token)
    assert header['alg'] == 'none'
    assert claims['sub'] == 'user-42'
    assert claims['preferred_username'] == 'alice'


def test_decode_jwt_unverified_rejects_malformed():
    with pytest.raises(OIDCError):
        oidc.decode_jwt_unverified('not-a-jwt')


# --- discover ---------------------------------------------------------------

def test_discover_parses_and_caches():
    document = {
        'authorization_endpoint': 'https://issuer.example.com/authorize',
        'token_endpoint': 'https://issuer.example.com/token',
    }
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.return_value = _json_response(document)
        first = oidc.discover(_config())
        second = oidc.discover(_config())
    assert first == document
    assert second == document
    # second call is served from the process cache, no extra network hit
    assert urlopen.call_count == 1


def test_discover_derives_url_from_issuer():
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.return_value = _json_response({'token_endpoint': 'x'})
        oidc.discover(_config(discovery_url=''))
        request = urlopen.call_args[0][0]
    assert request.full_url == 'https://issuer.example.com/.well-known/openid-configuration'


def test_discover_wraps_network_error():
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.side_effect = urllib.error.URLError('boom')
        with pytest.raises(OIDCError):
            oidc.discover(_config())


# --- token exchange ---------------------------------------------------------

def test_exchange_code_uses_basic_auth_and_pkce():
    discovery = {'token_endpoint': 'https://issuer.example.com/token'}
    tokens = {'access_token': 'AT', 'id_token': 'IT', 'token_type': 'Bearer'}
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.return_value = _json_response(tokens)
        result = oidc.exchange_code(
            _config(), discovery, code='auth-code', code_verifier='verifier-xyz'
        )
        request = urlopen.call_args[0][0]

    assert result == tokens
    expected = base64.b64encode(b'client-abc:secret-xyz').decode('ascii')
    assert request.get_header('Authorization') == 'Basic {}'.format(expected)

    body = dict(urllib.parse.parse_qsl(request.data.decode('utf-8')))
    assert body['grant_type'] == 'authorization_code'
    assert body['code'] == 'auth-code'
    assert body['code_verifier'] == 'verifier-xyz'
    assert body['redirect_uri'] == 'https://app.example.com/callback'
    assert body['client_id'] == 'client-abc'


def test_exchange_code_wraps_http_error():
    discovery = {'token_endpoint': 'https://issuer.example.com/token'}
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.side_effect = urllib.error.HTTPError(
            'url', 400, 'Bad Request', {}, None
        )
        with pytest.raises(OIDCError):
            oidc.exchange_code(_config(), discovery, code='c', code_verifier='v')


# --- userinfo ---------------------------------------------------------------

def test_fetch_userinfo_sends_bearer_and_parses():
    discovery = {'userinfo_endpoint': 'https://issuer.example.com/userinfo'}
    claims = {'sub': 'user-42', 'email': 'alice@example.com', 'role': 'user'}
    with mock.patch('app.common.oidc.urllib.request.urlopen') as urlopen:
        urlopen.return_value = _json_response(claims)
        result = oidc.fetch_userinfo(_config(), discovery, access_token='AT')
        request = urlopen.call_args[0][0]
    assert result == claims
    assert request.get_header('Authorization') == 'Bearer AT'


def test_fetch_userinfo_missing_endpoint():
    with pytest.raises(OIDCError):
        oidc.fetch_userinfo(_config(), {}, access_token='AT')


# --- soft-optional signature verification -----------------------------------

def test_verify_signature_returns_none_without_library():
    with mock.patch('app.common.oidc._load_jwt_library', return_value=None):
        result = oidc.verify_signature_if_possible({'jwks_uri': 'https://x/jwks'}, 'a.b.c')
    assert result is None


def test_verify_signature_false_when_jwks_uri_missing():
    fake_jwt = mock.Mock()
    with mock.patch('app.common.oidc._load_jwt_library', return_value=fake_jwt):
        result = oidc.verify_signature_if_possible({}, 'a.b.c')
    assert result is False


def test_verify_signature_true_when_library_verifies():
    fake_jwt = mock.Mock()
    fake_client = mock.Mock()
    fake_client.get_signing_key_from_jwt.return_value = mock.Mock(key='PUBKEY')
    fake_jwt.PyJWKClient.return_value = fake_client
    fake_jwt.decode.return_value = {'sub': 'ok'}
    with mock.patch('app.common.oidc._load_jwt_library', return_value=fake_jwt):
        result = oidc.verify_signature_if_possible({'jwks_uri': 'https://x/jwks'}, 'a.b.c')
    assert result is True


def test_verify_signature_false_when_library_raises():
    fake_jwt = mock.Mock()
    fake_jwt.PyJWKClient.side_effect = ValueError('bad key')
    with mock.patch('app.common.oidc._load_jwt_library', return_value=fake_jwt):
        result = oidc.verify_signature_if_possible({'jwks_uri': 'https://x/jwks'}, 'a.b.c')
    assert result is False
