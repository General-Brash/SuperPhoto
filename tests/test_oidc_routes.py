"""End-to-end tests for the OIDC login/callback routes (W3 acceptance).

These exercise the FastAPI routes with the OIDC provider mocked, so no network
access or crypto library is required. Environment variables are set *before*
importing the app so the config-time OIDC_ENABLED flag is evaluated as true.
"""

import importlib
import os
import urllib.parse

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def oidc_client(tmp_path, monkeypatch):
    # OIDC config is resolved at import time; set env then reload config + main.
    monkeypatch.setenv('OIDC_ENABLED', 'true')
    monkeypatch.setenv('OIDC_ISSUER', 'https://auth.taffy.edu.kg')
    monkeypatch.setenv('OIDC_CLIENT_ID', 'superphoto')
    monkeypatch.setenv('OIDC_CLIENT_SECRET', 'topsecret')
    monkeypatch.setenv('OIDC_REDIRECT_URI', 'https://app.example/api/auth/oidc/callback')
    monkeypatch.setenv('SUPERPHOTO_SESSION_SECRET', 'unit-test-secret')
    monkeypatch.setenv('SUPERPHOTO_COOKIE_SECURE', 'false')
    monkeypatch.setenv('REALESRGAN_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('SUPERPHOTO_STATIC_DIR', str(tmp_path))

    import app.common.config as config
    importlib.reload(config)
    import app.common.db as db
    importlib.reload(db)
    import app.common.auth as auth
    importlib.reload(auth)
    import app.api.main as main
    importlib.reload(main)

    db.DB_PATH = tmp_path / 'jobs.db'
    main.init_db()

    discovery = {
        'issuer': 'https://auth.taffy.edu.kg',
        'authorization_endpoint': 'https://auth.taffy.edu.kg/oauth/authorize',
        'token_endpoint': 'https://auth.taffy.edu.kg/oauth/token',
        'userinfo_endpoint': 'https://auth.taffy.edu.kg/oauth/userinfo',
        'jwks_uri': 'https://auth.taffy.edu.kg/oauth/jwks',
    }
    monkeypatch.setattr(main.oidc_lib, 'discover', lambda config, **kw: discovery)
    yield main, TestClient(main.app, follow_redirects=False)


def test_auth_me_exposes_oidc_enabled(oidc_client):
    main, client = oidc_client
    body = client.get('/api/auth/me').json()
    assert body['oidc_enabled'] is True


def test_login_builds_authorization_url_and_sets_flow_cookie(oidc_client):
    main, client = oidc_client
    resp = client.get('/api/auth/oidc/login')
    assert resp.status_code == 303
    location = resp.headers['location']
    query = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
    assert query['response_type'] == ['code']
    assert query['code_challenge_method'] == ['S256']
    assert query['client_id'] == ['superphoto']
    assert 'code_challenge' in query and 'state' in query and 'nonce' in query
    assert main.OIDC_FLOW_COOKIE in resp.cookies


def test_callback_creates_user_and_session(oidc_client, monkeypatch):
    main, client = oidc_client
    # Drive login first so the signed flow cookie (state/nonce/verifier) is set.
    login = client.get('/api/auth/oidc/login')
    state = urllib.parse.parse_qs(urllib.parse.urlparse(login.headers['location']).query)['state'][0]

    monkeypatch.setattr(main.oidc_lib, 'exchange_code',
                        lambda *a, **k: {'id_token': 'x.y.z', 'access_token': 'at'})
    claims = {
        'iss': 'https://auth.taffy.edu.kg', 'aud': 'superphoto',
        'sub': 'sub-123', 'exp': 4102444800, 'nonce': 'flow-nonce',
    }
    monkeypatch.setattr(main.oidc_lib, 'verify_signature_if_possible', lambda *a, **k: claims)
    monkeypatch.setattr(main.oidc_lib, 'validate_id_token', lambda *a, **k: None)
    monkeypatch.setattr(main.oidc_lib, 'fetch_userinfo',
                        lambda *a, **k: {'sub': 'sub-123', 'preferred_username': 'alice'})

    resp = client.get(f'/api/auth/oidc/callback?code=abc&state={state}')
    assert resp.status_code == 303
    assert resp.headers['location'] == '/'
    assert 'superphoto_session' in resp.cookies

    # A user + identity row must now exist (query through the app's own connection).
    from app.common.db import connect
    with connect() as conn:
        identity = conn.execute('SELECT * FROM oidc_identities WHERE sub=?', ('sub-123',)).fetchone()
        assert identity is not None
        user = conn.execute('SELECT * FROM users WHERE id=?', (identity['user_id'],)).fetchone()
        assert user['password_hash'] == 'oidc$disabled'
        assert user['role'] == 'user'


def test_callback_rejects_bad_state(oidc_client):
    main, client = oidc_client
    client.get('/api/auth/oidc/login')  # sets flow cookie with a real state
    resp = client.get('/api/auth/oidc/callback?code=abc&state=wrong-state')
    assert resp.status_code == 303
    assert 'oidc_error' in resp.headers['location']


def test_callback_rejects_userinfo_subject_mismatch(oidc_client, monkeypatch):
    main, client = oidc_client
    login = client.get('/api/auth/oidc/login')
    state = urllib.parse.parse_qs(urllib.parse.urlparse(login.headers['location']).query)['state'][0]
    monkeypatch.setattr(main.oidc_lib, 'exchange_code', lambda *a, **k: {'id_token': 'signed', 'access_token': 'at'})
    monkeypatch.setattr(main.oidc_lib, 'verify_signature_if_possible', lambda *a, **k: {'sub': 'id-sub'})
    monkeypatch.setattr(main.oidc_lib, 'validate_id_token', lambda *a, **k: None)
    monkeypatch.setattr(main.oidc_lib, 'fetch_userinfo', lambda *a, **k: {'sub': 'other-sub'})
    resp = client.get(f'/api/auth/oidc/callback?code=abc&state={state}')
    assert resp.status_code == 303
    assert 'oidc_error' in resp.headers['location']


def test_oidc_unique_username_handles_blank_and_collision(oidc_client):
    main, _client = oidc_client
    from app.common.db import connect
    with connect() as connection:
        first = main._oidc_unique_username(connection, {'preferred_username': '!!!'})
        assert first.startswith('sub2_')
        connection.execute(
            "INSERT INTO users(id, username, password_hash, role, daily_quota, active_quota, created_at, updated_at) "
            "VALUES (?, ?, ?, 'user', 30, 10, ?, ?)",
            ('collision', 'alice', 'hash', '2026-01-01', '2026-01-01'),
        )
        second = main._oidc_unique_username(connection, {'preferred_username': 'alice'})
        assert second != 'alice'
        assert len(second) <= 24


def test_oidc_routes_are_disabled_when_flag_is_false(oidc_client, monkeypatch):
    main, client = oidc_client
    monkeypatch.setattr(main, 'OIDC_ENABLED', False)
    assert client.get('/api/auth/oidc/login').status_code == 404
    assert client.get('/api/auth/oidc/callback').status_code == 404
