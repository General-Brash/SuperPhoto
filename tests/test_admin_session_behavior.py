"""Regression coverage for W2 admin session invalidation semantics."""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def admin_app(tmp_path, monkeypatch):
    monkeypatch.setenv('OIDC_ENABLED', 'false')
    monkeypatch.setenv('SUPERPHOTO_ADMIN_USERNAME', 'admin_user')
    monkeypatch.setenv('SUPERPHOTO_ADMIN_PASSWORD', 'admin-password')
    monkeypatch.setenv('SUPERPHOTO_SESSION_SECRET', 'unit-test-secret-' * 3)
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
    client = TestClient(main.app)
    yield main, db, auth, client


def _login_admin(client):
    bootstrap = client.get('/api/auth/me').json()
    response = client.post(
        '/api/auth/login',
        json={'username': 'admin_user', 'password': 'admin-password'},
        headers={'X-CSRF-Token': bootstrap['csrf_token']},
    )
    assert response.status_code == 200
    return client.get('/api/auth/me').json()['csrf_token']


def test_quota_change_revokes_all_target_user_sessions(admin_app):
    _main, db, auth, client = admin_app
    with db.connect() as connection:
        target_id = 'target-user'
        now = db.utc_now()
        connection.execute(
            "INSERT INTO users(id, username, password_hash, role, daily_quota, active_quota, created_at, updated_at) "
            "VALUES (?, ?, ?, 'user', 30, 10, ?, ?)",
            (target_id, 'target_user', 'hash', now, now),
        )
        auth.create_session(connection, __import__('starlette.responses', fromlist=['Response']).Response(), target_id)
        auth.create_session(connection, __import__('starlette.responses', fromlist=['Response']).Response(), target_id)
        assert connection.execute('SELECT COUNT(*) FROM sessions WHERE user_id=?', (target_id,)).fetchone()[0] == 2

    csrf = _login_admin(client)
    response = client.patch(
        f'/api/admin/users/{target_id}',
        json={'daily_quota': 31},
        headers={'X-CSRF-Token': csrf},
    )
    assert response.status_code == 200
    with db.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM sessions WHERE user_id=?', (target_id,)).fetchone()[0] == 0


def test_admin_self_noop_update_preserves_current_session(admin_app):
    _main, db, _auth, client = admin_app
    csrf = _login_admin(client)
    before = client.get('/api/auth/me')
    assert before.status_code == 200 and before.json()['authenticated'] is True
    admin_id = before.json()['user']['id']
    response = client.patch(
        f'/api/admin/users/{admin_id}',
        json={},
        headers={'X-CSRF-Token': csrf},
    )
    assert response.status_code == 200
    after = client.get('/api/auth/me')
    assert after.status_code == 200 and after.json()['authenticated'] is True
    with db.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM sessions WHERE user_id=?', (admin_id,)).fetchone()[0] == 1
