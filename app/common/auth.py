import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, Response

from .config import COOKIE_SECURE, ROLE_ADMIN, SESSION_COOKIE, SESSION_SECRET, SESSION_TTL_HOURS
from .db import utc_now
from .security import random_token, token_hash


def future_iso(hours):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def signed_token(token):
    signature = hmac.new(SESSION_SECRET.encode('utf-8'), token.encode('utf-8'), hashlib.sha256).hexdigest()
    return f'{token}.{signature}'


def unsigned_token(cookie_value):
    try:
        token, signature = cookie_value.rsplit('.', 1)
    except (AttributeError, ValueError):
        return None
    expected = hmac.new(SESSION_SECRET.encode('utf-8'), token.encode('utf-8'), hashlib.sha256).hexdigest()
    return token if hmac.compare_digest(signature, expected) else None


def set_session_cookie(response, token):
    response.set_cookie(
        SESSION_COOKIE,
        signed_token(token),
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite='lax',
        path='/',
    )


def create_session(connection, response, user_id=None):
    token = random_token()
    session = {
        'id': secrets.token_hex(16),
        'token_hash': token_hash(token),
        'csrf_token': random_token(24),
        'user_id': user_id,
        'created_at': utc_now(),
        'expires_at': future_iso(SESSION_TTL_HOURS),
        'last_seen_at': utc_now(),
    }
    connection.execute(
        '''INSERT INTO sessions(id, token_hash, csrf_token, user_id, created_at, expires_at, last_seen_at)
           VALUES (:id, :token_hash, :csrf_token, :user_id, :created_at, :expires_at, :last_seen_at)''',
        session,
    )
    set_session_cookie(response, token)
    return session


def get_session(connection, request, response, create=True):
    token = unsigned_token(request.cookies.get(SESSION_COOKIE))
    row = None
    if token:
        row = connection.execute(
            '''SELECT sessions.*, users.username, users.role, users.daily_quota,
                      users.active_quota, users.disabled
               FROM sessions LEFT JOIN users ON users.id=sessions.user_id
               WHERE sessions.token_hash=? AND sessions.expires_at > ?''',
            (token_hash(token), utc_now()),
        ).fetchone()
    if row and not row['disabled']:
        connection.execute(
            'UPDATE sessions SET last_seen_at=?, expires_at=? WHERE id=?',
            (utc_now(), future_iso(SESSION_TTL_HOURS), row['id']),
        )
        set_session_cookie(response, token)
        return dict(row)
    if not create:
        return None
    return create_session(connection, response)


def rotate_session(connection, request, response, user_id):
    old = get_session(connection, request, response, create=False)
    if old:
        connection.execute('UPDATE jobs SET owner_user_id=? WHERE owner_session_id=?', (user_id, old['id']))
        connection.execute('UPDATE batches SET owner_user_id=? WHERE owner_session_id=?', (user_id, old['id']))
        connection.execute('DELETE FROM sessions WHERE id=?', (old['id'],))
    return create_session(connection, response, user_id=user_id)


def clear_session(connection, request, response):
    token = unsigned_token(request.cookies.get(SESSION_COOKIE))
    if token:
        connection.execute('DELETE FROM sessions WHERE token_hash=?', (token_hash(token),))
    response.delete_cookie(SESSION_COOKIE, path='/', secure=COOKIE_SECURE, samesite='lax')


def require_csrf(request, session):
    supplied = request.headers.get('X-CSRF-Token', '')
    if not supplied or not secrets.compare_digest(supplied, session['csrf_token']):
        raise HTTPException(403, 'CSRF validation failed')


def require_user(session):
    if not session or not session.get('user_id'):
        raise HTTPException(401, 'Login required')
    return session


def require_admin(session):
    require_user(session)
    if session.get('role') != ROLE_ADMIN:
        raise HTTPException(403, 'Administrator permission required')
    return session


def owner_clause(session, prefix=''):
    column = lambda name: f'{prefix}{name}'
    if session.get('user_id'):
        return f'{column("owner_user_id") }=?', (session['user_id'],)
    return f'{column("owner_session_id") }=? AND {column("owner_user_id")} IS NULL', (session['id'],)
