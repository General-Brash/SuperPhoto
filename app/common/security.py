import base64
import hashlib
import hmac
import secrets


def hash_password(password):
    if len(password) < 10:
        raise ValueError('Password must contain at least 10 characters')
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return 'scrypt$16384$8$1$' + base64.urlsafe_b64encode(salt).decode() + '$' + base64.urlsafe_b64encode(derived).decode()


def verify_password(password, encoded):
    try:
        algorithm, n, r, p, salt_text, expected_text = encoded.split('$')
        if algorithm != 'scrypt':
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode())
        expected = base64.urlsafe_b64decode(expected_text.encode())
        actual = hashlib.scrypt(
            password.encode('utf-8'), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def random_token(bytes_count=32):
    return secrets.token_urlsafe(bytes_count)


def token_hash(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()
