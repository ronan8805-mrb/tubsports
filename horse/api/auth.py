"""
Authentication module for the Horse Racing 13D system.
JWT-based auth with admin and member roles.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb
import jwt
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "h0rs3-r4c1ng-13d-s3cr3t-k3y-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300
_login_attempts: dict[str, list[float]] = {}

AUTH_DB_PATH = DATA_DIR / "auth.duckdb"

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _get_auth_con(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(AUTH_DB_PATH), read_only=read_only)


def _log_security_event(username: str, event_type: str, detail: str = "") -> None:
    try:
        con = _get_auth_con()
        try:
            con.execute(
                "INSERT INTO security_events (username, event_type, detail) VALUES (?, ?, ?)",
                [username, event_type, detail],
            )
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Failed to log security event: {e}")


def _hash_fingerprint(fp: str) -> str:
    return hashlib.sha256(fp.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Password hashing (stdlib only, no bcrypt needed)
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hashed.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, stored_hash)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        payload["sub"] = int(payload["sub"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Dependency: extract current user from Authorization header
# ---------------------------------------------------------------------------

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    return _decode_token(token)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Users table bootstrap
# ---------------------------------------------------------------------------

def ensure_users_table() -> None:
    con = _get_auth_con()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    INTEGER PRIMARY KEY,
                username   VARCHAR NOT NULL UNIQUE,
                password_hash VARCHAR NOT NULL,
                salt       VARCHAR NOT NULL,
                role       VARCHAR NOT NULL DEFAULT 'member',
                device_hash VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_user_id START 1
        """)
        # Add device_hash column if table already exists without it
        try:
            con.execute("ALTER TABLE users ADD COLUMN device_hash VARCHAR")
        except Exception:
            pass
        con.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id   INTEGER PRIMARY KEY DEFAULT nextval('seq_security_id'),
                username   VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                detail     VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        try:
            con.execute("CREATE SEQUENCE IF NOT EXISTS seq_security_id START 1")
        except Exception:
            pass
    finally:
        con.close()


def seed_admin(username: str = "admin", password: str = "Wanker1994") -> None:
    con = _get_auth_con()
    try:
        existing = con.execute(
            "SELECT user_id FROM users WHERE username = ?", [username]
        ).fetchone()
        if existing:
            return
        uid = con.execute("SELECT nextval('seq_user_id')").fetchone()[0]
        pw_hash, salt = _hash_password(password)
        con.execute(
            "INSERT INTO users (user_id, username, password_hash, salt, role) VALUES (?, ?, ?, ?, 'admin')",
            [uid, username, pw_hash, salt],
        )
        logger.info("Admin account seeded: %s", username)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str
    device_fingerprint: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "member"


class UserInfo(BaseModel):
    user_id: int
    username: str
    role: str
    created_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    import time
    now = time.time()
    key = body.username.lower()
    attempts = _login_attempts.get(key, [])
    attempts = [t for t in attempts if now - t < LOGIN_WINDOW_SECONDS]
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        _log_security_event(body.username, "rate_limited", "Too many login attempts")
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again in 5 minutes.")

    con = _get_auth_con(read_only=True)
    try:
        row = con.execute(
            "SELECT user_id, username, password_hash, salt, role, device_hash FROM users WHERE username = ?",
            [body.username],
        ).fetchone()
    finally:
        con.close()

    if not row:
        attempts.append(now)
        _login_attempts[key] = attempts
        if len(attempts) >= 2:
            _log_security_event(body.username, "failed_login", f"Unknown username ({len(attempts)} attempts)")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    uid, uname, pw_hash, salt, role, stored_device = row
    if not _verify_password(body.password, pw_hash, salt):
        attempts.append(now)
        _login_attempts[key] = attempts
        if len(attempts) >= 2:
            _log_security_event(uname, "failed_login", f"Wrong password ({len(attempts)} attempts)")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Device lock check (admin exempt)
    if role != "admin" and body.device_fingerprint:
        fp_hash = _hash_fingerprint(body.device_fingerprint)
        if stored_device is None:
            con = _get_auth_con()
            try:
                con.execute("UPDATE users SET device_hash = ? WHERE user_id = ?", [fp_hash, uid])
            finally:
                con.close()
        elif stored_device != fp_hash:
            _log_security_event(uname, "device_mismatch", "Login attempt from unregistered device")
            raise HTTPException(
                status_code=403,
                detail="Account locked to another device. Contact admin to reset.",
            )

    _login_attempts.pop(key, None)
    token = _create_token(uid, uname, role)
    return LoginResponse(token=token, username=uname, role=role)


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    return {"user_id": user["sub"], "username": user["username"], "role": user["role"]}


@router.get("/users", response_model=list[UserInfo])
def list_users(user: dict = Depends(require_admin)):
    con = _get_auth_con(read_only=True)
    try:
        rows = con.execute(
            "SELECT user_id, username, role, created_at FROM users ORDER BY user_id"
        ).fetchall()
        return [
            UserInfo(user_id=r[0], username=r[1], role=r[2], created_at=str(r[3]))
            for r in rows
        ]
    finally:
        con.close()


@router.post("/users", response_model=UserInfo)
def create_user(body: CreateUserRequest, user: dict = Depends(require_admin)):
    if body.role not in ("admin", "member"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'member'")

    con = _get_auth_con()
    try:
        existing = con.execute(
            "SELECT user_id FROM users WHERE username = ?", [body.username]
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Username already exists")

        uid = con.execute("SELECT nextval('seq_user_id')").fetchone()[0]
        pw_hash, salt = _hash_password(body.password)
        con.execute(
            "INSERT INTO users (user_id, username, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
            [uid, body.username, pw_hash, salt, body.role],
        )
        return UserInfo(
            user_id=uid, username=body.username, role=body.role,
            created_at=datetime.now().isoformat(),
        )
    finally:
        con.close()


@router.delete("/users/{user_id}")
def delete_user(user_id: int, user: dict = Depends(require_admin)):
    if user["sub"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    con = _get_auth_con()
    try:
        existing = con.execute(
            "SELECT username FROM users WHERE user_id = ?", [user_id]
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="User not found")

        con.execute("DELETE FROM users WHERE user_id = ?", [user_id])
        return {"deleted": True, "username": existing[0]}
    finally:
        con.close()


@router.get("/security-events")
def get_security_events(user: dict = Depends(require_admin)):
    con = _get_auth_con(read_only=True)
    try:
        rows = con.execute("""
            SELECT event_id, username, event_type, detail, created_at
            FROM security_events
            ORDER BY created_at DESC
            LIMIT 50
        """).fetchall()
        return [
            {
                "event_id": r[0], "username": r[1], "event_type": r[2],
                "detail": r[3], "created_at": str(r[4]),
            }
            for r in rows
        ]
    finally:
        con.close()


@router.post("/users/{user_id}/reset-device")
def reset_device(user_id: int, user: dict = Depends(require_admin)):
    con = _get_auth_con()
    try:
        row = con.execute("SELECT username FROM users WHERE user_id = ?", [user_id]).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        con.execute("UPDATE users SET device_hash = NULL WHERE user_id = ?", [user_id])
        _log_security_event(row[0], "device_reset", f"Device reset by admin {user['username']}")
        return {"reset": True, "username": row[0]}
    finally:
        con.close()
