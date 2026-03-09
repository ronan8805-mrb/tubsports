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

AUTH_DB_PATH = DATA_DIR / "auth.duckdb"

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _get_auth_con(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(AUTH_DB_PATH), read_only=read_only)


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
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_user_id START 1
        """)
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
    con = _get_auth_con(read_only=True)
    try:
        row = con.execute(
            "SELECT user_id, username, password_hash, salt, role FROM users WHERE username = ?",
            [body.username],
        ).fetchone()
    finally:
        con.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    uid, uname, pw_hash, salt, role = row
    if not _verify_password(body.password, pw_hash, salt):
        raise HTTPException(status_code=401, detail="Invalid credentials")

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
