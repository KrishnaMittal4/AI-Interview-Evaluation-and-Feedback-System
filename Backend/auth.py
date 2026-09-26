"""
auth.py — Aura AI | Multi-User Authentication & Session Store
=============================================================
Provides:
  - SQLite-backed user store  (users, sessions, interview_history tables)
  - bcrypt password hashing
  - JWT access tokens (HS256, 24h expiry)
  - Per-user interview history persistence
  - FastAPI dependency  get_current_user()  for protected routes

WHY SQLITE?
  Zero external services — runs anywhere Python runs.
  Swap the DB_PATH for a Postgres URL and swap sqlite3 for asyncpg
  when you're ready to scale horizontally.

TABLES
------
users
  id          TEXT  PRIMARY KEY  (uuid4)
  username    TEXT  UNIQUE NOT NULL
  email       TEXT  UNIQUE NOT NULL
  password_hash TEXT NOT NULL
  created_at  REAL
  last_login  REAL
  total_sessions INT DEFAULT 0
  avg_score   REAL  DEFAULT 0.0
  best_score  REAL  DEFAULT 0.0
  role_focus  TEXT  DEFAULT ''     -- last used role

interview_history
  id          TEXT PRIMARY KEY  (uuid4)
  user_id     TEXT NOT NULL  REFERENCES users(id)
  session_id  TEXT NOT NULL
  role        TEXT
  difficulty  TEXT
  num_questions INT
  avg_score   REAL
  avg_nervousness REAL
  avg_star    REAL
  hr_recommendation TEXT
  answers_json TEXT   -- JSON blob of answer records
  report_json  TEXT   -- JSON blob of full report
  completed_at REAL

ENVIRONMENT
-----------
  JWT_SECRET   — signing secret (defaults to a hard-coded dev key, CHANGE IN PROD)
  DB_PATH      — path to SQLite file (default: ./aura_users.db)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET  = os.getenv("JWT_SECRET", "aura-ai-dev-secret-change-in-production-2025")
JWT_ALG     = "HS256"
JWT_EXPIRY  = 60 * 60 * 24      # 24 hours in seconds
DB_PATH     = os.getenv("DB_PATH", "./aura_users.db")


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
    return conn


@contextmanager
def _db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    REAL NOT NULL,
            last_login    REAL,
            total_sessions INT DEFAULT 0,
            avg_score     REAL DEFAULT 0.0,
            best_score    REAL DEFAULT 0.0,
            role_focus    TEXT DEFAULT '',
            streak_current  INT  DEFAULT 0,
            streak_best     INT  DEFAULT 0,
            streak_last_date TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS interview_history (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            session_id      TEXT NOT NULL,
            role            TEXT,
            difficulty      TEXT,
            num_questions   INT,
            avg_score       REAL,
            avg_nervousness REAL,
            avg_star        REAL,
            hr_recommendation TEXT,
            answers_json    TEXT,
            report_json     TEXT,
            completed_at    REAL
        );

        CREATE INDEX IF NOT EXISTS idx_history_user ON interview_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_history_time  ON interview_history(completed_at DESC);

        CREATE TABLE IF NOT EXISTS daily_completions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            date_str    TEXT NOT NULL,
            question    TEXT NOT NULL,
            score       REAL NOT NULL,
            answer      TEXT,
            completed_at REAL NOT NULL,
            UNIQUE(user_id, date_str)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_user ON daily_completions(user_id);
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_completions(date_str);
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PASSWORD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  JWT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_token(user_id: str, username: str) -> str:
    payload = {
        "sub":      user_id,
        "username": username,
        "iat":      int(time.time()),
        "exp":      int(time.time()) + JWT_EXPIRY,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> Dict:
    """Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])


# ══════════════════════════════════════════════════════════════════════════════
#  USER CRUD
# ══════════════════════════════════════════════════════════════════════════════

def create_user(username: str, email: str, password: str) -> Dict:
    """
    Create a new user. Raises ValueError on duplicate username/email.
    Returns the user dict (no password_hash).
    """
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    now     = time.time()

    with _db() as conn:
        try:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username.strip().lower(), email.strip().lower(), pw_hash, now),
            )
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username already taken.")
            if "email" in str(e):
                raise ValueError("Email already registered.")
            raise ValueError("Registration failed.")

    return {"id": user_id, "username": username, "email": email, "created_at": now}


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict]:
    """
    Verify credentials. Returns user dict on success, None on failure.
    """
    q = username_or_email.strip().lower()
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? OR email=?", (q, q)
        ).fetchone()

    if not row or not verify_password(password, row["password_hash"]):
        return None

    # Update last_login
    with _db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (time.time(), row["id"]))

    return dict(row)


def get_user_by_id(user_id: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_profile(user_id: str) -> Optional[Dict]:
    """Returns public profile fields (no password_hash)."""
    u = get_user_by_id(user_id)
    if not u:
        return None
    u.pop("password_hash", None)
    return u


def update_user_stats(user_id: str, avg_score: float, best_score: float) -> None:
    """
    Recompute user-level aggregated stats after a completed session.
    avg_score and best_score are already the session values — we blend them
    into the running lifetime average.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT total_sessions, avg_score, best_score FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row:
            return

        n   = row["total_sessions"]
        old = row["avg_score"]
        old_best = row["best_score"]

        new_n    = n + 1
        new_avg  = round((old * n + avg_score) / new_n, 3)
        new_best = max(old_best, best_score)

        conn.execute(
            "UPDATE users SET total_sessions=?, avg_score=?, best_score=? WHERE id=?",
            (new_n, new_avg, new_best, user_id),
        )


def update_user_role(user_id: str, role: str) -> None:
    with _db() as conn:
        conn.execute("UPDATE users SET role_focus=? WHERE id=?", (role, user_id))


# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_interview(
    user_id: str,
    session_id: str,
    role: str,
    difficulty: str,
    num_questions: int,
    avg_score: float,
    avg_nervousness: float,
    avg_star: float,
    hr_recommendation: str,
    answers: list,
    report: dict,
) -> str:
    """
    Persist a completed interview to interview_history.
    Returns the new history record id.
    """
    record_id = str(uuid.uuid4())
    now       = time.time()

    with _db() as conn:
        conn.execute(
            """INSERT INTO interview_history
               (id, user_id, session_id, role, difficulty, num_questions,
                avg_score, avg_nervousness, avg_star, hr_recommendation,
                answers_json, report_json, completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id, user_id, session_id, role, difficulty, num_questions,
                avg_score, avg_nervousness, avg_star, hr_recommendation,
                json.dumps(answers, default=str),
                json.dumps(report, default=str),
                now,
            ),
        )

    # Update lifetime stats
    update_user_stats(user_id, avg_score, avg_score)
    update_user_role(user_id, role)

    return record_id


def get_history(user_id: str, limit: int = 20, offset: int = 0) -> list:
    """Return paginated interview history for a user (newest first)."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, session_id, role, difficulty, num_questions,
                      avg_score, avg_nervousness, avg_star, hr_recommendation,
                      completed_at
               FROM interview_history
               WHERE user_id=?
               ORDER BY completed_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_history_detail(record_id: str, user_id: str) -> Optional[Dict]:
    """Return full detail for one history record (including answers + report)."""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM interview_history WHERE id=? AND user_id=?",
            (record_id, user_id),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["answers"] = json.loads(d.pop("answers_json", "[]") or "[]")
    d["report"]  = json.loads(d.pop("report_json", "{}") or "{}")
    return d


def get_user_stats_summary(user_id: str) -> Dict:
    """
    Returns aggregated stats for dashboard display including streak info.
    """
    with _db() as conn:
        profile = conn.execute(
            """SELECT total_sessions, avg_score, best_score, role_focus,
                      streak_current, streak_best, streak_last_date
               FROM users WHERE id=?""",
            (user_id,),
        ).fetchone()

        recent = conn.execute(
            """SELECT avg_score, avg_nervousness, avg_star, role, completed_at
               FROM interview_history WHERE user_id=?
               ORDER BY completed_at DESC LIMIT 10""",
            (user_id,),
        ).fetchall()

        role_dist = conn.execute(
            """SELECT role, COUNT(*) as cnt FROM interview_history
               WHERE user_id=? GROUP BY role ORDER BY cnt DESC""",
            (user_id,),
        ).fetchall()

    if not profile:
        return {}

    trend = [
        {"score": r["avg_score"], "role": r["role"], "ts": r["completed_at"]}
        for r in recent
    ]

    return {
        "total_sessions":    profile["total_sessions"],
        "avg_score":         round(profile["avg_score"], 2),
        "best_score":        round(profile["best_score"], 2),
        "role_focus":        profile["role_focus"],
        "avg_nervousness":   round(sum(r["avg_nervousness"] for r in recent) / max(len(recent), 1), 3),
        "avg_star_coverage": round(sum(r["avg_star"] for r in recent) / max(len(recent), 1), 3),
        "score_trend":       trend,
        "role_distribution": [{"role": r["role"], "count": r["cnt"]} for r in role_dist],
        "streak_current":    profile["streak_current"] or 0,
        "streak_best":       profile["streak_best"] or 0,
        "streak_last_date":  profile["streak_last_date"] or "",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DAILY CHALLENGE & STREAK
# ══════════════════════════════════════════════════════════════════════════════

def get_daily_status(user_id: str) -> Dict:
    """
    Return whether the user has completed today's daily challenge and their streak.
    """
    from datetime import date as _date
    today_str = _date.today().isoformat()

    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM daily_completions WHERE user_id=? AND date_str=?",
            (user_id, today_str),
        ).fetchone()
        profile = conn.execute(
            "SELECT streak_current, streak_best, streak_last_date FROM users WHERE id=?",
            (user_id,),
        ).fetchone()

    return {
        "completed_today":  row is not None,
        "daily_score":      row["score"] if row else None,
        "streak_current":   profile["streak_current"] if profile else 0,
        "streak_best":      profile["streak_best"] if profile else 0,
        "streak_last_date": profile["streak_last_date"] if profile else "",
    }


def save_daily_completion(
    user_id: str,
    date_str: str,
    question: str,
    score:    float,
    answer:   str = "",
) -> Dict:
    """
    Record a daily challenge completion and update the user's streak.

    Streak rules:
      - Yesterday completion → streak +1
      - Gap > 1 day          → streak resets to 1
      - Same day again       → no-op (already_done=True)
    """
    from datetime import date as _date, timedelta as _timedelta
    import uuid as _uuid, time as _time

    today = _date.today()

    with _db() as conn:
        existing = conn.execute(
            "SELECT score FROM daily_completions WHERE user_id=? AND date_str=?",
            (user_id, date_str),
        ).fetchone()
        if existing:
            profile = conn.execute(
                "SELECT streak_current, streak_best FROM users WHERE id=?", (user_id,)
            ).fetchone()
            return {
                "new_streak":    profile["streak_current"] if profile else 1,
                "streak_best":   profile["streak_best"] if profile else 1,
                "streak_broken": False,
                "already_done":  True,
            }

        conn.execute(
            """INSERT INTO daily_completions
               (id, user_id, date_str, question, score, answer, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(_uuid.uuid4()), user_id, date_str, question, score, answer, _time.time()),
        )

        profile = conn.execute(
            "SELECT streak_current, streak_best, streak_last_date FROM users WHERE id=?",
            (user_id,),
        ).fetchone()

        current   = profile["streak_current"] if profile else 0
        best      = profile["streak_best"] if profile else 0
        last_date = profile["streak_last_date"] if profile else ""

        streak_broken = False
        if last_date:
            try:
                last = _date.fromisoformat(last_date)
                if today - last == _timedelta(days=1):
                    current += 1
                elif today > last:
                    current = 1
                    streak_broken = True
            except ValueError:
                current = 1
        else:
            current = 1

        best = max(best, current)
        conn.execute(
            "UPDATE users SET streak_current=?, streak_best=?, streak_last_date=? WHERE id=?",
            (current, best, date_str, user_id),
        )

    return {
        "new_streak":    current,
        "streak_best":   best,
        "streak_broken": streak_broken,
        "already_done":  False,
    }


class DailySubmitRequest(BaseModel):
    date_str: str
    question: str
    score:    float
    answer:   str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI DEPENDENCY — get_current_user
# ══════════════════════════════════════════════════════════════════════════════

class TokenData(BaseModel):
    user_id:  str
    username: str


def get_current_user(authorization: str = Header(default="")) -> TokenData:
    """
    FastAPI dependency. Extracts Bearer token from Authorization header.
    Raises HTTP 401 on missing / invalid / expired token.

    Usage in routes:
        @app.get("/protected")
        async def protected(user: TokenData = Depends(get_current_user)):
            ...
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization.startswith("Bearer "):
        raise credentials_exc

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
        return TokenData(user_id=payload["sub"], username=payload["username"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise credentials_exc


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS  (used by main.py routes)
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password:          str


class SaveInterviewRequest(BaseModel):
    session_id:        str
    role:              str
    difficulty:        str
    num_questions:     int
    avg_score:         float
    avg_nervousness:   float = 0.0
    avg_star:          float = 0.0
    hr_recommendation: str   = "Maybe"
    answers:           list  = []
    report:            dict  = {}


# ── Init on import ─────────────────────────────────────────────────────────────
init_db()