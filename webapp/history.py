# =============================================================================
# history.py — Persistent prediction history using SQLite.
#
# WHY SQLITE?
#   SQLite is a lightweight database stored as a single file — no server
#   required.  It ships with Python (no extra install needed).
#   It persists predictions between app restarts, unlike st.session_state
#   which resets every time the browser refreshes.
#
# PRIVACY NOTE:
#   We store only (timestamp, predicted_digit, confidence, top_3_classes).
#   Drawings/images are NEVER stored.
#
# MULTI-USER NOTE:
#   Every row is tagged with a session_id generated per browser session.
#   Reads and writes are scoped to the current session, so one visitor can
#   never see another visitor's predictions.
# =============================================================================

import os
import csv
import io
import sqlite3
import datetime
import json
from typing import List, Dict, Optional

from config import HISTORY_DB_PATH

# ── Database file lives in the project data/ folder ───────────────────────────
DB_PATH = HISTORY_DB_PATH


# =============================================================================
# Schema management
# =============================================================================

def _get_connection() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.

    A fresh connection is opened per call and closed by the caller, so there
    is no shared connection state between the Streamlit session threads.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def init_db() -> None:
    """
    Create the predictions table if it does not already exist and apply
    lightweight migrations so older databases keep working.

    Table schema
    ─────────────
    id          INTEGER  — auto-increment primary key
    session_id  TEXT     — per-browser-session identifier (isolation)
    timestamp   TEXT     — ISO-8601 datetime string  e.g. "2026-10-10 09:15:40"
    predicted   INTEGER  — the digit the model predicted (0–9)
    confidence  REAL     — model confidence as a decimal (0.0–1.0)
    top3        TEXT     — JSON string  e.g. '[[3, 0.97], [5, 0.02], [8, 0.01]]'
    user_label  INTEGER  — the true digit supplied by the user (NULL until given)
    is_correct  INTEGER  — 1/0 once feedback is given, else NULL
    """
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL DEFAULT 'legacy',
            timestamp   TEXT    NOT NULL,
            predicted   INTEGER NOT NULL,
            confidence  REAL    NOT NULL,
            top3        TEXT    NOT NULL
        )
    """)

    # ── Migrations for columns added after the first release ──────────────
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    migrations = {
        "session_id": "ALTER TABLE predictions ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'",
        "user_label": "ALTER TABLE predictions ADD COLUMN user_label INTEGER",
        "is_correct": "ALTER TABLE predictions ADD COLUMN is_correct INTEGER",
    }
    for column, statement in migrations.items():
        if column not in existing:
            conn.execute(statement)

    conn.commit()
    conn.close()


# =============================================================================
# Write operations
# =============================================================================

def save_prediction(session_id: str,
                    predicted: int,
                    confidence: float,
                    probabilities: List[float]) -> int:
    """
    Insert one prediction record into the database.

    Parameters
    ----------
    session_id    : str   — browser session identifier (for isolation)
    predicted     : int   — predicted digit (0–9)
    confidence    : float — probability of the top class
    probabilities : list of 10 floats — full softmax output

    Returns
    -------
    int — the new row id (used later to attach user feedback)

    Notes
    -----
    Only the top-3 classes are stored to keep the DB lightweight.
    The full probability list is NOT stored (not needed for history display).
    """
    # Build top-3: list of [digit, probability] sorted by probability desc
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    top3    = [[int(d), round(float(p), 4)] for d, p in indexed[:3]]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn   = _get_connection()
    cursor = conn.execute(
        "INSERT INTO predictions (session_id, timestamp, predicted, confidence, top3) "
        "VALUES (?,?,?,?,?)",
        (session_id, timestamp, int(predicted), float(confidence), json.dumps(top3))
    )
    new_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return new_id


def set_feedback(row_id: int, user_label: int) -> None:
    """
    Record the user's correction for a prediction.

    Parameters
    ----------
    row_id     : int — id of the prediction row being corrected
    user_label : int — the true digit (0–9) as stated by the user
    """
    conn = _get_connection()
    row  = conn.execute(
        "SELECT predicted FROM predictions WHERE id = ?", (row_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return

    is_correct = 1 if int(row["predicted"]) == int(user_label) else 0
    conn.execute(
        "UPDATE predictions SET user_label = ?, is_correct = ? WHERE id = ?",
        (int(user_label), is_correct, row_id)
    )
    conn.commit()
    conn.close()


# =============================================================================
# Read operations
# =============================================================================

def get_history(session_id: str, limit: int = 50) -> List[Dict]:
    """
    Return the most recent `limit` predictions for one session, newest first.

    Returns
    -------
    List of dicts, each with keys:
        id, timestamp, predicted, confidence, top3 (as Python list),
        user_label, is_correct
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, timestamp, predicted, confidence, top3, user_label, is_correct "
        "FROM predictions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()

    return [
        {
            "id"        : row["id"],
            "timestamp" : row["timestamp"],
            "predicted" : row["predicted"],
            "confidence": row["confidence"],
            "top3"      : json.loads(row["top3"]),
            "user_label": row["user_label"],
            "is_correct": row["is_correct"],
        }
        for row in rows
    ]


def get_summary_stats(session_id: str) -> Dict:
    """
    Return aggregate statistics about the predictions in one session.

    Returns
    -------
    dict with:
        total        : int   — total number of predictions made
        avg_conf     : float — average confidence across all predictions
        digit_counts : dict  — {digit: count} for each digit 0–9
        most_common  : int   — digit predicted most often
    """
    conn = _get_connection()

    total_row = conn.execute(
        "SELECT COUNT(*) as n FROM predictions WHERE session_id = ?", (session_id,)
    ).fetchone()
    total = total_row["n"] if total_row else 0

    if total == 0:
        conn.close()
        return {"total": 0, "avg_conf": 0.0, "digit_counts": {}, "most_common": None}

    avg_conf_row = conn.execute(
        "SELECT AVG(confidence) as avg_c FROM predictions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    avg_conf = float(avg_conf_row["avg_c"]) if avg_conf_row else 0.0

    rows = conn.execute(
        "SELECT predicted, COUNT(*) as cnt FROM predictions "
        "WHERE session_id = ? GROUP BY predicted",
        (session_id,)
    ).fetchall()
    conn.close()

    digit_counts = {row["predicted"]: row["cnt"] for row in rows}
    most_common  = max(digit_counts, key=digit_counts.get) if digit_counts else None

    return {
        "total"       : total,
        "avg_conf"    : avg_conf,
        "digit_counts": digit_counts,
        "most_common" : most_common,
    }


def export_history_csv(session_id: str) -> str:
    """
    Return this session's full prediction history as a CSV string.

    Used by the 'Download history (CSV)' button.
    """
    rows = get_history(session_id, limit=1_000_000)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "timestamp", "predicted", "confidence",
                     "top3_classes", "user_label", "is_correct"])
    for r in rows:
        top3_str = " | ".join(f"{d}:{p:.4f}" for d, p in r["top3"])
        writer.writerow([
            r["id"], r["timestamp"], r["predicted"],
            f"{r['confidence']:.4f}", top3_str,
            "" if r["user_label"] is None else r["user_label"],
            "" if r["is_correct"] is None else r["is_correct"],
        ])
    return buffer.getvalue()


# =============================================================================
# Delete operations
# =============================================================================

def delete_prediction(row_id: int) -> int:
    """
    Delete a single prediction record.

    Returns
    -------
    int — number of rows deleted (0 or 1)
    """
    conn   = _get_connection()
    cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (row_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def clear_history(session_id: Optional[str] = None) -> int:
    """
    Delete prediction records.

    Parameters
    ----------
    session_id : if given, only this session's records are deleted;
                 if None, the entire table is cleared.

    Returns
    -------
    int — number of rows deleted
    """
    conn = _get_connection()
    if session_id is None:
        cursor = conn.execute("DELETE FROM predictions")
    else:
        cursor = conn.execute(
            "DELETE FROM predictions WHERE session_id = ?", (session_id,)
        )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
