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
# =============================================================================

import os
import sqlite3
import datetime
import json
from typing import List, Dict, Optional

# ── Database file lives next to the webapp package ────────────────────────────
_WEBAPP_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEBAPP_DIR)
DB_PATH      = os.path.join(_PROJECT_DIR, "data", "prediction_history.db")


# =============================================================================
# Schema management
# =============================================================================

def _get_connection() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.
    Using check_same_thread=False is safe here because Streamlit runs
    in a single-threaded request model.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def init_db() -> None:
    """
    Create the predictions table if it does not already exist.
    Call this once at app startup.

    Table schema
    ─────────────
    id          INTEGER  — auto-increment primary key
    timestamp   TEXT     — ISO-8601 datetime string  e.g. "2026-10-10 09:15:40"
    predicted   INTEGER  — the digit the model predicted (0–9)
    confidence  REAL     — model confidence as a decimal (0.0–1.0)
    top3        TEXT     — JSON string  e.g. '[[3, 0.97], [5, 0.02], [8, 0.01]]'
    """
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            predicted   INTEGER NOT NULL,
            confidence  REAL    NOT NULL,
            top3        TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# =============================================================================
# Write operations
# =============================================================================

def save_prediction(predicted: int,
                    confidence: float,
                    probabilities: List[float]) -> None:
    """
    Insert one prediction record into the database.

    Parameters
    ----------
    predicted     : int   — predicted digit (0–9)
    confidence    : float — probability of the top class
    probabilities : list of 10 floats — full softmax output

    Notes
    -----
    Only the top-3 classes are stored to keep the DB lightweight.
    The full probability list is NOT stored (not needed for history display).
    """
    # Build top-3: list of [digit, probability] sorted by probability desc
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    top3    = [[int(d), round(float(p), 4)] for d, p in indexed[:3]]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = _get_connection()
    conn.execute(
        "INSERT INTO predictions (timestamp, predicted, confidence, top3) VALUES (?,?,?,?)",
        (timestamp, int(predicted), float(confidence), json.dumps(top3))
    )
    conn.commit()
    conn.close()


# =============================================================================
# Read operations
# =============================================================================

def get_history(limit: int = 50) -> List[Dict]:
    """
    Return the most recent `limit` predictions, newest first.

    Returns
    -------
    List of dicts, each with keys:
        id, timestamp, predicted, confidence, top3 (as Python list)
    """
    conn  = _get_connection()
    rows  = conn.execute(
        "SELECT id, timestamp, predicted, confidence, top3 "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "id"        : row["id"],
            "timestamp" : row["timestamp"],
            "predicted" : row["predicted"],
            "confidence": row["confidence"],
            "top3"      : json.loads(row["top3"]),
        })
    return result


def get_summary_stats() -> Dict:
    """
    Return aggregate statistics about all stored predictions.

    Returns
    -------
    dict with:
        total        : int   — total number of predictions made
        avg_conf     : float — average confidence across all predictions
        digit_counts : dict  — {digit: count} for each digit 0–9
        most_common  : int   — digit predicted most often
    """
    conn = _get_connection()

    total_row = conn.execute("SELECT COUNT(*) as n FROM predictions").fetchone()
    total     = total_row["n"] if total_row else 0

    if total == 0:
        conn.close()
        return {"total": 0, "avg_conf": 0.0, "digit_counts": {}, "most_common": None}

    avg_conf_row = conn.execute(
        "SELECT AVG(confidence) as avg_c FROM predictions"
    ).fetchone()
    avg_conf = float(avg_conf_row["avg_c"]) if avg_conf_row else 0.0

    rows = conn.execute(
        "SELECT predicted, COUNT(*) as cnt FROM predictions GROUP BY predicted"
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


# =============================================================================
# Delete operations
# =============================================================================

def clear_history() -> int:
    """
    Delete ALL records from the predictions table.

    Returns
    -------
    int — number of rows deleted
    """
    conn    = _get_connection()
    cursor  = conn.execute("DELETE FROM predictions")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
