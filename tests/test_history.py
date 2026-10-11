# =============================================================================
# test_history.py — Automated tests for the SQLite prediction history layer.
#
# Run:  python -m pytest tests/test_history.py -v
# =============================================================================

import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import webapp.history as history


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the module at a throwaway DB file and initialise the schema."""
    db_file = tmp_path / "test_history.db"
    monkeypatch.setattr(history, "DB_PATH", str(db_file))
    history.init_db()
    return str(db_file)


def _probs(predicted: int, confidence: float) -> list:
    """Synthetic 10-class softmax vector with one dominant class."""
    probs = [0.0] * 10
    probs[predicted] = confidence
    probs[predicted % 9] = round(1.0 - confidence, 4)   # put remainder elsewhere
    return probs


class TestSaveAndRead:

    def test_save_prediction_returns_increasing_ids(self, db):
        id1 = history.save_prediction("s1", 3, 0.97, _probs(3, 0.97))
        id2 = history.save_prediction("s1", 7, 0.91, _probs(7, 0.91))
        assert isinstance(id1, int)
        assert isinstance(id2, int)
        assert id2 > id1

    def test_get_history_returns_stored_row(self, db):
        row_id = history.save_prediction("s1", 5, 0.88, _probs(5, 0.88))
        rows = history.get_history("s1")
        assert len(rows) == 1
        assert rows[0]["id"] == row_id
        assert rows[0]["predicted"] == 5
        assert rows[0]["confidence"] == pytest.approx(0.88)
        assert len(rows[0]["top3"]) == 3

    def test_history_is_sorted_newest_first(self, db):
        history.save_prediction("s1", 1, 0.9, _probs(1, 0.9))
        history.save_prediction("s1", 2, 0.9, _probs(2, 0.9))
        rows = history.get_history("s1")
        assert [r["predicted"] for r in rows] == [2, 1]


class TestSessionIsolation:

    def test_sessions_do_not_see_each_other(self, db):
        history.save_prediction("alice", 3, 0.9, _probs(3, 0.9))
        history.save_prediction("bob",   8, 0.9, _probs(8, 0.9))

        assert len(history.get_history("alice")) == 1
        assert history.get_history("alice")[0]["predicted"] == 3
        assert len(history.get_history("bob")) == 1
        assert history.get_history("bob")[0]["predicted"] == 8

    def test_summary_stats_are_scoped_to_session(self, db):
        history.save_prediction("alice", 3, 0.9, _probs(3, 0.9))
        history.save_prediction("alice", 3, 0.8, _probs(3, 0.8))
        history.save_prediction("bob",   8, 0.7, _probs(8, 0.7))

        stats = history.get_summary_stats("alice")
        assert stats["total"] == 2
        assert stats["digit_counts"] == {3: 2}
        assert stats["most_common"] == 3
        assert stats["avg_conf"] == pytest.approx(0.85)

        bob_stats = history.get_summary_stats("bob")
        assert bob_stats["total"] == 1

    def test_clear_history_only_clears_session(self, db):
        history.save_prediction("alice", 1, 0.9, _probs(1, 0.9))
        history.save_prediction("bob",   2, 0.9, _probs(2, 0.9))

        assert history.clear_history("alice") == 1
        assert history.get_summary_stats("alice")["total"] == 0
        assert history.get_summary_stats("bob")["total"] == 1


class TestFeedback:

    def test_set_feedback_correct(self, db):
        row_id = history.save_prediction("s1", 5, 0.9, _probs(5, 0.9))
        history.set_feedback(row_id, user_label=5)
        rows = history.get_history("s1")
        assert rows[0]["user_label"] == 5
        assert rows[0]["is_correct"] == 1

    def test_set_feedback_incorrect(self, db):
        row_id = history.save_prediction("s1", 5, 0.9, _probs(5, 0.9))
        history.set_feedback(row_id, user_label=9)
        rows = history.get_history("s1")
        assert rows[0]["user_label"] == 9
        assert rows[0]["is_correct"] == 0

    def test_feedback_is_null_when_not_given(self, db):
        history.save_prediction("s1", 5, 0.9, _probs(5, 0.9))
        rows = history.get_history("s1")
        assert rows[0]["user_label"] is None
        assert rows[0]["is_correct"] is None


class TestDelete:

    def test_delete_prediction_removes_one_row(self, db):
        id1 = history.save_prediction("s1", 1, 0.9, _probs(1, 0.9))
        id2 = history.save_prediction("s1", 2, 0.9, _probs(2, 0.9))

        assert history.delete_prediction(id1) == 1
        rows = history.get_history("s1")
        assert [r["id"] for r in rows] == [id2]

    def test_delete_prediction_missing_id_returns_zero(self, db):
        assert history.delete_prediction(9999) == 0


class TestExport:

    def test_export_history_csv_has_header(self, db):
        history.save_prediction("s1", 4, 0.9, _probs(4, 0.9))
        csv_text = history.export_history_csv("s1")
        lines = csv_text.strip().splitlines()
        assert "predicted" in lines[0]
        assert "confidence" in lines[0]
        assert len(lines) == 2  # header + one row


class TestMigration:

    def test_old_schema_gets_new_columns(self, tmp_path, monkeypatch):
        """A v1 table without the feedback columns must upgrade cleanly."""
        db_file = tmp_path / "old_history.db"
        conn = sqlite3.connect(db_file)
        conn.execute("""
            CREATE TABLE predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                predicted   INTEGER NOT NULL,
                confidence  REAL    NOT NULL,
                top3        TEXT    NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO predictions (timestamp, predicted, confidence, top3) "
            "VALUES ('2026-01-01 00:00:00', 3, 0.9, '[[3,0.9]]')"
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(history, "DB_PATH", str(db_file))
        history.init_db()

        conn = sqlite3.connect(db_file)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        assert {"session_id", "user_label", "is_correct"} <= columns
        # Existing legacy rows are readable (default session_id = 'legacy')
        rows = history.get_history("legacy")
        assert len(rows) == 1
        assert rows[0]["predicted"] == 3
        conn.close()