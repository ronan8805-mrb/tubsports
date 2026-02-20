"""
Separate DuckDB database for predictions and reconciliation.

Stores predictions, actual results (copied from main DB), and model
version metadata.  Completely isolated from horse.duckdb so the API
can write predictions without conflicting with read-only race queries.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb

from .config import DATA_DIR

logger = logging.getLogger(__name__)

PRED_DB_PATH = DATA_DIR / "predictions.duckdb"


def get_pred_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the predictions database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(PRED_DB_PATH), read_only=read_only)


def init_prediction_db() -> None:
    """Create all tables if they don't exist."""
    con = get_pred_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id   INTEGER PRIMARY KEY,
                race_id         INTEGER NOT NULL,
                horse_name      VARCHAR NOT NULL,
                win_prob        DOUBLE,
                place_prob      DOUBLE,
                win_rank        INTEGER,
                place_rank      INTEGER,
                model_version   VARCHAR,
                model_type      VARCHAR,
                predicted_at    TIMESTAMP DEFAULT current_timestamp,
                UNIQUE(race_id, horse_name, model_version)
            )
        """)
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_pred_id START 1")

        con.execute("""
            CREATE TABLE IF NOT EXISTS reconciled_results (
                id              INTEGER PRIMARY KEY,
                race_id         INTEGER NOT NULL,
                horse_name      VARCHAR NOT NULL,
                actual_position INTEGER,
                sp_decimal      DOUBLE,
                num_runners     INTEGER,
                meeting_date    DATE,
                course          VARCHAR,
                reconciled_at   TIMESTAMP DEFAULT current_timestamp,
                UNIQUE(race_id, horse_name)
            )
        """)
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_recon_id START 1")

        con.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id              INTEGER PRIMARY KEY,
                model_version   VARCHAR NOT NULL,
                target          VARCHAR NOT NULL,
                trained_at      DATE,
                auc             DOUBLE,
                brier           DOUBLE,
                logloss         DOUBLE,
                accuracy        DOUBLE,
                feature_count   INTEGER,
                train_size      INTEGER,
                test_size       INTEGER,
                created_at      TIMESTAMP DEFAULT current_timestamp,
                UNIQUE(model_version, target)
            )
        """)
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_model_id START 1")

        _create_indexes(con)
        logger.info(f"Prediction DB initialized at {PRED_DB_PATH}")
    finally:
        con.close()


def _create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    indexes = [
        ("idx_pred_race", "predictions", "race_id"),
        ("idx_pred_horse", "predictions", "horse_name"),
        ("idx_pred_at", "predictions", "predicted_at"),
        ("idx_recon_race", "reconciled_results", "race_id"),
        ("idx_recon_date", "reconciled_results", "meeting_date"),
    ]
    for name, table, column in indexes:
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})")
        except Exception:
            pass


def migrate_from_main_db(main_db_path: Optional[Path] = None) -> int:
    """One-time migration: copy predictions from horse.duckdb to predictions.duckdb.
    Returns the number of rows migrated.  Skips rows already present."""
    from .config import DB_PATH
    src_path = str(main_db_path or DB_PATH)

    try:
        src = duckdb.connect(src_path, read_only=True)
    except Exception as e:
        logger.warning(f"Cannot open main DB for migration: {e}")
        return 0

    try:
        count = src.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if count == 0:
            return 0

        rows = src.execute("""
            SELECT race_id, horse_name, win_prob, place_prob,
                   win_rank, place_rank, model_version, model_type, predicted_at
            FROM predictions
        """).fetchall()
    except Exception as e:
        logger.info(f"No predictions table in main DB or read error: {e}")
        return 0
    finally:
        src.close()

    if not rows:
        return 0

    dst = get_pred_connection()
    migrated = 0
    try:
        for r in rows:
            try:
                pid = dst.execute("SELECT nextval('seq_pred_id')").fetchone()[0]
                dst.execute("""
                    INSERT INTO predictions (prediction_id, race_id, horse_name,
                        win_prob, place_prob, win_rank, place_rank,
                        model_version, model_type, predicted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (race_id, horse_name, model_version) DO NOTHING
                """, [pid, *r])
                migrated += 1
            except Exception:
                pass
        logger.info(f"Migrated {migrated}/{len(rows)} predictions to predictions.duckdb")
    finally:
        dst.close()
    return migrated


# Auto-initialize on import
init_prediction_db()
