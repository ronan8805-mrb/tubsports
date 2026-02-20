"""
DuckDB schema and helpers for UK/IRE Horse Racing.
Completely separate from the greyhound database.

Designed to store EVERY field from The Racing API (PRO plan).
All tables include raw_json columns for future-proofing.
All entities keyed by API IDs for idempotent inserts.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import duckdb

from .config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


# ===================================================================
# Connection
# ===================================================================

def get_connection(db_path: Optional[Path] = None,
                   read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the horse racing DuckDB database."""
    path = str(db_path or DB_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


# ===================================================================
# Schema initialisation
# ===================================================================

def init_database(db_path: Optional[Path] = None) -> None:
    """Create all tables for the horse racing system."""
    con = get_connection(db_path)
    try:
        _create_core_tables(con)
        _create_entity_tables(con)
        _create_enrichment_tables(con)
        _create_support_tables(con)
        _create_indexes(con)
        cleanup_duplicate_draws(con)
        logger.info(f"Horse racing database initialized at {db_path or DB_PATH}")
    finally:
        con.close()


def _create_core_tables(con):
    """Meetings, races, results -- the backbone."""

    # --- Meetings (one per course per day) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            meeting_id      INTEGER PRIMARY KEY,
            course          VARCHAR NOT NULL,
            country         VARCHAR NOT NULL DEFAULT 'GB',
            region          VARCHAR,
            region_code     VARCHAR,
            meeting_date    DATE NOT NULL,
            going           VARCHAR,
            rail_position   VARCHAR,
            weather_temp    DOUBLE,
            weather_humidity DOUBLE,
            weather_rain    DOUBLE,
            weather_wind_speed DOUBLE,
            weather_wind_dir DOUBLE,
            weather_pressure DOUBLE,
            non_runners_count INTEGER DEFAULT 0,
            api_course_id   VARCHAR,
            scraped_at      TIMESTAMP DEFAULT current_timestamp,
            UNIQUE(course, meeting_date)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_meeting_id START 1")

    # --- Races (one per race within a meeting) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS races (
            race_id             INTEGER PRIMARY KEY,
            api_race_id         VARCHAR UNIQUE,
            meeting_id          INTEGER NOT NULL REFERENCES meetings(meeting_id),
            race_number         INTEGER,
            race_time           VARCHAR,
            off_dt              TIMESTAMP,
            race_name           VARCHAR,
            distance_furlongs   DOUBLE,
            distance_yards      INTEGER,
            distance_metres     INTEGER,
            race_type           VARCHAR,
            race_class          VARCHAR,
            pattern             VARCHAR,
            rating_band         VARCHAR,
            age_band            VARCHAR,
            sex_rest            VARCHAR,
            surface             VARCHAR,
            going_description   VARCHAR,
            jumps_detail        VARCHAR,
            handicap_flag       BOOLEAN DEFAULT FALSE,
            prize               VARCHAR,
            num_runners         INTEGER,
            non_runners_text    VARCHAR,
            tote_win            VARCHAR,
            tote_pl             VARCHAR,
            tote_ex             VARCHAR,
            tote_csf            VARCHAR,
            tote_tricast        VARCHAR,
            tote_trifecta       VARCHAR,
            winning_time_detail VARCHAR,
            region              VARCHAR,
            region_code         VARCHAR,
            api_course_id       VARCHAR,
            raw_json            VARCHAR,
            UNIQUE(meeting_id, race_number)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_race_id START 1")

    # --- Results (one per runner per race) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            result_id       INTEGER PRIMARY KEY,
            race_id         INTEGER NOT NULL REFERENCES races(race_id),
            horse_id        INTEGER NOT NULL,
            horse_name      VARCHAR NOT NULL,
            api_horse_id    VARCHAR,
            position        INTEGER,
            number          INTEGER,
            draw            INTEGER,
            beaten_lengths  DOUBLE,
            ovr_btn         DOUBLE,
            finish_time     DOUBLE,
            sp              VARCHAR,
            sp_decimal      DOUBLE,
            betfair_sp      DOUBLE,
            rpr             INTEGER,
            tsr             INTEGER,
            official_rating INTEGER,
            weight_carried  VARCHAR,
            weight_lbs      INTEGER,
            age             INTEGER,
            sex             VARCHAR,
            headgear        VARCHAR,
            comment         VARCHAR,
            jockey_name     VARCHAR,
            jockey_claim_lbs INTEGER,
            trainer_name    VARCHAR,
            owner           VARCHAR,
            sire            VARCHAR,
            dam             VARCHAR,
            damsire         VARCHAR,
            silk_url        VARCHAR,
            prize           VARCHAR,
            api_jockey_id   VARCHAR,
            api_trainer_id  VARCHAR,
            api_owner_id    VARCHAR,
            api_sire_id     VARCHAR,
            api_dam_id      VARCHAR,
            api_damsire_id  VARCHAR,
            jockey_id       INTEGER,
            trainer_id      INTEGER,
            speed_figure    DOUBLE,
            raw_json        VARCHAR,
            UNIQUE(race_id, horse_name)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_result_id START 1")


def _create_entity_tables(con):
    """Horses, jockeys, trainers, courses -- keyed by API IDs."""

    # --- Horses ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS horses (
            horse_id                INTEGER PRIMARY KEY,
            api_horse_id            VARCHAR UNIQUE,
            name                    VARCHAR NOT NULL UNIQUE,
            age                     INTEGER,
            sex                     VARCHAR,
            colour                  VARCHAR,
            country_bred            VARCHAR,
            sire                    VARCHAR,
            dam                     VARCHAR,
            damsire                 VARCHAR,
            grandsire               VARCHAR,
            granddam                VARCHAR,
            sire_sire               VARCHAR,
            sire_dam                VARCHAR,
            dam_sire                VARCHAR,
            dam_dam                 VARCHAR,
            api_sire_id             VARCHAR,
            api_dam_id              VARCHAR,
            api_damsire_id          VARCHAR,
            trainer_id              INTEGER,
            official_rating_flat    INTEGER,
            official_rating_aw      INTEGER,
            official_rating_chase   INTEGER,
            official_rating_hurdle  INTEGER,
            timeform_rating         INTEGER,
            best_flat_distance_furlongs DOUBLE,
            profile_scraped_at      TIMESTAMP,
            raw_json                VARCHAR,
            created_at              TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_horse_id START 1")

    # --- Jockeys ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS jockeys (
            jockey_id       INTEGER PRIMARY KEY,
            api_id          VARCHAR UNIQUE,
            name            VARCHAR NOT NULL UNIQUE,
            country         VARCHAR,
            total_rides     INTEGER DEFAULT 0,
            total_wins      INTEGER DEFAULT 0,
            win_rate        DOUBLE DEFAULT 0,
            place_rate      DOUBLE DEFAULT 0,
            profile_scraped_at TIMESTAMP,
            raw_json        VARCHAR,
            updated_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_jockey_id START 1")

    # --- Trainers ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS trainers (
            trainer_id      INTEGER PRIMARY KEY,
            api_id          VARCHAR UNIQUE,
            name            VARCHAR NOT NULL UNIQUE,
            country         VARCHAR,
            total_runners   INTEGER DEFAULT 0,
            total_wins      INTEGER DEFAULT 0,
            win_rate        DOUBLE DEFAULT 0,
            place_rate      DOUBLE DEFAULT 0,
            profile_scraped_at TIMESTAMP,
            raw_json        VARCHAR,
            updated_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_trainer_id START 1")

    # --- Courses ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            course          VARCHAR PRIMARY KEY,
            api_course_id   VARCHAR UNIQUE,
            country         VARCHAR NOT NULL DEFAULT 'GB',
            region_code     VARCHAR,
            direction       VARCHAR,
            surface         VARCHAR DEFAULT 'turf',
            track_type      VARCHAR,
            draw_bias_score DOUBLE DEFAULT 0,
            pace_bias       VARCHAR,
            lat             DOUBLE,
            lon             DOUBLE
        )
    """)


def _create_enrichment_tables(con):
    """Pedigree, sire/dam stats, ratings, horse form -- deep data."""

    # --- Pedigree (extended 4-5 generation) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS pedigree (
            pedigree_id     INTEGER PRIMARY KEY,
            horse_name      VARCHAR NOT NULL UNIQUE,
            api_horse_id    VARCHAR,
            sire            VARCHAR,
            dam             VARCHAR,
            sire_sire       VARCHAR,
            sire_dam        VARCHAR,
            dam_sire        VARCHAR,
            dam_dam         VARCHAR,
            sire_sire_sire  VARCHAR,
            sire_sire_dam   VARCHAR,
            sire_dam_sire   VARCHAR,
            sire_dam_dam    VARCHAR,
            dam_sire_sire   VARCHAR,
            dam_sire_dam    VARCHAR,
            dam_dam_sire    VARCHAR,
            dam_dam_dam     VARCHAR,
            scraped_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_pedigree_id START 1")

    # --- Sire stats (aggregated breeding data) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS sire_stats (
            sire_id             INTEGER PRIMARY KEY,
            api_sire_id         VARCHAR UNIQUE,
            sire_name           VARCHAR NOT NULL UNIQUE,
            progeny_count       INTEGER DEFAULT 0,
            progeny_wins        INTEGER DEFAULT 0,
            progeny_win_rate    DOUBLE DEFAULT 0,
            avg_distance_furlongs DOUBLE,
            going_preference    VARCHAR,
            total_prize_money   DOUBLE DEFAULT 0,
            flat_win_rate       DOUBLE DEFAULT 0,
            nh_win_rate         DOUBLE DEFAULT 0,
            raw_json            VARCHAR,
            updated_at          TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_sire_id START 1")

    # --- Sire distance breakdown (per-distance win rates) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS sire_distance_stats (
            api_sire_id     VARCHAR NOT NULL,
            distance_band   VARCHAR NOT NULL,
            distance_f      DOUBLE,
            runners         INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            win_rate        DOUBLE DEFAULT 0,
            place_rate      DOUBLE DEFAULT 0,
            PRIMARY KEY (api_sire_id, distance_band)
        )
    """)

    # --- Dam stats ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS dam_stats (
            dam_id              INTEGER PRIMARY KEY,
            api_dam_id          VARCHAR UNIQUE,
            dam_name            VARCHAR NOT NULL UNIQUE,
            progeny_count       INTEGER DEFAULT 0,
            progeny_wins        INTEGER DEFAULT 0,
            progeny_win_rate    DOUBLE DEFAULT 0,
            avg_distance_furlongs DOUBLE,
            going_preference    VARCHAR,
            raw_json            VARCHAR,
            updated_at          TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_dam_id START 1")

    # --- Damsire stats ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS damsire_stats (
            damsire_id          INTEGER PRIMARY KEY,
            api_damsire_id      VARCHAR UNIQUE,
            damsire_name        VARCHAR NOT NULL UNIQUE,
            grandoffspring_count INTEGER DEFAULT 0,
            grandoffspring_wins  INTEGER DEFAULT 0,
            grandoffspring_win_rate DOUBLE DEFAULT 0,
            raw_json            VARCHAR,
            updated_at          TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_damsire_id START 1")

    # --- Ratings history (BHA + IHRB weekly snapshots) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS ratings_history (
            rating_id       INTEGER PRIMARY KEY,
            horse_id        INTEGER REFERENCES horses(horse_id),
            horse_name      VARCHAR NOT NULL,
            rating_date     DATE NOT NULL,
            rating_type     VARCHAR NOT NULL,
            rating          INTEGER NOT NULL,
            source          VARCHAR NOT NULL,
            UNIQUE(horse_name, rating_date, rating_type, source)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_rating_id START 1")

    # --- Horse form (detailed per-race from profiles) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS horse_form (
            form_id             INTEGER PRIMARY KEY,
            horse_name          VARCHAR NOT NULL,
            api_horse_id        VARCHAR,
            race_date           DATE NOT NULL,
            course              VARCHAR,
            distance_furlongs   DOUBLE,
            race_type           VARCHAR,
            race_class          VARCHAR,
            going               VARCHAR,
            position            INTEGER,
            beaten_lengths      DOUBLE,
            finish_time         DOUBLE,
            sp_decimal          DOUBLE,
            weight_carried      VARCHAR,
            draw                INTEGER,
            jockey              VARCHAR,
            trainer             VARCHAR,
            official_rating     INTEGER,
            speed_figure        DOUBLE,
            comment             VARCHAR,
            num_runners         INTEGER,
            winner_name         VARCHAR,
            UNIQUE(horse_name, race_date, course)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_form_id START 1")


def _create_support_tables(con):
    """Market data, weather, backfill log, sentiment, predictions."""

    # --- Market data ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            market_id       INTEGER PRIMARY KEY,
            race_id         INTEGER REFERENCES races(race_id),
            horse_name      VARCHAR NOT NULL,
            morning_price   DOUBLE,
            sp_decimal      DOUBLE,
            betfair_sp      DOUBLE,
            exchange_volume DOUBLE,
            back_odds       DOUBLE,
            lay_odds        DOUBLE,
            market_move_pct DOUBLE,
            steam_flag      BOOLEAN DEFAULT FALSE,
            captured_at     TIMESTAMP DEFAULT current_timestamp,
            UNIQUE(race_id, horse_name)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_market_id START 1")

    # --- Weather (from Open-Meteo) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            weather_id      INTEGER PRIMARY KEY,
            course          VARCHAR NOT NULL,
            weather_date    DATE NOT NULL,
            temperature     DOUBLE,
            humidity        DOUBLE,
            precipitation   DOUBLE,
            wind_speed      DOUBLE,
            wind_direction  DOUBLE,
            pressure        DOUBLE,
            fetched_at      TIMESTAMP DEFAULT current_timestamp,
            UNIQUE(course, weather_date)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_weather_id START 1")

    # --- Backfill log (tracks ingestion progress per chunk/phase) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS backfill_log (
            log_id              INTEGER PRIMARY KEY,
            chunk_key           VARCHAR NOT NULL,
            phase               VARCHAR NOT NULL DEFAULT 'results',
            api_total           INTEGER,
            db_race_count       INTEGER,
            db_result_count     INTEGER,
            db_horse_count      INTEGER,
            mean_runners_per_race DOUBLE,
            status              VARCHAR DEFAULT 'pending',
            started_at          TIMESTAMP,
            completed_at        TIMESTAMP,
            error_message       VARCHAR,
            UNIQUE(chunk_key, phase)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_log_id START 1")

    # --- Sentiment (future: X/Twitter) ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            sentiment_id    INTEGER PRIMARY KEY,
            entity_type     VARCHAR NOT NULL,
            entity_name     VARCHAR NOT NULL,
            sentiment_date  DATE NOT NULL,
            sentiment_score DOUBLE,
            hype_level      DOUBLE,
            mention_count   INTEGER DEFAULT 0,
            UNIQUE(entity_type, entity_name, sentiment_date)
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_sentiment_id START 1")

    # --- Predictions cache ---
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id   INTEGER PRIMARY KEY,
            race_id         INTEGER REFERENCES races(race_id),
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
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_prediction_id START 1")


def _create_indexes(con):
    """Indexes optimised for feature engineering queries."""
    indexes = [
        # Core lookups
        ("idx_results_race", "results", "race_id"),
        ("idx_results_horse", "results", "horse_id"),
        ("idx_results_position", "results", "position"),
        ("idx_results_api_horse", "results", "api_horse_id"),
        ("idx_results_api_jockey", "results", "api_jockey_id"),
        ("idx_results_api_trainer", "results", "api_trainer_id"),
        # Meeting/race lookups
        ("idx_meetings_date", "meetings", "meeting_date"),
        ("idx_meetings_course", "meetings", "course"),
        ("idx_races_meeting", "races", "meeting_id"),
        ("idx_races_api_race", "races", "api_race_id"),
        ("idx_races_off_dt", "races", "off_dt"),
        ("idx_races_region", "races", "region_code"),
        ("idx_races_type", "races", "race_type"),
        # Form and ratings
        ("idx_horse_form_name", "horse_form", "horse_name"),
        ("idx_horse_form_date", "horse_form", "race_date"),
        ("idx_ratings_horse", "ratings_history", "horse_name"),
        ("idx_ratings_date", "ratings_history", "rating_date"),
        # Horses
        ("idx_horses_api", "horses", "api_horse_id"),
        ("idx_horses_sire", "horses", "sire"),
        ("idx_horses_dam", "horses", "dam"),
        # Support
        ("idx_market_race", "market_data", "race_id"),
        ("idx_predictions_race", "predictions", "race_id"),
        ("idx_backfill_chunk", "backfill_log", "chunk_key"),
    ]
    for name, table, column in indexes:
        _safe_index(con, name, table, column)


def _safe_index(con, name: str, table: str, column: str):
    """Create index if it doesn't exist."""
    try:
        con.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})")
    except Exception:
        pass


# ===================================================================
# Cleanup
# ===================================================================

def cleanup_duplicate_draws(con: duckdb.DuckDBPyConnection) -> int:
    """Remove ghost entries: duplicate (race_id, draw) rows.
    Keeps the newest result_id per (race_id, draw)."""
    ghost_ids = con.execute("""
        WITH ranked AS (
            SELECT result_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY race_id, draw
                       ORDER BY result_id DESC
                   ) as rn
            FROM results
            WHERE draw IS NOT NULL AND draw > 0
        )
        SELECT result_id FROM ranked WHERE rn > 1
    """).fetchall()

    if not ghost_ids:
        return 0

    ids = [r[0] for r in ghost_ids]
    batch_size = 10000
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        con.execute(
            f"DELETE FROM results WHERE result_id IN ({placeholders})", batch
        )

    logger.warning(f"cleanup_duplicate_draws: removed {len(ids)} ghost rows")
    return len(ids)


# ===================================================================
# API-ID based upsert helpers (idempotent, used by Racing API client)
# ===================================================================

def upsert_meeting(con, course: str, meeting_date, country: str = "GB",
                   **kwargs) -> int:
    """Insert or return existing meeting_id. Keyed by (course, meeting_date)."""
    row = con.execute(
        "SELECT meeting_id FROM meetings WHERE course = ? AND meeting_date = ?",
        [course, meeting_date]
    ).fetchone()
    if row:
        return row[0]

    mid = con.execute("SELECT nextval('seq_meeting_id')").fetchone()[0]
    con.execute("""
        INSERT INTO meetings (meeting_id, course, country, region, region_code,
                              meeting_date, going, api_course_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [mid, course, country,
          kwargs.get("region"), kwargs.get("region_code"),
          meeting_date, kwargs.get("going"), kwargs.get("api_course_id")])
    return mid


def upsert_race_by_api_id(con, api_race_id: str, meeting_id: int,
                          **kw) -> int:
    """Insert or return existing race. Keyed by api_race_id (globally unique)."""
    row = con.execute(
        "SELECT race_id FROM races WHERE api_race_id = ?", [api_race_id]
    ).fetchone()
    if row:
        return row[0]

    rid = con.execute("SELECT nextval('seq_race_id')").fetchone()[0]
    con.execute("""
        INSERT INTO races (
            race_id, api_race_id, meeting_id, race_number, race_time, off_dt,
            race_name, distance_furlongs, distance_yards, distance_metres,
            race_type, race_class, pattern, rating_band, age_band, sex_rest,
            surface, going_description, jumps_detail, handicap_flag, prize,
            num_runners, non_runners_text,
            tote_win, tote_pl, tote_ex, tote_csf, tote_tricast, tote_trifecta,
            winning_time_detail, region, region_code, api_course_id, raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        rid, api_race_id, meeting_id,
        kw.get("race_number"), kw.get("race_time"), kw.get("off_dt"),
        kw.get("race_name"),
        kw.get("distance_furlongs"), kw.get("distance_yards"),
        kw.get("distance_metres"),
        kw.get("race_type"), kw.get("race_class"),
        kw.get("pattern"), kw.get("rating_band"),
        kw.get("age_band"), kw.get("sex_rest"),
        kw.get("surface"), kw.get("going_description"),
        kw.get("jumps_detail"), kw.get("handicap_flag", False),
        kw.get("prize"), kw.get("num_runners"),
        kw.get("non_runners_text"),
        kw.get("tote_win"), kw.get("tote_pl"), kw.get("tote_ex"),
        kw.get("tote_csf"), kw.get("tote_tricast"), kw.get("tote_trifecta"),
        kw.get("winning_time_detail"),
        kw.get("region"), kw.get("region_code"), kw.get("api_course_id"),
        kw.get("raw_json"),
    ])
    return rid


def upsert_horse_by_api_id(con, api_horse_id: str, name: str,
                           **kw) -> int:
    """Insert or return existing horse. Keyed by api_horse_id."""
    if api_horse_id:
        row = con.execute(
            "SELECT horse_id FROM horses WHERE api_horse_id = ?",
            [api_horse_id]
        ).fetchone()
        if row:
            return row[0]

    row = con.execute(
        "SELECT horse_id FROM horses WHERE name = ?", [name]
    ).fetchone()
    if row:
        if api_horse_id:
            con.execute(
                "UPDATE horses SET api_horse_id = ? WHERE horse_id = ?",
                [api_horse_id, row[0]]
            )
        return row[0]

    hid = con.execute("SELECT nextval('seq_horse_id')").fetchone()[0]
    con.execute("""
        INSERT INTO horses (horse_id, api_horse_id, name, age, sex,
                            sire, dam, damsire,
                            api_sire_id, api_dam_id, api_damsire_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [hid, api_horse_id, name,
          kw.get("age"), kw.get("sex"),
          kw.get("sire"), kw.get("dam"), kw.get("damsire"),
          kw.get("api_sire_id"), kw.get("api_dam_id"),
          kw.get("api_damsire_id")])
    return hid


def upsert_jockey_by_api_id(con, api_id: str, name: str,
                            country: str = None) -> int:
    """Insert or return existing jockey. Keyed by api_id."""
    if api_id:
        row = con.execute(
            "SELECT jockey_id FROM jockeys WHERE api_id = ?", [api_id]
        ).fetchone()
        if row:
            return row[0]

    row = con.execute(
        "SELECT jockey_id FROM jockeys WHERE name = ?", [name]
    ).fetchone()
    if row:
        if api_id:
            con.execute(
                "UPDATE jockeys SET api_id = ? WHERE jockey_id = ?",
                [api_id, row[0]]
            )
        return row[0]

    jid = con.execute("SELECT nextval('seq_jockey_id')").fetchone()[0]
    con.execute(
        "INSERT INTO jockeys (jockey_id, api_id, name, country) VALUES (?,?,?,?)",
        [jid, api_id, name, country]
    )
    return jid


def upsert_trainer_by_api_id(con, api_id: str, name: str,
                             country: str = None) -> int:
    """Insert or return existing trainer. Keyed by api_id."""
    if api_id:
        row = con.execute(
            "SELECT trainer_id FROM trainers WHERE api_id = ?", [api_id]
        ).fetchone()
        if row:
            return row[0]

    row = con.execute(
        "SELECT trainer_id FROM trainers WHERE name = ?", [name]
    ).fetchone()
    if row:
        if api_id:
            con.execute(
                "UPDATE trainers SET api_id = ? WHERE trainer_id = ?",
                [api_id, row[0]]
            )
        return row[0]

    tid = con.execute("SELECT nextval('seq_trainer_id')").fetchone()[0]
    con.execute(
        "INSERT INTO trainers (trainer_id, api_id, name, country) VALUES (?,?,?,?)",
        [tid, api_id, name, country]
    )
    return tid


def insert_result_from_api(con, race_id: int, runner: dict,
                           horse_id: int, jockey_id: int,
                           trainer_id: int) -> int:
    """Insert a single runner result from Racing API data.
    Idempotent: skips if (race_id, horse_name) already exists."""
    horse_name = runner.get("horse", "")
    row = con.execute(
        "SELECT result_id FROM results WHERE race_id = ? AND horse_name = ?",
        [race_id, horse_name]
    ).fetchone()
    if row:
        return row[0]

    resid = con.execute("SELECT nextval('seq_result_id')").fetchone()[0]

    pos = runner.get("position")
    try:
        pos = int(pos) if pos and str(pos).isdigit() else None
    except (ValueError, TypeError):
        pos = None

    sp_dec = runner.get("sp_dec")
    try:
        sp_dec = float(sp_dec) if sp_dec else None
    except (ValueError, TypeError):
        sp_dec = None

    bsp = runner.get("bsp")
    try:
        bsp = float(bsp) if bsp else None
    except (ValueError, TypeError):
        bsp = None

    btn = runner.get("btn")
    try:
        btn = float(btn) if btn and str(btn).replace(".", "").replace("-", "").isdigit() else None
    except (ValueError, TypeError):
        btn = None

    ovr = runner.get("ovr_btn")
    try:
        ovr = float(ovr) if ovr and str(ovr).replace(".", "").replace("-", "").isdigit() else None
    except (ValueError, TypeError):
        ovr = None

    or_val = runner.get("or")
    try:
        or_val = int(or_val) if or_val and str(or_val).isdigit() else None
    except (ValueError, TypeError):
        or_val = None

    rpr = runner.get("rpr")
    try:
        rpr = int(rpr) if rpr and str(rpr).isdigit() else None
    except (ValueError, TypeError):
        rpr = None

    tsr = runner.get("tsr")
    try:
        tsr = int(tsr) if tsr and str(tsr).isdigit() else None
    except (ValueError, TypeError):
        tsr = None

    wt_lbs = runner.get("weight_lbs")
    try:
        wt_lbs = int(wt_lbs) if wt_lbs else None
    except (ValueError, TypeError):
        wt_lbs = None

    age = runner.get("age")
    try:
        age = int(age) if age and str(age).isdigit() else None
    except (ValueError, TypeError):
        age = None

    num = runner.get("number")
    try:
        num = int(num) if num and str(num).isdigit() else None
    except (ValueError, TypeError):
        num = None

    draw = runner.get("draw")
    try:
        draw = int(draw) if draw and str(draw).isdigit() else None
    except (ValueError, TypeError):
        draw = None

    claim = runner.get("jockey_claim_lbs")
    try:
        claim = int(claim) if claim and str(claim).isdigit() else None
    except (ValueError, TypeError):
        claim = None

    raw = json.dumps(runner) if runner else None

    con.execute("""
        INSERT INTO results (
            result_id, race_id, horse_id, horse_name, api_horse_id,
            position, number, draw, beaten_lengths, ovr_btn, finish_time,
            sp, sp_decimal, betfair_sp, rpr, tsr, official_rating,
            weight_carried, weight_lbs, age, sex, headgear, comment,
            jockey_name, jockey_claim_lbs, trainer_name,
            owner, sire, dam, damsire, silk_url, prize,
            api_jockey_id, api_trainer_id, api_owner_id,
            api_sire_id, api_dam_id, api_damsire_id,
            jockey_id, trainer_id, raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        resid, race_id, horse_id, horse_name,
        runner.get("horse_id"),
        pos, num, draw, btn, ovr, None,
        runner.get("sp"), sp_dec, bsp, rpr, tsr, or_val,
        runner.get("weight"), wt_lbs, age,
        runner.get("sex"), runner.get("headgear"), runner.get("comment"),
        runner.get("jockey"), claim, runner.get("trainer"),
        runner.get("owner"), runner.get("sire"), runner.get("dam"),
        runner.get("damsire"), runner.get("silk_url"), runner.get("prize"),
        runner.get("jockey_id"), runner.get("trainer_id"),
        runner.get("owner_id"),
        runner.get("sire_id"), runner.get("dam_id"),
        runner.get("damsire_id"),
        jockey_id, trainer_id, raw,
    ])
    return resid


# ===================================================================
# Legacy helpers (kept for backward compatibility with HRI/BHA scrapers)
# ===================================================================

def upsert_horse(con, name: str, **kwargs) -> int:
    """Legacy: insert or return existing horse_id by name."""
    return upsert_horse_by_api_id(con, kwargs.get("api_horse_id"), name, **kwargs)


def upsert_jockey(con, name: str, country: str = None) -> int:
    """Legacy: insert or return existing jockey_id by name."""
    return upsert_jockey_by_api_id(con, None, name, country)


def upsert_trainer(con, name: str, country: str = None) -> int:
    """Legacy: insert or return existing trainer_id by name."""
    return upsert_trainer_by_api_id(con, None, name, country)


def upsert_race(con, meeting_id: int, race_number: int = None,
                **kwargs) -> int:
    """Legacy: insert or return existing race_id by (meeting_id, race_number)."""
    row = con.execute(
        "SELECT race_id FROM races WHERE meeting_id = ? AND race_number = ?",
        [meeting_id, race_number]
    ).fetchone()
    if row:
        return row[0]
    return upsert_race_by_api_id(
        con, kwargs.get("api_race_id", f"legacy_{meeting_id}_{race_number}"),
        meeting_id, race_number=race_number, **kwargs
    )


def insert_rating(con, horse_name: str, rating_date, rating_type: str,
                  rating: int, source: str) -> int:
    """Insert a rating snapshot. Skip duplicates."""
    row = con.execute("""
        SELECT rating_id FROM ratings_history
        WHERE horse_name = ? AND rating_date = ? AND rating_type = ? AND source = ?
    """, [horse_name, rating_date, rating_type, source]).fetchone()
    if row:
        return row[0]

    rid = con.execute("SELECT nextval('seq_rating_id')").fetchone()[0]
    horse_id = None
    try:
        h = con.execute(
            "SELECT horse_id FROM horses WHERE name = ?", [horse_name]
        ).fetchone()
        if h:
            horse_id = h[0]
    except Exception:
        pass

    con.execute("""
        INSERT INTO ratings_history (rating_id, horse_id, horse_name,
                                     rating_date, rating_type, rating, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [rid, horse_id, horse_name, rating_date, rating_type, rating, source])
    return rid


# ===================================================================
# Backfill log helpers
# ===================================================================

def get_backfill_status(con, chunk_key, phase: str = "results") -> Optional[str]:
    """Check if a chunk+phase has been backfilled."""
    row = con.execute(
        "SELECT status FROM backfill_log WHERE chunk_key = ? AND phase = ?",
        [str(chunk_key), phase]
    ).fetchone()
    return row[0] if row else None


def set_backfill_started(con, chunk_key, phase: str = "results"):
    """Mark a chunk as started in the backfill log."""
    key = str(chunk_key)
    existing = con.execute(
        "SELECT log_id FROM backfill_log WHERE chunk_key = ? AND phase = ?",
        [key, phase]
    ).fetchone()
    if existing:
        con.execute("""
            UPDATE backfill_log SET status = 'in_progress',
                                    started_at = current_timestamp
            WHERE log_id = ?
        """, [existing[0]])
    else:
        lid = con.execute("SELECT nextval('seq_log_id')").fetchone()[0]
        con.execute("""
            INSERT INTO backfill_log (log_id, chunk_key, phase, status, started_at)
            VALUES (?, ?, ?, 'in_progress', current_timestamp)
        """, [lid, key, phase])


def set_backfill_completed(con, chunk_key, phase: str = "results",
                           **stats):
    """Mark a chunk as completed with validation stats."""
    con.execute("""
        UPDATE backfill_log
        SET status = ?, api_total = ?, db_race_count = ?,
            db_result_count = ?, db_horse_count = ?,
            mean_runners_per_race = ?, completed_at = current_timestamp,
            error_message = ?
        WHERE chunk_key = ? AND phase = ?
    """, [
        stats.get("status", "completed"),
        stats.get("api_total"), stats.get("db_race_count"),
        stats.get("db_result_count"), stats.get("db_horse_count"),
        stats.get("mean_runners_per_race"),
        stats.get("error_message"),
        str(chunk_key), phase,
    ])


# ===================================================================
# Stats
# ===================================================================

def get_db_stats(con) -> dict:
    """Return row counts for all tables."""
    tables = [
        "meetings", "races", "results", "horses", "jockeys", "trainers",
        "horse_form", "ratings_history", "market_data", "weather",
        "pedigree", "sire_stats", "dam_stats", "damsire_stats",
        "sentiment", "predictions", "backfill_log",
    ]
    stats = {}
    for t in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            stats[t] = count
        except Exception:
            stats[t] = 0
    return stats
