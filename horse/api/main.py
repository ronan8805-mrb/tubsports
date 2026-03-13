"""
FastAPI backend for the Horse Racing 13D Prediction System.
Runs on port 8002, completely separate from greyhound (port 8000).
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import json

from ..config import API_PORT, DATA_DIR, DB_PATH
from ..db import get_connection, get_db_stats, init_database
from ..features import build_features_for_races, get_feature_columns
from ..prediction_db import get_pred_connection

PREDICTIONS_CACHE_DIR = DATA_DIR / "predictions_cache"


def _load_cache(target_date: str) -> dict | None:
    cache_path = PREDICTIONS_CACHE_DIR / f"{target_date}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
from .auth import router as auth_router, ensure_users_table, seed_admin, require_admin, get_current_user
from .schemas import (
    BestBetRunner,
    BestBetsResponse,
    DateInfo,
    DimensionScore,
    DriverItem,
    HealthResponse,
    RaceCard,
    RaceInfo,
    RunnerPrediction,
    ThirteenDRace,
)
from .thirteen_d import compute_thirteen_d_scores, compute_drivers

logger = logging.getLogger(__name__)



app = FastAPI(
    title="Horse Racing 13D Engine",
    description="UK/IRE Horse Racing Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tubsports.com",
        "https://www.tubsports.com",
        "https://api.tubsports.com",
        "http://localhost:5174",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


def _get_con():
    """Get a read-only DB connection."""
    return get_connection(read_only=True)


@app.on_event("startup")
def _init_db_on_startup():
    """Create main database if it doesn't exist (fresh Render deploy)."""
    try:
        init_database()
    except Exception as e:
        logger.warning(f"DB init: {e}")


@app.on_event("startup")
def _init_auth_on_startup():
    """Create users table and seed admin account."""
    try:
        ensure_users_table()
        seed_admin()
    except Exception as e:
        logger.warning(f"Auth init: {e}")


@app.on_event("startup")
def _migrate_predictions_on_startup():
    """One-time migration: move any predictions from horse.duckdb to predictions.duckdb."""
    try:
        from ..prediction_db import migrate_from_main_db, get_pred_connection
        pred_con = get_pred_connection(read_only=True)
        try:
            count = pred_con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        finally:
            pred_con.close()
        if count == 0:
            migrated = migrate_from_main_db()
            if migrated > 0:
                logger.info(f"Migrated {migrated} predictions from horse.duckdb")
    except Exception as e:
        logger.debug(f"Prediction migration check: {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health():
    con = _get_con()
    try:
        stats = get_db_stats(con)
        return HealthResponse(
            status="ok",
            timestamp=datetime.now().isoformat(),
            database=str(DB_PATH),
            meetings=stats.get("meetings", 0),
            races=stats.get("races", 0),
            results=stats.get("results", 0),
            horse_form=stats.get("horse_form", 0),
            engine="13D",
        )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Race Card
# ---------------------------------------------------------------------------

@app.get("/api/races", response_model=RaceCard)
def get_races(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    course: Optional[str] = Query(None),
    region: Optional[str] = Query(None, description="Region code: GB, IE, FR, US, AU, HK, JP, or INTL for all international"),
):
    con = _get_con()
    try:
        query_date = date or str(datetime.now().date())

        conditions = ["m.meeting_date = ?"]
        params = [query_date]

        if course:
            conditions.append("LOWER(m.course) = LOWER(?)")
            params.append(course)
        if region:
            if region.upper() == "INTL":
                conditions.append("UPPER(m.country) NOT IN ('GB', 'IE')")
            else:
                conditions.append("UPPER(m.country) = ?")
                params.append(region.upper())

        where = " AND ".join(conditions)

        rows = con.execute(f"""
            SELECT ra.race_id, m.meeting_date, m.course,
                   ra.race_number, ra.race_name,
                   COALESCE(ra.race_time, strftime(ra.off_dt, '%H:%M')) as race_time,
                   ra.distance_furlongs, ra.race_type, ra.race_class,
                   ra.going_description, ra.surface, ra.num_runners,
                   ra.region_code
            FROM races ra
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE {where}
            ORDER BY m.course, COALESCE(ra.off_dt, '9999-01-01'), ra.race_number
        """, params).fetchall()

        cache = _load_cache(query_date)
        cached_races = cache.get("races", {}) if cache else {}

        # Build a lookup: race_id -> actual active runner count from cache
        cached_runner_counts = {}
        for rid_str, race_data in cached_races.items():
            cached_runner_counts[int(rid_str)] = len(race_data.get("runners", []))

        # Races explicitly skipped during precompute (entries only, no jockeys)
        cached_entries_only = set(cache.get("skipped_race_ids", [])) if cache else set()

        races = []
        for r in rows:
            rid = r[0]

            # Skip races that are entries-only (no jockeys declared)
            if rid in cached_entries_only:
                continue

            # Use cached active runner count if available, otherwise DB value
            num_runners = cached_runner_counts.get(rid, r[11] or 0)
            cached_meta = cached_races.get(str(rid), {}).get("meta", {})
            top_gap = cached_meta.get("top_gap")
            full_field = cached_meta.get("full_field")

            races.append(RaceInfo(
                race_id=rid,
                meeting_date=str(r[1]),
                course=r[2] or "",
                race_number=r[3],
                race_name=r[4],
                race_time=r[5],
                distance_furlongs=r[6],
                race_type=r[7],
                race_class=r[8],
                going=r[9],
                surface=r[10],
                num_runners=num_runners,
                region_code=r[12],
                top_gap=top_gap,
                full_field=full_field,
            ))

        return RaceCard(
            races=races,
            meeting_date=query_date,
            course=course,
            total_races=len(races),
            timestamp=datetime.now().isoformat(),
        )
    finally:
        con.close()


@app.get("/api/races/dates")
def get_available_dates(limit: int = Query(30, ge=1, le=365)):
    con = _get_con()
    try:
        rows = con.execute("""
            SELECT m.meeting_date,
                   COUNT(DISTINCT m.meeting_id) AS meetings,
                   COUNT(DISTINCT ra.race_id) AS races
            FROM meetings m
            JOIN races ra ON ra.meeting_id = m.meeting_id
            WHERE m.meeting_date >= current_date
            GROUP BY m.meeting_date
            ORDER BY m.meeting_date ASC
            LIMIT ?
        """, [limit]).fetchall()

        dates = [
            DateInfo(date=str(r[0]), meetings=r[1], races=r[2])
            for r in rows
        ]

        return {
            "dates": [d.model_dump() for d in dates],
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 13D Race Predictions
# ---------------------------------------------------------------------------

@app.get("/api/race/{race_id}/thirteen_d", response_model=ThirteenDRace)
def get_thirteen_d(race_id: int):
    con = _get_con()
    try:
        race_row = con.execute("""
            SELECT ra.race_id, m.meeting_date, m.course,
                   ra.race_number, ra.race_name,
                   COALESCE(ra.race_time, strftime(ra.off_dt, '%H:%M')) as race_time,
                   ra.distance_furlongs, ra.race_type, ra.race_class,
                   ra.going_description, ra.surface, ra.num_runners,
                   ra.region_code, ra.api_race_id
            FROM races ra
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE ra.race_id = ?
        """, [race_id]).fetchone()

        if not race_row:
            raise HTTPException(status_code=404, detail=f"Race {race_id} not found")

        api_race_id = race_row[13]
        meeting_date = str(race_row[1])

        race_info = RaceInfo(
            race_id=race_row[0],
            meeting_date=meeting_date,
            course=race_row[2] or "",
            race_number=race_row[3],
            race_name=race_row[4],
            race_time=race_row[5],
            distance_furlongs=race_row[6],
            race_type=race_row[7],
            race_class=race_row[8],
            going=race_row[9],
            surface=race_row[10],
            num_runners=race_row[11] or 0,
            region_code=race_row[12],
        )

        # Try pre-computed cache first (no heavy ML on server)
        cache = _load_cache(meeting_date)
        cached_race = cache.get("races", {}).get(str(race_id)) if cache else None

        if cached_race:
            cached_runners = cached_race.get("runners", [])
            runner_api_ids = cached_race.get("runner_api_ids", {})

            # Override num_runners with actual active runner count from cache
            race_info.num_runners = len(cached_runners)
            cached_meta = cached_race.get("meta", {})
            if "top_gap" in cached_meta:
                race_info.top_gap = cached_meta["top_gap"]
            if "full_field" in cached_meta:
                race_info.full_field = cached_meta["full_field"]

            jt_rows = con.execute("""
                SELECT horse_name, jockey_name, trainer_name, api_horse_id, silk_url
                FROM results WHERE race_id = ?
            """, [race_id]).fetchall()
            jt_map = {r[0]: (r[1], r[2], r[4]) for r in jt_rows}

            runners = []
            for cr in cached_runners:
                dims = cr.get("dimensions", {})
                dim_list = []
                if isinstance(dims, dict):
                    dim_list = [DimensionScore(name=k, score=v.get("score", 5), label=v.get("label", ""),
                                               detail=v.get("detail", "")) for k, v in dims.items()]
                elif isinstance(dims, list):
                    dim_list = [DimensionScore(**d) for d in dims]

                jt = jt_map.get(cr["horse_name"], (None, None, None))
                runners.append(RunnerPrediction(
                    horse_name=cr["horse_name"],
                    draw=None,
                    jockey=jt[0],
                    trainer=jt[1],
                    silk_url=jt[2] if len(jt) > 2 else None,
                    age=None,
                    weight_lbs=None,
                    official_rating=cr.get("official_rating"),
                    win_prob=cr.get("win_prob", 0),
                    place_prob=cr.get("place_prob", 0),
                    win_rank=0,
                    place_rank=0,
                    fair_odds=round(1.0 / max(cr.get("win_prob", 0.001), 0.001), 2),
                    dimensions=dim_list,
                    positive_drivers=[DriverItem(**d) for d in cr.get("positive_drivers", [])],
                    negative_drivers=[DriverItem(**d) for d in cr.get("negative_drivers", [])],
                ))

            runners.sort(key=lambda r: r.win_prob, reverse=True)
            for i, r in enumerate(runners):
                r.win_rank = i + 1
            runners_by_place = sorted(runners, key=lambda r: r.place_prob, reverse=True)
            for i, r in enumerate(runners_by_place):
                r.place_rank = i + 1

            _attach_odds(race_id, runners, api_race_id, runner_api_ids)

            return ThirteenDRace(
                race_info=race_info,
                runners=runners,
                timestamp=cache.get("generated_at", datetime.now().isoformat()),
                predictions_status="cached_model",
            )

        # No cache — lightweight fallback using ratings only (no ML models)
        features_df = build_features_for_races(con, [race_id])

        if features_df.empty:
            return ThirteenDRace(
                race_info=race_info, runners=[], timestamp=datetime.now().isoformat(),
                predictions_status="no_data",
            )

        dim_scores = compute_thirteen_d_scores(features_df, race_id)

        win_probs = {}
        place_probs = {}
        ratings = features_df[["horse_name", "official_rating"]].copy()
        ratings["official_rating"] = ratings["official_rating"].fillna(50)
        total = ratings["official_rating"].sum()
        if total > 0:
            for _, row in ratings.iterrows():
                name = row["horse_name"]
                prob = float(row["official_rating"]) / total
                win_probs[name] = prob
                place_probs[name] = min(prob * 2.5, 0.95)

        runners = []
        for _, row in features_df.iterrows():
            name = row.get("horse_name", "")
            wp = win_probs.get(name, 1.0 / max(race_info.num_runners, 1))
            pp = place_probs.get(name, min(wp * 2.5, 0.95))

            dims = dim_scores.get(name, [])
            pos_drivers, neg_drivers = compute_drivers(features_df, name, race_id)

            runners.append(RunnerPrediction(
                horse_name=name,
                draw=int(row["draw"]) if row.get("draw") and not pd.isna(row.get("draw")) else None,
                jockey=None, trainer=None,
                age=int(row["age"]) if row.get("age") and not pd.isna(row.get("age")) else None,
                weight_lbs=int(row["weight_lbs"]) if row.get("weight_lbs") and not pd.isna(row.get("weight_lbs")) else None,
                official_rating=int(row["official_rating"]) if row.get("official_rating") and not pd.isna(row.get("official_rating")) else None,
                win_prob=round(wp, 4), place_prob=round(pp, 4),
                win_rank=0, place_rank=0,
                fair_odds=round(1.0 / max(wp, 0.001), 2),
                dimensions=[DimensionScore(**d) for d in dims],
                positive_drivers=[DriverItem(**d) for d in pos_drivers],
                negative_drivers=[DriverItem(**d) for d in neg_drivers],
            ))

        runners.sort(key=lambda r: r.win_prob, reverse=True)
        for i, r in enumerate(runners):
            r.win_rank = i + 1
        runners_by_place = sorted(runners, key=lambda r: r.place_prob, reverse=True)
        for i, r in enumerate(runners_by_place):
            r.place_rank = i + 1

        runner_api_ids = {}
        try:
            jt_rows = con.execute("""
                SELECT horse_name, jockey_name, trainer_name, api_horse_id, silk_url
                FROM results WHERE race_id = ?
            """, [race_id]).fetchall()
            jt_map = {r[0]: (r[1], r[2], r[4]) for r in jt_rows}
            for r in jt_rows:
                if r[3]:
                    runner_api_ids[r[0]] = r[3]
            for r in runners:
                jt = jt_map.get(r.horse_name)
                if jt:
                    r.jockey = jt[0]
                    r.trainer = jt[1]
                    r.silk_url = jt[2]
        except Exception:
            pass

        _attach_odds(race_id, runners, api_race_id, runner_api_ids)
        model_type = "fallback"
    finally:
        con.close()

    _save_predictions(race_id, runners, model_type)

    return ThirteenDRace(
        race_info=race_info, runners=runners,
        timestamp=datetime.now().isoformat(),
        predictions_status=model_type,
    )


_odds_cache: dict = {}  # race_id -> (timestamp, {horse_name: back_odds})
_ODDS_CACHE_TTL = 120  # 2 minutes


def _compute_value_flag(fair_odds: float, back_odds: float) -> str:
    """Compare model price vs bookmaker price."""
    if back_odds > fair_odds * 1.1:
        return "VALUE"
    elif back_odds < fair_odds * 0.9:
        return "SHORT"
    return "FAIR"


def _best_decimal_from_odds(bookmaker_odds: list) -> float | None:
    """Pick the best (highest) decimal price from a list of bookmaker odds."""
    best = None
    for bm in bookmaker_odds:
        try:
            dec = float(bm.get("decimal", 0))
            if dec > 1.0 and (best is None or dec > best):
                best = dec
        except (ValueError, TypeError):
            continue
    return best


def _attach_odds(race_id: int, runners: list,
                  api_race_id: str = None,
                  runner_api_ids: dict = None):
    """Attach live bookmaker odds to runners -- fetched directly from
    The Racing API into an in-memory cache.  No database reads or writes."""
    now_ts = datetime.now().timestamp()

    cached = _odds_cache.get(race_id)
    if cached and (now_ts - cached[0]) < _ODDS_CACHE_TTL:
        odds_map = cached[1]
    else:
        odds_map = {}
        if api_race_id and runner_api_ids:
            try:
                from ..scrapers.racing_api import RacingAPIClient
                client = RacingAPIClient()
                for horse_name, api_horse_id in runner_api_ids.items():
                    try:
                        data = client.get_odds_runner(api_race_id, api_horse_id)
                        dec = _best_decimal_from_odds(data.get("odds", []))
                        if dec and dec > 1.0:
                            odds_map[horse_name] = dec
                    except Exception as e:
                        logger.debug(f"Odds API call for {horse_name}: {e}")
                        continue
                logger.info(f"Fetched live odds for {len(odds_map)}/{len(runner_api_ids)} runners in race {race_id}")
            except Exception as e:
                logger.warning(f"Failed to fetch live odds for race {race_id}: {e}")

        _odds_cache[race_id] = (now_ts, odds_map)

    for r in runners:
        back = odds_map.get(r.horse_name)
        if back and back > 0:
            r.back_odds = round(back, 2)
            r.value_flag = _compute_value_flag(r.fair_odds, back)
            try:
                cached_entry = _odds_cache.get(race_id)
                if cached_entry:
                    r.odds_updated_at = datetime.fromtimestamp(
                        cached_entry[0]
                    ).strftime("%H:%M")
            except Exception:
                pass


def _save_predictions(race_id: int, runners: list, model_type: str):
    """Persist predictions to predictions.duckdb (separate DB, no conflicts)."""
    try:
        con = get_pred_connection()
        try:
            for r in runners:
                pid = con.execute("SELECT nextval('seq_pred_id')").fetchone()[0]
                con.execute("""
                    INSERT INTO predictions (prediction_id, race_id, horse_name,
                        win_prob, place_prob, win_rank, place_rank,
                        model_version, model_type, predicted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'v1', ?, current_timestamp)
                    ON CONFLICT (race_id, horse_name, model_version) DO UPDATE SET
                        win_prob = excluded.win_prob,
                        place_prob = excluded.place_prob,
                        win_rank = excluded.win_rank,
                        place_rank = excluded.place_rank,
                        model_type = excluded.model_type,
                        predicted_at = excluded.predicted_at
                """, [pid, race_id, r.horse_name, r.win_prob, r.place_prob,
                      r.win_rank, r.place_rank, model_type])
            logger.info(f"Saved {len(runners)} predictions for race {race_id}")
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Failed to save predictions for race {race_id}: {e}")


# ---------------------------------------------------------------------------
# Performance / Feedback Loop
# ---------------------------------------------------------------------------

@app.get("/api/performance")
def get_performance(days: int = Query(30, ge=1, le=365)):
    """Full performance summary: hit rate, ROI, calibration, daily breakdown."""
    from ..reconcile import reconcile_predictions
    return reconcile_predictions(days)


@app.get("/api/performance/summary")
def get_performance_summary():
    """Quick prediction counts -- how many saved, how many reconciled."""
    from ..reconcile import get_prediction_count
    return get_prediction_count()


@app.post("/api/reconcile")
def trigger_reconcile():
    """Copy actual results from horse.duckdb into predictions.duckdb and return stats."""
    from ..reconcile import copy_actual_results, get_prediction_count
    copied = copy_actual_results()
    counts = get_prediction_count()
    counts["results_copied"] = copied
    return counts


# ---------------------------------------------------------------------------
# Scrape racecards on demand
# ---------------------------------------------------------------------------

_scrape_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}
_scrape_lock = threading.Lock()


def _run_scrape():
    """Background worker: fetch upcoming racecards from The Racing API."""
    try:
        from ..scrapers.racing_api import RacingAPIClient, fetch_upcoming_racecards
        client = RacingAPIClient()
        fetch_upcoming_racecards(client)
        with _scrape_lock:
            _scrape_state["running"] = False
            _scrape_state["finished_at"] = datetime.now().isoformat()
            _scrape_state["result"] = "success"
            _scrape_state["error"] = None
        logger.info("Racecard scrape completed successfully")
    except Exception as e:
        logger.error(f"Racecard scrape failed: {e}")
        with _scrape_lock:
            _scrape_state["running"] = False
            _scrape_state["finished_at"] = datetime.now().isoformat()
            _scrape_state["result"] = "error"
            _scrape_state["error"] = str(e)


@app.post("/api/scrape-racecards")
def scrape_racecards(user: dict = Depends(require_admin)):
    """Trigger a background scrape of upcoming racecards. Admin only."""
    with _scrape_lock:
        if _scrape_state["running"]:
            return {"status": "already_running", "started_at": _scrape_state["started_at"]}
        _scrape_state["running"] = True
        _scrape_state["started_at"] = datetime.now().isoformat()
        _scrape_state["finished_at"] = None
        _scrape_state["result"] = None
        _scrape_state["error"] = None

    thread = threading.Thread(target=_run_scrape, daemon=True)
    thread.start()
    return {"status": "started", "started_at": _scrape_state["started_at"]}


@app.get("/api/scrape-status")
def scrape_status():
    """Check the current status of a racecard scrape."""
    with _scrape_lock:
        return dict(_scrape_state)


# ---------------------------------------------------------------------------
# Refresh odds on demand
# ---------------------------------------------------------------------------

_odds_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "races_processed": 0,
    "error": None,
}
_odds_lock = threading.Lock()


def _run_odds_refresh():
    """Background worker: fetch odds for today/tomorrow races directly from
    The Racing API into the in-memory cache.  No DB writes at all."""
    try:
        from ..scrapers.racing_api import RacingAPIClient
        from datetime import date, timedelta

        # Read race list from DB (read-only, quick)
        con = get_connection(read_only=True)
        try:
            today = date.today()
            tomorrow = today + timedelta(days=1)
            rows = con.execute("""
                SELECT ra.race_id, ra.api_race_id
                FROM races ra
                JOIN meetings m ON ra.meeting_id = m.meeting_id
                WHERE m.meeting_date IN (?, ?)
                  AND ra.api_race_id IS NOT NULL
                ORDER BY ra.off_dt
            """, [str(today), str(tomorrow)]).fetchall()

            # Grab runner api_horse_ids for each race
            race_runners = {}
            for race_id, api_race_id in rows:
                runner_rows = con.execute("""
                    SELECT horse_name, api_horse_id
                    FROM results
                    WHERE race_id = ? AND api_horse_id IS NOT NULL
                """, [race_id]).fetchall()
                race_runners[race_id] = {
                    "api_race_id": api_race_id,
                    "runners": {r[0]: r[1] for r in runner_rows},
                }
        finally:
            con.close()

        logger.info(f"Odds refresh: fetching from API for {len(race_runners)} races")
        client = RacingAPIClient()
        count = 0

        for race_id, info in race_runners.items():
            api_race_id = info["api_race_id"]
            runner_ids = info["runners"]
            if not runner_ids:
                continue

            odds_map = {}
            for horse_name, api_horse_id in runner_ids.items():
                try:
                    data = client.get_odds_runner(api_race_id, api_horse_id)
                    dec = _best_decimal_from_odds(data.get("odds", []))
                    if dec and dec > 1.0:
                        odds_map[horse_name] = dec
                except Exception as e:
                    logger.debug(f"Odds refresh for {horse_name}: {e}")
                    continue

            _odds_cache[race_id] = (datetime.now().timestamp(), odds_map)
            count += 1

        # Persist odds to market_data table for historical analysis
        try:
            persist_con = get_connection(read_only=False)
            try:
                for race_id, (ts, odds_map) in _odds_cache.items():
                    for horse_name, dec_odds in odds_map.items():
                        try:
                            existing = persist_con.execute(
                                "SELECT market_id FROM market_data WHERE race_id = ? AND horse_name = ?",
                                [race_id, horse_name]
                            ).fetchone()
                            if existing:
                                persist_con.execute("""
                                    UPDATE market_data SET back_odds = ?, captured_at = current_timestamp
                                    WHERE market_id = ?
                                """, [dec_odds, existing[0]])
                            else:
                                mid = persist_con.execute("SELECT nextval('seq_market_id')").fetchone()[0]
                                persist_con.execute("""
                                    INSERT INTO market_data (market_id, race_id, horse_name, back_odds, captured_at)
                                    VALUES (?, ?, ?, ?, current_timestamp)
                                """, [mid, race_id, horse_name, dec_odds])
                        except Exception:
                            pass
            finally:
                persist_con.close()
            logger.info(f"Odds persisted to market_data table")
        except Exception as e:
            logger.debug(f"Odds persistence failed: {e}")

        with _odds_lock:
            _odds_state["running"] = False
            _odds_state["finished_at"] = datetime.now().isoformat()
            _odds_state["result"] = "success"
            _odds_state["races_processed"] = count
            _odds_state["error"] = None
        logger.info(f"Odds refresh completed: {count} races")
    except Exception as e:
        logger.error(f"Odds refresh failed: {e}")
        with _odds_lock:
            _odds_state["running"] = False
            _odds_state["finished_at"] = datetime.now().isoformat()
            _odds_state["result"] = "error"
            _odds_state["error"] = str(e)


@app.post("/api/refresh-odds")
def refresh_odds(user: dict = Depends(require_admin)):
    """Trigger a background refresh of live odds for today/tomorrow. Admin only."""
    with _odds_lock:
        if _odds_state["running"]:
            return {"status": "already_running", "started_at": _odds_state["started_at"]}
        _odds_state["running"] = True
        _odds_state["started_at"] = datetime.now().isoformat()
        _odds_state["finished_at"] = None
        _odds_state["result"] = None
        _odds_state["races_processed"] = 0
        _odds_state["error"] = None

    thread = threading.Thread(target=_run_odds_refresh, daemon=True)
    thread.start()
    return {"status": "started", "started_at": _odds_state["started_at"]}


@app.get("/api/odds-status")
def odds_status():
    """Check the current status of an odds refresh."""
    with _odds_lock:
        return dict(_odds_state)


@app.post("/api/odds/clear-cache")
def clear_odds_cache():
    """Clear the in-memory odds cache so next race load fetches fresh."""
    _odds_cache.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Best Bets - Top 3 E/W picks across all meetings for a date
# ---------------------------------------------------------------------------

@app.get("/api/best-bets", response_model=BestBetsResponse)
def get_best_bets(date: Optional[str] = Query(None, description="YYYY-MM-DD")):
    """Top picks — served from pre-computed cache (generated locally)."""
    from datetime import date as date_type

    target_date = date or str(date_type.today())

    cache = _load_cache(target_date)
    if cache and "best_bets" in cache:
        bb = cache["best_bets"]
        picks = [BestBetRunner(**p) for p in bb.get("picks", [])]
        return BestBetsResponse(
            date=target_date,
            picks=picks,
            total_races_scanned=bb.get("total_races_scanned", 0),
            timestamp=bb.get("timestamp", datetime.now().isoformat()),
        )

    return BestBetsResponse(
        date=target_date, picks=[], total_races_scanned=0,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Value betting & model monitoring endpoints
# ---------------------------------------------------------------------------

@app.get("/api/value-bets/{race_id}")
def get_value_bets(race_id: int):
    """Value bets from pre-computed cache — no ML on server."""
    try:
        con = _get_con()
        try:
            row = con.execute(
                "SELECT m.meeting_date FROM races ra JOIN meetings m ON ra.meeting_id = m.meeting_id WHERE ra.race_id = ?",
                [race_id],
            ).fetchone()
        finally:
            con.close()

        if not row:
            return {"error": "Race not found"}

        cache = _load_cache(str(row[0]))
        cached_race = cache.get("races", {}).get(str(race_id)) if cache else None

        if not cached_race:
            return {"bets": [], "message": "No pre-computed predictions. Run PUSH_PREDICTIONS.bat locally."}

        bets = []
        for r in cached_race.get("runners", []):
            wp = r.get("win_prob", 0)
            if wp > 0.05:
                fair = 1.0 / max(wp, 0.001)
                bets.append({
                    "horse_name": r["horse_name"],
                    "win_prob": round(wp, 4),
                    "back_odds": 0,
                    "fair_odds": round(fair, 2),
                    "ev": 0,
                    "value_score": round(wp, 4),
                    "kelly": 0,
                    "recommended_stake_pct": 0,
                })

        bets.sort(key=lambda b: b["win_prob"], reverse=True)
        return {"race_id": race_id, "bets": bets[:5]}

    except Exception as e:
        logger.warning(f"Value bets failed: {e}")
        return {"error": str(e)}


@app.get("/api/model-monitoring")
def model_monitoring():
    """Get model monitoring history and drift detection."""
    try:
        from ..online import load_monitoring_history
        history = load_monitoring_history()
        return {"history": history[-30:], "total_snapshots": len(history)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/execution-log")
def get_execution_log():
    """Get the betting execution log."""
    try:
        from ..execution import EXECUTION_LOG_PATH
        import json as _json
        if EXECUTION_LOG_PATH.exists():
            data = _json.loads(EXECUTION_LOG_PATH.read_text(encoding="utf-8"))
            total_pnl = sum(b.get("pnl", 0) or 0 for b in data)
            return {
                "bets": data[-50:],
                "total_bets": len(data),
                "total_pnl": round(total_pnl, 2),
            }
        return {"bets": [], "total_bets": 0, "total_pnl": 0}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/pace-analysis/{race_id}")
def get_pace_analysis(race_id: int):
    """Pace analysis — lightweight, no ML models needed."""
    try:
        con = _get_con()
        try:
            row = con.execute(
                "SELECT m.meeting_date FROM races ra JOIN meetings m ON ra.meeting_id = m.meeting_id WHERE ra.race_id = ?",
                [race_id],
            ).fetchone()

            if not row:
                return {"error": "Race not found"}

            cache = _load_cache(str(row[0]))
            cached_race = cache.get("races", {}).get(str(race_id)) if cache else None

            if cached_race:
                runners = [{"horse_name": r["horse_name"]} for r in cached_race.get("runners", [])]
                return {
                    "race_id": race_id,
                    "analysis": {"shape": "pre-computed", "narrative": "Pace data available in race detail view."},
                    "runners": runners,
                }

            return {"race_id": race_id, "analysis": {"shape": "unknown", "narrative": "No pace data available"}, "runners": []}
        finally:
            con.close()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/staking-calc")
def staking_calculator(
    prob: float = Query(..., description="Model probability"),
    odds: float = Query(..., description="Decimal odds"),
    bank: float = Query(1000, description="Current bank"),
):
    """Calculate optimal stake for a bet."""
    try:
        from ..staking import calculate_stake
        rec = calculate_stake(prob, odds, bank)
        return {
            "model_prob": prob,
            "decimal_odds": odds,
            "kelly_full": rec.kelly_full,
            "kelly_fractional": rec.kelly_fractional,
            "stake_amount": rec.stake_amount,
            "stake_pct": rec.stake_pct,
            "ev": rec.ev,
            "value_score": rec.value_score,
            "confidence_tier": rec.confidence_tier,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Static files (production build)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
_INDEX_HTML = _FRONTEND_DIST / "index.html"

if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        from fastapi.responses import FileResponse, Response
        if full_path and (asset := _FRONTEND_DIST / full_path).is_file():
            return FileResponse(str(asset))
        content = _INDEX_HTML.read_bytes()
        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


def start():
    """Run the horse racing API server."""
    import uvicorn
    logger.info(f"Starting Horse Racing 13D API on port {API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")


if __name__ == "__main__":
    start()
