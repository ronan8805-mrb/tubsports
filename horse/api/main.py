"""
FastAPI backend for the Horse Racing 13D Prediction System.
Runs on port 8002, completely separate from greyhound (port 8000).
"""

import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import API_PORT, DB_PATH
from ..db import get_connection, get_db_stats
from ..features import build_features_for_races, get_feature_columns
from ..prediction_db import get_pred_connection
from .schemas import (
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_con():
    """Get a read-only DB connection."""
    return get_connection(read_only=True)


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

        races = []
        for r in rows:
            races.append(RaceInfo(
                race_id=r[0],
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
                num_runners=r[11] or 0,
                region_code=r[12],
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
            GROUP BY m.meeting_date
            ORDER BY m.meeting_date DESC
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

        api_race_id = race_row[13]  # from ra.api_race_id

        race_info = RaceInfo(
            race_id=race_row[0],
            meeting_date=str(race_row[1]),
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

        features_df = build_features_for_races(con, [race_id])

        if features_df.empty:
            return ThirteenDRace(
                race_info=race_info,
                runners=[],
                timestamp=datetime.now().isoformat(),
                predictions_status="no_data",
            )

        dim_scores = compute_thirteen_d_scores(features_df, race_id)

        # Try loading trained model predictions
        win_probs = {}
        place_probs = {}
        try:
            from ..models import load_model
            win_model = load_model("win")
            place_model = load_model("place")

            feature_names = win_model["feature_names"]
            import xgboost as xgb

            X = features_df.reindex(columns=feature_names).copy()
            import pandas as pd
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            lgb_win = win_model["lgb_model"].predict(X)
            xgb_win = win_model["xgb_model"].predict(
                xgb.DMatrix(X, feature_names=feature_names)
            )
            ens_win = 0.5 * lgb_win + 0.5 * xgb_win

            lgb_place = place_model["lgb_model"].predict(X)
            xgb_place = place_model["xgb_model"].predict(
                xgb.DMatrix(X, feature_names=feature_names)
            )
            ens_place = 0.5 * lgb_place + 0.5 * xgb_place

            for i, (_, row) in enumerate(features_df.iterrows()):
                name = row.get("horse_name", "")
                if name:
                    win_probs[name] = float(ens_win[i])
                    place_probs[name] = float(ens_place[i])
        except Exception as e:
            logger.info(f"Model predictions not available: {e}")

        # If no model, use rating-based fallback
        if not win_probs:
            ratings = features_df[["horse_name", "official_rating"]].copy()
            ratings["official_rating"] = ratings["official_rating"].fillna(50)
            total = ratings["official_rating"].sum()
            if total > 0:
                for _, row in ratings.iterrows():
                    name = row["horse_name"]
                    prob = float(row["official_rating"]) / total
                    win_probs[name] = prob
                    place_probs[name] = min(prob * 2.5, 0.95)

        # Build runner predictions
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
                jockey=None,
                trainer=None,
                age=int(row["age"]) if row.get("age") and not pd.isna(row.get("age")) else None,
                weight_lbs=int(row["weight_lbs"]) if row.get("weight_lbs") and not pd.isna(row.get("weight_lbs")) else None,
                official_rating=int(row["official_rating"]) if row.get("official_rating") and not pd.isna(row.get("official_rating")) else None,
                win_prob=round(wp, 4),
                place_prob=round(pp, 4),
                win_rank=0,
                place_rank=0,
                fair_odds=round(1.0 / max(wp, 0.001), 2),
                dimensions=[DimensionScore(**d) for d in dims],
                positive_drivers=[DriverItem(**d) for d in pos_drivers],
                negative_drivers=[DriverItem(**d) for d in neg_drivers],
            ))

        # Assign ranks
        runners.sort(key=lambda r: r.win_prob, reverse=True)
        for i, r in enumerate(runners):
            r.win_rank = i + 1

        runners_by_place = sorted(runners, key=lambda r: r.place_prob, reverse=True)
        for i, r in enumerate(runners_by_place):
            r.place_rank = i + 1

        # Get jockey/trainer names + api_horse_id from results table
        runner_api_ids = {}  # horse_name -> api_horse_id
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

        # Attach live odds directly from Racing API (no DB)
        _attach_odds(race_id, runners, api_race_id, runner_api_ids)

        model_type = "model" if win_probs else "fallback"
    finally:
        con.close()

    # Persist predictions AFTER read connection is closed (avoids DuckDB conflict)
    _save_predictions(race_id, runners, model_type)

    return ThirteenDRace(
        race_info=race_info,
        runners=runners,
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
def scrape_racecards():
    """Trigger a background scrape of upcoming racecards."""
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

            # Store in memory cache only -- no DB writes
            _odds_cache[race_id] = (datetime.now().timestamp(), odds_map)
            count += 1

        with _odds_lock:
            _odds_state["running"] = False
            _odds_state["finished_at"] = datetime.now().isoformat()
            _odds_state["result"] = "success"
            _odds_state["races_processed"] = count
            _odds_state["error"] = None
        logger.info(f"Odds refresh completed: {count} races (in-memory only)")
    except Exception as e:
        logger.error(f"Odds refresh failed: {e}")
        with _odds_lock:
            _odds_state["running"] = False
            _odds_state["finished_at"] = datetime.now().isoformat()
            _odds_state["result"] = "error"
            _odds_state["error"] = str(e)


@app.post("/api/refresh-odds")
def refresh_odds():
    """Trigger a background refresh of live odds for today/tomorrow."""
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
# Static files (production build)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")


def start():
    """Run the horse racing API server."""
    import uvicorn
    logger.info(f"Starting Horse Racing 13D API on port {API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")


if __name__ == "__main__":
    start()
