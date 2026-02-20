"""
The Racing API client and backfill engine.

PRO plan: full access to results, racecards, horses, jockeys, trainers,
sires, dams, damsires, odds, courses, regions.

API constraint: /results endpoint limited to 12 months in the past.
Per-horse /horses/{id}/results has no date limit (career history).

Strategy:
  Phase 2 - Fetch all results from the last 12 months (month-by-month chunks)
  Phase 4 - For every horse found, fetch their FULL career history

Rate limit: 3 req/s (API allows 5, we stay conservative).
Auth: HTTP Basic.
Pagination: limit/skip offset.
"""

import json
import logging
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional

import requests
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    wait_fixed, retry_if_exception_type
)

from ..config import (
    RACING_API_BASE, RACING_API_USERNAME, RACING_API_PASSWORD,
    RACING_API_RATE_LIMIT,
)
from ..db import (
    get_connection, init_database,
    upsert_meeting, upsert_race_by_api_id,
    upsert_horse_by_api_id, upsert_jockey_by_api_id,
    upsert_trainer_by_api_id, insert_result_from_api,
    get_backfill_status, set_backfill_started, set_backfill_completed,
    get_db_stats,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Rate limiter
# ===================================================================

class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, calls_per_second: int = 3):
        self._interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()


# ===================================================================
# API Client
# ===================================================================

class RacingAPIClient:
    """HTTP client for The Racing API with auth, rate limiting, and retry."""

    def __init__(self):
        if not RACING_API_USERNAME or not RACING_API_PASSWORD:
            raise ValueError(
                "Racing API credentials not set. "
                "Check horse/.env has RACING_API_USERNAME and RACING_API_PASSWORD."
            )
        self.session = requests.Session()
        self.session.auth = (RACING_API_USERNAME, RACING_API_PASSWORD)
        self.session.headers.update({
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        })
        self.base = RACING_API_BASE.rstrip("/")
        self.limiter = RateLimiter(RACING_API_RATE_LIMIT)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Core GET with rate limiting and auto-retry on 429/5xx."""
        max_retries = 5
        for attempt in range(max_retries):
            self.limiter.wait()
            url = f"{self.base}{endpoint}"
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 60)
                    logger.warning(f"Connection error on {endpoint}: {e}. "
                                   f"Retry {attempt+1}/{max_retries} in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                raise

            if resp.status_code == 429:
                wait_time = 65
                logger.warning(f"Rate limited (429) on {endpoint}. "
                               f"Waiting {wait_time}s... (attempt {attempt+1})")
                time.sleep(wait_time)
                continue

            if resp.status_code >= 500 and attempt < max_retries - 1:
                wait_time = min(2 ** (attempt + 1), 60)
                logger.warning(f"Server error {resp.status_code} on {endpoint}. "
                               f"Retry {attempt+1}/{max_retries} in {wait_time}s")
                time.sleep(wait_time)
                continue

            if resp.status_code == 401:
                raise PermissionError("Racing API 401 Unauthorized. Check credentials.")

            if resp.status_code == 422:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = resp.text[:200]
                raise ValueError(f"API 422: {detail} | endpoint={endpoint} params={params}")

            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"Max retries ({max_retries}) exceeded for {endpoint}")

    # --- Results ---
    def get_results(self, start_date: str, end_date: str,
                    limit: int = 50, skip: int = 0,
                    region: str = None) -> dict:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "skip": skip,
        }
        if region:
            params["region"] = region
        return self._get("/results", params)

    def get_result(self, race_id: str) -> dict:
        return self._get(f"/results/{race_id}")

    # --- Racecards ---
    def get_racecards_pro(self, date: str = None) -> dict:
        params = {"date": date} if date else None
        return self._get("/racecards/pro", params)

    # --- Odds ---
    def get_odds_runner(self, race_id: str, horse_id: str) -> dict:
        """Get live odds for a single runner in a race (PRO plan).
        Returns: {race_id, horse_id, horse, odds: [{bookmaker, decimal, fractional, ...}]}
        """
        return self._get(f"/odds/{race_id}/{horse_id}")

    # --- Horses ---
    def get_horse(self, horse_id: str) -> dict:
        return self._get(f"/horses/{horse_id}")

    def get_horse_results(self, horse_id: str, start_date: str = None,
                          end_date: str = None,
                          limit: int = 50, skip: int = 0) -> dict:
        params = {"limit": limit, "skip": skip}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get(f"/horses/{horse_id}/results", params)

    def get_horse_distance_time(self, horse_id: str) -> dict:
        return self._get(f"/horses/{horse_id}/analysis/distance-times")

    # --- Jockeys ---
    def get_jockey_results(self, jockey_id: str,
                           limit: int = 50, skip: int = 0) -> dict:
        return self._get(f"/jockeys/{jockey_id}/results",
                         {"limit": limit, "skip": skip})

    def get_jockey_course_analysis(self, jockey_id: str) -> dict:
        return self._get(f"/jockeys/{jockey_id}/analysis/courses")

    def get_jockey_distance_analysis(self, jockey_id: str) -> dict:
        return self._get(f"/jockeys/{jockey_id}/analysis/distances")

    def get_jockey_trainer_analysis(self, jockey_id: str) -> dict:
        return self._get(f"/jockeys/{jockey_id}/analysis/trainers")

    # --- Trainers ---
    def get_trainer_results(self, trainer_id: str,
                            limit: int = 50, skip: int = 0) -> dict:
        return self._get(f"/trainers/{trainer_id}/results",
                         {"limit": limit, "skip": skip})

    def get_trainer_course_analysis(self, trainer_id: str) -> dict:
        return self._get(f"/trainers/{trainer_id}/analysis/courses")

    def get_trainer_jockey_analysis(self, trainer_id: str) -> dict:
        return self._get(f"/trainers/{trainer_id}/analysis/jockeys")

    def get_trainer_age_analysis(self, trainer_id: str) -> dict:
        return self._get(f"/trainers/{trainer_id}/analysis/horse-age")

    # --- Sires ---
    def get_sire_results(self, sire_id: str,
                         limit: int = 50, skip: int = 0) -> dict:
        return self._get(f"/sires/{sire_id}/results",
                         {"limit": limit, "skip": skip})

    def get_sire_class_analysis(self, sire_id: str) -> dict:
        return self._get(f"/sires/{sire_id}/analysis/classes")

    def get_sire_distance_analysis(self, sire_id: str) -> dict:
        return self._get(f"/sires/{sire_id}/analysis/distances")

    # --- Dams ---
    def get_dam_results(self, dam_id: str,
                        limit: int = 50, skip: int = 0) -> dict:
        return self._get(f"/dams/{dam_id}/results",
                         {"limit": limit, "skip": skip})

    def get_dam_class_analysis(self, dam_id: str) -> dict:
        return self._get(f"/dams/{dam_id}/analysis/classes")

    def get_dam_distance_analysis(self, dam_id: str) -> dict:
        return self._get(f"/dams/{dam_id}/analysis/distances")

    # --- Damsires ---
    def get_damsire_results(self, damsire_id: str,
                            limit: int = 50, skip: int = 0) -> dict:
        return self._get(f"/damsires/{damsire_id}/results",
                         {"limit": limit, "skip": skip})

    # --- Courses ---
    def get_courses(self, region_codes: list = None) -> dict:
        params = {}
        if region_codes:
            params["region_codes"] = region_codes
        return self._get("/courses", params)

    def get_regions(self) -> dict:
        return self._get("/courses/regions")


# ===================================================================
# Backfill: results (Phase 2)
# ===================================================================

def _parse_off_dt(off_dt_str: str) -> Optional[datetime]:
    """Parse ISO datetime string from API."""
    if not off_dt_str:
        return None
    try:
        return datetime.fromisoformat(off_dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _parse_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_race_number(race_data: dict, index: int) -> int:
    """Extract a race number from the off time or use index."""
    off = race_data.get("off", "")
    if off:
        try:
            parts = off.split(":")
            return int(parts[0]) * 100 + int(parts[1])
        except (ValueError, IndexError):
            pass
    return index + 1


def _generate_month_chunks(months_back: int = 12):
    """Generate (start_date, end_date, label) tuples for month-by-month fetching.
    Goes from oldest allowed month to today."""
    today = date.today()
    chunks = []
    for m in range(months_back, 0, -1):
        chunk_start = today - timedelta(days=m * 30)
        chunk_end = chunk_start + timedelta(days=29)
        if chunk_end > today:
            chunk_end = today
        label = chunk_start.strftime("%Y-%m")
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat(), label))
    return chunks


def backfill_all_results(client: RacingAPIClient, months_back: int = 12):
    """
    Pull ALL results from the last 12 months, month by month.
    API constraint: start_date must be within 12 months of today.
    Fully resumable via backfill_log (keyed by month label).
    """
    init_database()
    con = get_connection()
    chunks = _generate_month_chunks(months_back)

    logger.info(f"Backfill plan: {len(chunks)} monthly chunks, "
                f"{chunks[0][2]} to {chunks[-1][2]}")

    try:
        for start_date, end_date, label in chunks:
            status = get_backfill_status(con, label, "results")
            if status == "completed":
                logger.info(f"Month {label}: already completed, skipping.")
                continue

            logger.info(f"{'='*60}")
            logger.info(f"BACKFILL MONTH {label} ({start_date} to {end_date})")
            logger.info(f"{'='*60}")

            set_backfill_started(con, label, "results")
            chunk_start_time = time.monotonic()

            total_races = 0
            total_runners = 0
            api_total = 0
            skip = 0
            limit = 50
            page = 0

            while True:
                try:
                    data = client.get_results(start_date, end_date,
                                              limit=limit, skip=skip)
                except ValueError as e:
                    if "422" in str(e):
                        logger.warning(f"Month {label}: API rejected date range: {e}")
                        set_backfill_completed(con, label, "results",
                                              status="skipped",
                                              error_message=str(e)[:500])
                        break
                    raise
                except Exception as e:
                    logger.error(f"Month {label} page {page} failed: {e}")
                    set_backfill_completed(con, label, "results",
                                          status="error",
                                          error_message=str(e)[:500])
                    break

                results_list = data.get("results", [])
                if page == 0:
                    api_total = data.get("total", 0)
                    logger.info(f"Month {label}: API reports {api_total} total results")

                if not results_list:
                    break

                for idx, race_data in enumerate(results_list):
                    try:
                        _ingest_race(con, race_data, idx)
                        runners = race_data.get("runners", [])
                        total_races += 1
                        total_runners += len(runners)
                    except Exception as e:
                        logger.error(
                            f"Race ingest error: {race_data.get('race_id', '?')}: {e}"
                        )

                page += 1
                skip += limit

                if page % 20 == 0:
                    logger.info(
                        f"  Month {label}: page {page}, "
                        f"{total_races} races, {total_runners} runners..."
                    )

                if len(results_list) < limit:
                    break

            elapsed = time.monotonic() - chunk_start_time
            mean_rpr = (total_runners / total_races) if total_races > 0 else 0

            validation_status = "completed"
            error_msg = None
            if total_races == 0 and api_total > 0:
                validation_status = "warning"
                error_msg = f"API reported {api_total} but 0 races ingested"
            elif mean_rpr > 0 and mean_rpr < 3:
                validation_status = "warning"
                error_msg = f"Low mean runners/race: {mean_rpr:.1f}"

            set_backfill_completed(con, label, "results",
                                  status=validation_status,
                                  api_total=api_total,
                                  db_race_count=total_races,
                                  db_result_count=total_runners,
                                  mean_runners_per_race=round(mean_rpr, 2),
                                  error_message=error_msg)

            logger.info(
                f"Month {label} DONE: {total_races} races, "
                f"{total_runners} runners, "
                f"mean {mean_rpr:.1f} runners/race, "
                f"{elapsed:.1f}s [{validation_status}]"
            )
    finally:
        con.close()


def fetch_recent_results(days_back: int = 3) -> dict:
    """Fetch results for the last N days and ingest into horse.duckdb.
    Lightweight alternative to full backfill -- designed to run before retrain."""
    from datetime import date, timedelta

    client = RacingAPIClient()
    init_database()
    con = get_connection()

    end = date.today()
    start = end - timedelta(days=days_back)

    logger.info(f"Fetching recent results: {start} to {end}")
    total_races = 0
    total_runners = 0

    try:
        skip = 0
        limit = 50
        while True:
            try:
                data = client.get_results(
                    start.isoformat(), end.isoformat(),
                    limit=limit, skip=skip,
                )
            except Exception as e:
                logger.warning(f"Recent results fetch failed: {e}")
                break

            results_list = data.get("results", [])
            if not results_list:
                break

            for idx, race_data in enumerate(results_list):
                try:
                    _ingest_race(con, race_data, idx)
                    total_races += 1
                    total_runners += len(race_data.get("runners", []))
                except Exception:
                    pass

            skip += limit
            if len(results_list) < limit:
                break

        logger.info(f"Recent results: {total_races} races, {total_runners} runners ingested")
    finally:
        con.close()

    return {"races": total_races, "runners": total_runners}


def _ingest_race(con, race_data: dict, index: int):
    """Ingest a single race and all its runners into the DB."""
    api_race_id = race_data.get("race_id", "")
    course = race_data.get("course", "Unknown")
    race_date_str = race_data.get("date", "")
    region = race_data.get("region", "")
    region_code = _region_to_code(region)
    country = _region_to_country(region)

    meeting_id = upsert_meeting(
        con, course, race_date_str, country=country,
        going=race_data.get("going"),
        region=region, region_code=region_code,
        api_course_id=race_data.get("course_id"),
    )

    race_number = _extract_race_number(race_data, index)
    off_dt = _parse_off_dt(race_data.get("off_dt"))
    dist_f = _parse_float(race_data.get("dist_f", "").replace("f", ""))
    dist_y = None
    try:
        dist_y = int(race_data.get("dist_y", 0)) or None
    except (ValueError, TypeError):
        pass
    dist_m = None
    try:
        dist_m = int(race_data.get("dist_m", 0)) or None
    except (ValueError, TypeError):
        pass

    runners = race_data.get("runners", [])
    raw_json = json.dumps(race_data)

    race_class = race_data.get("class", "")

    race_id = upsert_race_by_api_id(
        con, api_race_id, meeting_id,
        race_number=race_number,
        race_time=race_data.get("off"),
        off_dt=off_dt,
        race_name=race_data.get("race_name"),
        distance_furlongs=dist_f,
        distance_yards=dist_y,
        distance_metres=dist_m,
        race_type=race_data.get("type"),
        race_class=race_class,
        pattern=race_data.get("pattern"),
        rating_band=race_data.get("rating_band"),
        age_band=race_data.get("age_band"),
        sex_rest=race_data.get("sex_rest"),
        surface=race_data.get("surface"),
        going_description=race_data.get("going"),
        jumps_detail=race_data.get("jumps"),
        prize=None,
        num_runners=len(runners),
        non_runners_text=race_data.get("non_runners"),
        tote_win=race_data.get("tote_win"),
        tote_pl=race_data.get("tote_pl"),
        tote_ex=race_data.get("tote_ex"),
        tote_csf=race_data.get("tote_csf"),
        tote_tricast=race_data.get("tote_tricast"),
        tote_trifecta=race_data.get("tote_trifecta"),
        winning_time_detail=race_data.get("winning_time_detail"),
        region=region,
        region_code=region_code,
        api_course_id=race_data.get("course_id"),
        raw_json=raw_json,
    )

    for runner in runners:
        try:
            _ingest_runner(con, race_id, runner)
        except Exception as e:
            logger.error(
                f"Runner ingest error: {runner.get('horse', '?')} "
                f"in {api_race_id}: {e}"
            )


def _ingest_runner(con, race_id: int, runner: dict):
    """Ingest a single runner into horses, jockeys, trainers, results."""
    horse_name = runner.get("horse", "")
    if not horse_name:
        return

    api_horse_id = runner.get("horse_id")
    age = None
    try:
        age = int(runner.get("age", 0)) or None
    except (ValueError, TypeError):
        pass

    horse_id = upsert_horse_by_api_id(
        con, api_horse_id, horse_name,
        age=age, sex=runner.get("sex"),
        sire=runner.get("sire"), dam=runner.get("dam"),
        damsire=runner.get("damsire"),
        api_sire_id=runner.get("sire_id"),
        api_dam_id=runner.get("dam_id"),
        api_damsire_id=runner.get("damsire_id"),
    )

    jockey_name = runner.get("jockey", "")
    jockey_id = 0
    if jockey_name:
        jockey_id = upsert_jockey_by_api_id(
            con, runner.get("jockey_id"), jockey_name
        )

    trainer_name = runner.get("trainer", "")
    trainer_id = 0
    if trainer_name:
        trainer_id = upsert_trainer_by_api_id(
            con, runner.get("trainer_id"), trainer_name
        )

    insert_result_from_api(con, race_id, runner, horse_id,
                           jockey_id, trainer_id)


def _region_to_code(region: str) -> str:
    """Map API region name to lowercase code."""
    mapping = {
        "Great Britain": "gb", "Ireland": "ire", "IRE": "ire",
        "France": "fr", "USA": "us", "Hong Kong": "hk",
        "Australia": "aus", "South Africa": "sa",
        "United Arab Emirates": "uae", "Japan": "jpn",
        "Germany": "ger", "Canada": "can", "Italy": "ity",
        "Argentina": "arg", "New Zealand": "nz",
    }
    return mapping.get(region, region.lower()[:3] if region else "")


def _region_to_country(region: str) -> str:
    """Map API region to ISO-ish country code."""
    mapping = {
        "Great Britain": "GB", "Ireland": "IE", "IRE": "IE",
        "France": "FR", "USA": "US", "Hong Kong": "HK",
        "Australia": "AU", "South Africa": "ZA",
        "United Arab Emirates": "AE", "Japan": "JP",
        "Germany": "DE", "Canada": "CA", "Italy": "IT",
    }
    return mapping.get(region, region[:2].upper() if region else "GB")


# ===================================================================
# Course sync (Phase 1)
# ===================================================================

def sync_courses(client: RacingAPIClient):
    """Fetch all courses and regions from the API and store them."""
    init_database()
    con = get_connection()
    try:
        regions_data = client.get_regions()
        logger.info(f"Fetched {len(regions_data)} regions")

        courses_data = client.get_courses()
        courses_list = courses_data.get("courses", [])
        logger.info(f"Fetched {len(courses_list)} courses")

        inserted = 0
        for c in courses_list:
            course_name = c.get("course", "")
            api_id = c.get("id", "")
            region = c.get("region", "")
            region_code = c.get("region_code", "")
            country = _region_to_country(region)

            try:
                existing = con.execute(
                    "SELECT course FROM courses WHERE course = ?",
                    [course_name]
                ).fetchone()
                if existing:
                    con.execute("""
                        UPDATE courses SET api_course_id = ?, region_code = ?,
                                           country = ?
                        WHERE course = ?
                    """, [api_id, region_code, country, course_name])
                else:
                    con.execute("""
                        INSERT INTO courses (course, api_course_id, country,
                                             region_code)
                        VALUES (?, ?, ?, ?)
                    """, [course_name, api_id, country, region_code])
                    inserted += 1
            except Exception as e:
                logger.error(f"Course insert error {course_name}: {e}")

        logger.info(f"Courses synced: {inserted} new, {len(courses_list)} total")
    finally:
        con.close()


# ===================================================================
# Horse profile enrichment (Phase 4)
# ===================================================================

def fetch_all_horse_profiles(client: RacingAPIClient):
    """Fetch full profile for horses in GB/IRE regions (where API has data).
    Marks 404 horses as attempted so they aren't retried."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT h.api_horse_id
            FROM horses h
            WHERE h.api_horse_id IS NOT NULL
              AND h.api_horse_id != ''
              AND h.profile_scraped_at IS NULL
        """).fetchall()
        todo = [r[0] for r in rows]

        total_horses = con.execute(
            "SELECT COUNT(*) FROM horses WHERE api_horse_id IS NOT NULL"
        ).fetchone()[0]
        already_done = total_horses - len(todo)

        logger.info(
            f"Horse profiles: {len(todo)} to fetch "
            f"({already_done} already done, {total_horses} total)"
        )

        success = 0
        not_found = 0
        errors = 0

        for i, api_id in enumerate(todo):
            try:
                data = client.get_horse(api_id)
                _store_horse_profile(con, api_id, data)
                success += 1
            except Exception as e:
                err_str = str(e)
                if "404" in err_str:
                    not_found += 1
                    con.execute("""
                        UPDATE horses SET profile_scraped_at = current_timestamp
                        WHERE api_horse_id = ?
                    """, [api_id])
                else:
                    errors += 1
                    if errors <= 20:
                        logger.error(f"Horse profile {api_id}: {e}")

            if (i + 1) % 200 == 0:
                logger.info(
                    f"  Horse profiles: {i+1}/{len(todo)} "
                    f"(ok={success}, 404={not_found}, err={errors})"
                )

        logger.info(
            f"Horse profiles complete: {len(todo)} processed "
            f"(ok={success}, 404={not_found}, err={errors})"
        )
    finally:
        con.close()


def _store_horse_profile(con, api_horse_id: str, data: dict):
    """Store full horse profile from /v1/horses/{id} response."""
    raw = json.dumps(data)
    con.execute("""
        UPDATE horses SET
            colour = ?, country_bred = ?,
            sire = ?, dam = ?, damsire = ?,
            sire_sire = ?, sire_dam = ?, dam_sire = ?, dam_dam = ?,
            raw_json = ?, profile_scraped_at = current_timestamp
        WHERE api_horse_id = ?
    """, [
        data.get("colour"), data.get("region"),
        data.get("sire"), data.get("dam"), data.get("damsire"),
        data.get("sire_sire"), data.get("sire_dam"),
        data.get("dam_sire"), data.get("dam_dam"),
        raw, api_horse_id,
    ])

    horse_name = data.get("horse", "")
    if horse_name:
        try:
            existing = con.execute(
                "SELECT pedigree_id FROM pedigree WHERE horse_name = ?",
                [horse_name]
            ).fetchone()
            if not existing:
                pid = con.execute(
                    "SELECT nextval('seq_pedigree_id')"
                ).fetchone()[0]
                con.execute("""
                    INSERT INTO pedigree (
                        pedigree_id, horse_name, api_horse_id,
                        sire, dam, sire_sire, sire_dam,
                        dam_sire, dam_dam
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                """, [
                    pid, horse_name, api_horse_id,
                    data.get("sire"), data.get("dam"),
                    data.get("sire_sire"), data.get("sire_dam"),
                    data.get("dam_sire"), data.get("dam_dam"),
                ])
        except Exception as e:
            logger.error(f"Pedigree insert {horse_name}: {e}")


# ===================================================================
# Horse form history (Phase 4b) -- deep career data per horse
# ===================================================================

def fetch_all_horse_form(client: RacingAPIClient):
    """Fetch full career results for every horse in the DB.
    This gives us years of historical form data even though the
    /results endpoint only goes back 12 months.
    Stores into the horse_form table."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_horse_id, name FROM horses
            WHERE api_horse_id IS NOT NULL AND api_horse_id != ''
        """).fetchall()

        already = con.execute("""
            SELECT DISTINCT api_horse_id FROM horse_form
            WHERE api_horse_id IS NOT NULL
        """).fetchall()
        done_set = {r[0] for r in already}

        todo = [(aid, name) for aid, name in rows if aid not in done_set]
        logger.info(
            f"Horse form history: {len(todo)} to fetch "
            f"({len(done_set)} already done, {len(rows)} total horses)"
        )

        success = 0
        not_found = 0
        errors = 0
        total_form_entries = 0

        for i, (api_id, horse_name) in enumerate(todo):
            try:
                skip = 0
                limit = 100
                all_results = []
                while True:
                    data = client.get_horse_results(api_id, limit=limit, skip=skip)
                    results = data.get("results", [])
                    all_results.extend(results)
                    if len(results) < limit:
                        break
                    skip += limit

                for race in all_results:
                    _store_horse_form_entry(con, horse_name, api_id, race)

                total_form_entries += len(all_results)
                success += 1
            except Exception as e:
                if "404" in str(e):
                    not_found += 1
                else:
                    errors += 1
                    if errors <= 20:
                        logger.error(f"Horse form {horse_name} ({api_id}): {e}")

            if (i + 1) % 100 == 0:
                logger.info(
                    f"  Horse form: {i+1}/{len(todo)} "
                    f"(ok={success}, 404={not_found}, err={errors}, "
                    f"form_entries={total_form_entries:,})"
                )

        logger.info(
            f"Horse form history complete: {len(todo)} processed "
            f"(ok={success}, 404={not_found}, err={errors}, "
            f"total_form_entries={total_form_entries:,})"
        )
    finally:
        con.close()


def _store_horse_form_entry(con, horse_name: str, api_horse_id: str,
                            race_data: dict):
    """Store a single race entry from a horse's career into horse_form."""
    course = race_data.get("course", "")
    race_date = race_data.get("date", "")
    if not race_date or not course:
        return

    existing = con.execute(
        "SELECT form_id FROM horse_form WHERE horse_name = ? "
        "AND race_date = ? AND course = ?",
        [horse_name, race_date, course]
    ).fetchone()
    if existing:
        return

    runners = race_data.get("runners", [])
    my_runner = None
    winner_name = None
    for r in runners:
        if r.get("horse_id") == api_horse_id or r.get("horse") == horse_name:
            my_runner = r
        pos = r.get("position")
        if pos and str(pos) == "1":
            winner_name = r.get("horse", "")

    if not my_runner:
        return

    fid = con.execute("SELECT nextval('seq_form_id')").fetchone()[0]

    pos = my_runner.get("position")
    try:
        pos = int(pos) if pos and str(pos).isdigit() else None
    except (ValueError, TypeError):
        pos = None

    dist_f = _parse_float(race_data.get("dist_f", "").replace("f", "")
                          if race_data.get("dist_f") else None)
    sp_dec = _parse_float(my_runner.get("sp_dec"))
    btn = _parse_float(my_runner.get("btn")) if my_runner.get("btn") else None
    or_val = None
    try:
        or_val = int(my_runner.get("or")) if my_runner.get("or") else None
    except (ValueError, TypeError):
        pass

    draw = None
    try:
        draw = int(my_runner.get("draw")) if my_runner.get("draw") else None
    except (ValueError, TypeError):
        pass

    con.execute("""
        INSERT INTO horse_form (
            form_id, horse_name, api_horse_id, race_date, course,
            distance_furlongs, race_type, race_class, going,
            position, beaten_lengths, finish_time, sp_decimal,
            weight_carried, draw, jockey, trainer, official_rating,
            speed_figure, comment, num_runners, winner_name
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        fid, horse_name, api_horse_id, race_date, course,
        dist_f, race_data.get("type"), race_data.get("class"),
        race_data.get("going"),
        pos, btn, None, sp_dec,
        my_runner.get("weight"), draw,
        my_runner.get("jockey"), my_runner.get("trainer"),
        or_val, None, my_runner.get("comment"),
        len(runners), winner_name,
    ])


# ===================================================================
# Jockey enrichment (Phase 5)
# ===================================================================

def fetch_all_jockey_data(client: RacingAPIClient):
    """Fetch profile data for every jockey in the DB."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_id FROM jockeys
            WHERE api_id IS NOT NULL AND api_id != ''
              AND profile_scraped_at IS NULL
        """).fetchall()
        todo = [r[0] for r in rows]
        logger.info(f"Jockey enrichment: {len(todo)} to fetch")

        for i, api_id in enumerate(todo):
            try:
                analysis = client.get_jockey_course_analysis(api_id)
                raw = json.dumps(analysis)
                con.execute("""
                    UPDATE jockeys SET raw_json = ?,
                                       profile_scraped_at = current_timestamp
                    WHERE api_id = ?
                """, [raw, api_id])

                if (i + 1) % 100 == 0:
                    logger.info(f"  Jockeys: {i+1}/{len(todo)}")
            except Exception as e:
                logger.error(f"Jockey {api_id}: {e}")

        logger.info(f"Jockey enrichment complete: {len(todo)} fetched")
    finally:
        con.close()


# ===================================================================
# Trainer enrichment (Phase 5)
# ===================================================================

def fetch_all_trainer_data(client: RacingAPIClient):
    """Fetch profile data for every trainer in the DB."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_id FROM trainers
            WHERE api_id IS NOT NULL AND api_id != ''
              AND profile_scraped_at IS NULL
        """).fetchall()
        todo = [r[0] for r in rows]
        logger.info(f"Trainer enrichment: {len(todo)} to fetch")

        for i, api_id in enumerate(todo):
            try:
                analysis = client.get_trainer_course_analysis(api_id)
                raw = json.dumps(analysis)
                con.execute("""
                    UPDATE trainers SET raw_json = ?,
                                        profile_scraped_at = current_timestamp
                    WHERE api_id = ?
                """, [raw, api_id])

                if (i + 1) % 100 == 0:
                    logger.info(f"  Trainers: {i+1}/{len(todo)}")
            except Exception as e:
                logger.error(f"Trainer {api_id}: {e}")

        logger.info(f"Trainer enrichment complete: {len(todo)} fetched")
    finally:
        con.close()


# ===================================================================
# Sire enrichment (Phase 6)
# ===================================================================

def _parse_sire_analysis(analysis: dict) -> dict:
    """Extract structured fields from sire distance analysis API response."""
    distances = analysis.get("distances", [])
    total_runners = analysis.get("total_runners", 0)
    total_wins = sum(d.get("1st", 0) for d in distances)
    win_rate = total_wins / total_runners if total_runners > 0 else 0.0

    # Weighted average distance
    weighted_sum, total_r = 0.0, 0
    for d in distances:
        runners = d.get("runners", 0)
        dist_f_str = d.get("dist_f", "")
        import re as _re
        m = _re.match(r"([\d.]+)", dist_f_str.replace("f", "")) if dist_f_str else None
        dist_f = float(m.group(1)) if m else 0.0
        if runners > 0 and dist_f > 0:
            weighted_sum += dist_f * runners
            total_r += runners
    avg_dist = weighted_sum / total_r if total_r > 0 else None

    flat_r, flat_w, nh_r, nh_w = 0, 0, 0, 0
    for d in distances:
        dist_f_str = d.get("dist_f", "")
        m = _re.match(r"([\d.]+)", dist_f_str.replace("f", "")) if dist_f_str else None
        dist_f = float(m.group(1)) if m else 0.0
        r, w = d.get("runners", 0), d.get("1st", 0)
        if dist_f <= 16:
            flat_r += r; flat_w += w
        else:
            nh_r += r; nh_w += w

    return {
        "total_runners": total_runners, "total_wins": total_wins,
        "win_rate": win_rate, "avg_dist": avg_dist,
        "flat_rate": flat_w / flat_r if flat_r > 0 else 0.0,
        "nh_rate": nh_w / nh_r if nh_r > 0 else 0.0,
    }


def fetch_all_sire_data(client: RacingAPIClient):
    """Fetch progeny stats for every sire in the DB."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_sire_id FROM results
            WHERE api_sire_id IS NOT NULL AND api_sire_id != ''
        """).fetchall()
        all_ids = {r[0] for r in rows}

        done = con.execute("""
            SELECT api_sire_id FROM sire_stats
            WHERE api_sire_id IS NOT NULL AND progeny_count > 0
        """).fetchall()
        done_set = {r[0] for r in done}

        todo = [sid for sid in all_ids if sid not in done_set]
        logger.info(f"Sire enrichment: {len(todo)} to fetch")

        for i, api_id in enumerate(todo):
            try:
                analysis = client.get_sire_distance_analysis(api_id)
                sire_name = analysis.get("sire", api_id)
                raw = json.dumps(analysis)
                parsed = _parse_sire_analysis(analysis)

                con.execute("""
                    INSERT INTO sire_stats
                        (sire_id, api_sire_id, sire_name, progeny_count, progeny_wins,
                         progeny_win_rate, avg_distance_furlongs, flat_win_rate, nh_win_rate, raw_json)
                    VALUES (nextval('seq_sire_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (api_sire_id) DO UPDATE SET
                        progeny_count = EXCLUDED.progeny_count,
                        progeny_wins = EXCLUDED.progeny_wins,
                        progeny_win_rate = EXCLUDED.progeny_win_rate,
                        avg_distance_furlongs = EXCLUDED.avg_distance_furlongs,
                        flat_win_rate = EXCLUDED.flat_win_rate,
                        nh_win_rate = EXCLUDED.nh_win_rate,
                        raw_json = EXCLUDED.raw_json
                """, [api_id, sire_name, parsed["total_runners"], parsed["total_wins"],
                      parsed["win_rate"], parsed["avg_dist"], parsed["flat_rate"],
                      parsed["nh_rate"], raw])

                import re as _re
                for d in analysis.get("distances", []):
                    dist_band = d.get("dist_f", d.get("dist", ""))
                    m = _re.match(r"([\d.]+)", dist_band.replace("f", "")) if dist_band else None
                    dist_f = float(m.group(1)) if m else 0.0
                    runners = d.get("runners", 0)
                    wins = d.get("1st", 0)
                    places = wins + d.get("2nd", 0) + d.get("3rd", 0)
                    if runners >= 5:
                        con.execute("""
                            INSERT INTO sire_distance_stats
                                (api_sire_id, distance_band, distance_f, runners, wins, win_rate, place_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (api_sire_id, distance_band) DO UPDATE SET
                                runners = EXCLUDED.runners, wins = EXCLUDED.wins,
                                win_rate = EXCLUDED.win_rate, place_rate = EXCLUDED.place_rate
                        """, [api_id, dist_band, dist_f, runners, wins,
                              wins / runners if runners > 0 else 0.0,
                              places / runners if runners > 0 else 0.0])

                if (i + 1) % 100 == 0:
                    logger.info(f"  Sires: {i+1}/{len(todo)}")
            except Exception as e:
                if "UNIQUE" not in str(e).upper():
                    logger.error(f"Sire {api_id}: {e}")

        logger.info(f"Sire enrichment complete: {len(todo)} fetched")
    finally:
        con.close()


# ===================================================================
# Dam enrichment (Phase 6)
# ===================================================================

def fetch_all_dam_data(client: RacingAPIClient):
    """Fetch progeny stats for every dam in the DB."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_dam_id FROM results
            WHERE api_dam_id IS NOT NULL AND api_dam_id != ''
        """).fetchall()
        all_ids = {r[0] for r in rows}

        done = con.execute("""
            SELECT api_dam_id FROM dam_stats
            WHERE api_dam_id IS NOT NULL AND progeny_count > 0
        """).fetchall()
        done_set = {r[0] for r in done}

        todo = [did for did in all_ids if did not in done_set]
        logger.info(f"Dam enrichment: {len(todo)} to fetch")

        for i, api_id in enumerate(todo):
            try:
                analysis = client.get_dam_distance_analysis(api_id)

                if "detail" in analysis and "No results" in str(analysis.get("detail", "")):
                    continue

                dam_name = analysis.get("dam", api_id)
                total_runners = analysis.get("total_runners", 0)
                if total_runners == 0:
                    continue

                distances = analysis.get("distances", [])
                raw = json.dumps(analysis)
                total_wins = sum(d.get("1st", 0) for d in distances)
                win_rate = total_wins / total_runners if total_runners > 0 else 0.0

                import re as _re
                weighted_sum, total_r = 0.0, 0
                for d in distances:
                    r = d.get("runners", 0)
                    df_str = d.get("dist_f", "")
                    m = _re.match(r"([\d.]+)", df_str.replace("f", "")) if df_str else None
                    df = float(m.group(1)) if m else 0.0
                    if r > 0 and df > 0:
                        weighted_sum += df * r
                        total_r += r
                avg_dist = weighted_sum / total_r if total_r > 0 else None

                con.execute("""
                    INSERT INTO dam_stats
                        (dam_id, api_dam_id, dam_name, progeny_count, progeny_wins,
                         progeny_win_rate, avg_distance_furlongs, raw_json)
                    VALUES (nextval('seq_dam_id'), ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (api_dam_id) DO UPDATE SET
                        progeny_count = EXCLUDED.progeny_count,
                        progeny_wins = EXCLUDED.progeny_wins,
                        progeny_win_rate = EXCLUDED.progeny_win_rate,
                        avg_distance_furlongs = EXCLUDED.avg_distance_furlongs,
                        raw_json = EXCLUDED.raw_json
                """, [api_id, dam_name, total_runners, total_wins,
                      win_rate, avg_dist, raw])

                if (i + 1) % 200 == 0:
                    logger.info(f"  Dams: {i+1}/{len(todo)}")
            except Exception as e:
                if "UNIQUE" not in str(e).upper():
                    logger.error(f"Dam {api_id}: {e}")

        logger.info(f"Dam enrichment complete: {len(todo)} fetched")
    finally:
        con.close()


# ===================================================================
# Damsire enrichment (Phase 6)
# ===================================================================

def fetch_all_damsire_data(client: RacingAPIClient):
    """Fetch grandoffspring stats for every damsire in the DB."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT DISTINCT api_damsire_id FROM results
            WHERE api_damsire_id IS NOT NULL AND api_damsire_id != ''
        """).fetchall()
        all_ids = {r[0] for r in rows}

        done = con.execute("""
            SELECT api_damsire_id FROM damsire_stats
            WHERE api_damsire_id IS NOT NULL
        """).fetchall()
        done_set = {r[0] for r in done}

        todo = [did for did in all_ids if did not in done_set]
        logger.info(f"Damsire enrichment: {len(todo)} to fetch")

        for i, api_id in enumerate(todo):
            try:
                analysis = client.get_damsire_results(api_id, limit=1, skip=0)
                raw = json.dumps(analysis)
                damsire_name = api_id

                did = con.execute(
                    "SELECT nextval('seq_damsire_id')"
                ).fetchone()[0]
                con.execute("""
                    INSERT INTO damsire_stats (damsire_id, api_damsire_id,
                                               damsire_name, raw_json)
                    VALUES (?, ?, ?, ?)
                """, [did, api_id, damsire_name, raw])

                if (i + 1) % 100 == 0:
                    logger.info(f"  Damsires: {i+1}/{len(todo)}")
            except Exception as e:
                if "UNIQUE" not in str(e).upper():
                    logger.error(f"Damsire {api_id}: {e}")

        logger.info(f"Damsire enrichment complete: {len(todo)} fetched")
    finally:
        con.close()


# ===================================================================
# Racecards (upcoming races)
# ===================================================================

def fetch_upcoming_racecards(client: RacingAPIClient):
    """Fetch upcoming racecards (today + tomorrow) and store as meetings + races."""
    init_database()
    con = get_connection()
    try:
        today = date.today()
        tomorrow = today + timedelta(days=1)
        total = 0

        for d in [today, tomorrow]:
            date_str = d.strftime("%Y-%m-%d")
            try:
                data = client.get_racecards_pro(date=date_str)
                racecards = data.get("racecards", [])
                logger.info(f"Fetched {len(racecards)} racecards for {date_str}")

                for idx, rc in enumerate(racecards):
                    try:
                        _ingest_race(con, rc, idx)
                    except Exception as e:
                        logger.error(f"Racecard ingest error: {e}")

                total += len(racecards)
            except Exception as e:
                logger.warning(f"Failed to fetch racecards for {date_str}: {e}")

        logger.info(f"Racecards stored: {total} (today + tomorrow)")
    finally:
        con.close()


def _upsert_odds(con, internal_race_id: int, horse_name: str, back: float):
    """Upsert a single runner's odds into market_data."""
    existing = con.execute(
        "SELECT market_id FROM market_data WHERE race_id = ? AND horse_name = ?",
        [internal_race_id, horse_name]
    ).fetchone()

    if existing:
        con.execute("""
            UPDATE market_data
            SET back_odds = ?, captured_at = current_timestamp
            WHERE market_id = ?
        """, [back, existing[0]])
    else:
        mid = con.execute(
            "SELECT nextval('seq_market_id')"
        ).fetchone()[0]
        con.execute("""
            INSERT INTO market_data (market_id, race_id, horse_name,
                back_odds, captured_at)
            VALUES (?, ?, ?, ?, current_timestamp)
        """, [mid, internal_race_id, horse_name, back])


def fetch_odds_for_race(client: RacingAPIClient, api_race_id: str,
                        internal_race_id: int) -> dict:
    """Fetch live odds for every runner in a single race via per-runner API.

    The Racing API endpoint is /v1/odds/{race_id}/{horse_id}, returning:
    {race_id, horse_id, horse, odds: [{bookmaker, decimal, fractional, ...}]}

    We pick the best available price across bookmakers.
    Returns dict mapping horse_name -> back_odds for immediate use.
    """
    odds_map = {}

    con = get_connection()
    try:
        runner_rows = con.execute("""
            SELECT horse_name, api_horse_id
            FROM results
            WHERE race_id = ? AND api_horse_id IS NOT NULL
        """, [internal_race_id]).fetchall()
    finally:
        con.close()

    if not runner_rows:
        logger.debug(f"No runners with api_horse_id for race {api_race_id}")
        return odds_map

    con = get_connection()
    try:
        for horse_name, api_horse_id in runner_rows:
            try:
                data = client.get_odds_runner(api_race_id, api_horse_id)
                bookmaker_odds = data.get("odds", [])
                if not bookmaker_odds:
                    continue

                best_decimal = None
                for bm in bookmaker_odds:
                    try:
                        dec = float(bm.get("decimal", 0))
                        if dec > 1.0 and (best_decimal is None or dec > best_decimal):
                            best_decimal = dec
                    except (ValueError, TypeError):
                        continue

                if best_decimal and best_decimal > 1.0:
                    odds_map[horse_name] = best_decimal
                    _upsert_odds(con, internal_race_id, horse_name, best_decimal)

                time.sleep(0.22)

            except Exception as e:
                logger.debug(f"Odds fetch for {horse_name} ({api_horse_id}): {e}")
                continue

        logger.info(f"Stored odds for {len(odds_map)} runners in race {api_race_id}")
    finally:
        con.close()

    return odds_map


def fetch_odds_for_today(client: RacingAPIClient) -> int:
    """Fetch odds for all races today and tomorrow. Returns count of races processed."""
    from datetime import date, timedelta
    init_database()
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
    finally:
        con.close()

    logger.info(f"Fetching odds for {len(rows)} races (today + tomorrow)")
    count = 0
    for race_id, api_race_id in rows:
        fetch_odds_for_race(client, api_race_id, race_id)
        count += 1

    logger.info(f"Odds refresh complete: {count} races processed")
    return count
