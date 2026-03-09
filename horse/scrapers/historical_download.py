"""
Historical data downloader for 15-year training window.

Downloads race results month-by-month from The Racing API, saves:
  - Raw JSON: horse/data/raw/YYYY_raw.json
  - Cleaned Parquet: horse/data/results/YYYY.parquet

Then loads into horse.duckdb for training.

Usage:
    python -m horse.scrapers.historical_download              # download all years
    python -m horse.scrapers.historical_download --year 2015  # single year
    python -m horse.scrapers.historical_download --load-only  # skip download, just load DB
"""

import argparse
import json
import logging
import os
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from horse.config import (
    DATA_DIR, RAW_DIR, RESULTS_DIR, TRAIN_YEARS,
    RACING_API_USERNAME, RACING_API_PASSWORD,
    RACING_API_BASE, RACING_API_RATE_LIMIT,
)
from horse.db import get_connection, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("historical_download.log", mode="a"),
    ],
)
logger = logging.getLogger("horse.historical")


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_position(val) -> int:
    if val is None:
        return 0
    try:
        p = int(val)
        return p if p > 0 else 0
    except (ValueError, TypeError):
        return 0


def _parse_float(val, default=None):
    if val is None:
        return default
    try:
        return float(str(val).replace("f", "").strip())
    except (ValueError, TypeError):
        return default


class HistoricalAPIClient:
    """Minimal API client with aggressive rate-limit handling for bulk downloads."""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.auth = (RACING_API_USERNAME, RACING_API_PASSWORD)
        self.session.headers.update({
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        })
        self.base = RACING_API_BASE.rstrip("/")
        self._last_request = 0
        self._min_interval = 1.0 / max(RACING_API_RATE_LIMIT, 1)
        self.consecutive_429s = 0

    def get_results(self, start_date: str, end_date: str,
                    limit: int = 50, skip: int = 0) -> dict:
        """Fetch results with max 3 retries on 429. Raises on persistent block."""
        import requests as req
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "skip": skip,
        }

        for attempt in range(3):
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request = time.time()
            url = f"{self.base}/results"

            try:
                resp = self.session.get(url, params=params, timeout=30)
            except (req.exceptions.ConnectionError, req.exceptions.Timeout) as e:
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise

            if resp.status_code == 200:
                self.consecutive_429s = 0
                return resp.json()

            if resp.status_code == 429:
                self.consecutive_429s += 1
                if self.consecutive_429s >= 5:
                    raise RuntimeError(f"API blocked: 5 consecutive 429s for {start_date}")
                wait = 30 * (attempt + 1)
                logger.warning(f"429 rate limit. Waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue

            if resp.status_code == 422:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = resp.text[:200]
                raise ValueError(f"API 422: {detail}")

            if resp.status_code == 401:
                raise PermissionError("API 401: Check credentials in horse/.env")

            resp.raise_for_status()

        raise RuntimeError(f"Failed after 3 attempts for {start_date} to {end_date}")


def download_year(client, year: int) -> Path:
    """Download results for a year month-by-month. Returns path to raw JSON."""
    raw_path = RAW_DIR / f"{year}_raw.json"

    if raw_path.exists() and raw_path.stat().st_size > 1000:
        logger.info(f"{year}: Already downloaded ({raw_path.stat().st_size / 1024 / 1024:.1f} MB), skipping")
        return raw_path

    logger.info(f"{year}: Starting download (month by month)...")
    all_races = []
    blocked = False

    for month in range(1, 13):
        if year == date.today().year and month > date.today().month:
            break

        _, last_day = monthrange(year, month)
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day}"
        if year == date.today().year and month == date.today().month:
            end = date.today().isoformat()

        skip = 0
        limit = 50
        month_races = []

        while True:
            try:
                data = client.get_results(start, end, limit=limit, skip=skip)
            except RuntimeError as e:
                if "429" in str(e) or "blocked" in str(e).lower():
                    logger.warning(f"{year}-{month:02d}: API blocked this date range, skipping year")
                    blocked = True
                    break
                raise
            except ValueError as e:
                logger.warning(f"{year}-{month:02d}: API rejected: {e}")
                break
            except Exception as e:
                logger.error(f"{year}-{month:02d}: Error: {e}")
                break

            results_list = data.get("results", [])
            if not results_list:
                break

            month_races.extend(results_list)
            skip += limit

            if len(results_list) < limit:
                break

            time.sleep(0.5)

        if blocked:
            break

        if month_races:
            logger.info(f"  {year}-{month:02d}: {len(month_races)} races")
            all_races.extend(month_races)

        time.sleep(1)

    logger.info(f"{year}: Total {len(all_races)} races downloaded")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_races, f)

    if all_races:
        logger.info(f"{year}: Saved ({raw_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.warning(f"{year}: No data - API may not allow this date range")

    return raw_path


def normalize_year(year: int) -> Path:
    """Read raw JSON for a year and produce a cleaned Parquet file."""
    raw_path = RAW_DIR / f"{year}_raw.json"
    parquet_path = RESULTS_DIR / f"{year}.parquet"

    if parquet_path.exists() and parquet_path.stat().st_size > 100:
        existing = pd.read_parquet(parquet_path)
        if len(existing) > 0:
            logger.info(f"{year}: Parquet exists ({len(existing)} rows), skipping")
            return parquet_path

    if not raw_path.exists():
        logger.warning(f"{year}: No raw file, skipping")
        return parquet_path

    with open(raw_path, "r", encoding="utf-8") as f:
        races = json.load(f)

    if not races:
        logger.warning(f"{year}: Empty raw file, skipping")
        return parquet_path

    logger.info(f"{year}: Normalizing {len(races)} races to Parquet...")

    rows = []
    for race_data in races:
        api_race_id = race_data.get("race_id", "")
        course = race_data.get("course", "")
        race_date_str = race_data.get("date", "")
        region = race_data.get("region", "")
        going = race_data.get("going", "")
        race_type = race_data.get("type", "")
        race_class = race_data.get("class", "")
        surface = race_data.get("surface", "")
        dist_f = _parse_float(race_data.get("dist_f", ""))
        race_name = race_data.get("race_name", "")
        runners = race_data.get("runners", [])
        num_runners = len(runners)

        try:
            meeting_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        for runner in runners:
            position = _parse_position(runner.get("position"))
            if position == 0:
                continue

            rows.append({
                "api_race_id": api_race_id,
                "course": course,
                "meeting_date": meeting_date,
                "region": region,
                "race_name": race_name,
                "race_type": race_type,
                "race_class": race_class,
                "going": going,
                "surface": surface,
                "distance_furlongs": dist_f,
                "num_runners": num_runners,
                "horse_name": runner.get("horse", ""),
                "api_horse_id": runner.get("horse_id", ""),
                "position": position,
                "draw": runner.get("draw"),
                "sp_decimal": _parse_float(runner.get("sp_dec")),
                "official_rating": _parse_float(runner.get("or")),
                "weight_lbs": _parse_float(runner.get("lbs")),
                "age": _parse_float(runner.get("age")),
                "sex": runner.get("sex", ""),
                "api_jockey_id": runner.get("jockey_id", ""),
                "api_trainer_id": runner.get("trainer_id", ""),
                "api_sire_id": runner.get("sire_id", ""),
                "api_dam_id": runner.get("dam_id", ""),
                "beaten_lengths": _parse_float(runner.get("btn")),
                "finish_time": _parse_float(runner.get("time")),
                "region_code": region[:3].lower() if region else "",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning(f"{year}: No valid rows")
        return parquet_path

    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    df["position"] = df["position"].astype(int)
    df["num_runners"] = df["num_runners"].astype(int)
    df.to_parquet(parquet_path, index=False)

    logger.info(f"{year}: {len(df)} runners, {df['api_race_id'].nunique()} races "
                f"({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return parquet_path


def load_parquet_to_db():
    """Load all Parquet files into horse.duckdb."""
    from horse.scrapers.racing_api import _ingest_race

    init_database()
    con = get_connection()

    parquet_files = sorted(RESULTS_DIR.glob("*.parquet"))
    raw_files = {rf.stem.replace("_raw", ""): rf for rf in sorted(RAW_DIR.glob("*_raw.json"))}

    if not parquet_files:
        logger.warning("No Parquet files found")
        return

    logger.info(f"Loading {len(parquet_files)} year files into DuckDB...")
    total_races = 0
    total_runners = 0

    for pf in parquet_files:
        year_str = pf.stem
        raw_path = raw_files.get(year_str)

        if not raw_path or not raw_path.exists():
            continue

        df = pd.read_parquet(pf)
        if df.empty:
            continue

        expected = df["api_race_id"].nunique()
        existing = con.execute("""
            SELECT COUNT(DISTINCT ra.race_id) FROM races ra
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE EXTRACT(YEAR FROM m.meeting_date) = ?
        """, [int(year_str)]).fetchone()[0]

        if existing >= expected * 0.9 and existing > 0:
            logger.info(f"{year_str}: Already in DB ({existing} races), skipping")
            continue

        logger.info(f"{year_str}: Loading {expected} races into DB...")
        with open(raw_path, "r", encoding="utf-8") as f:
            races = json.load(f)

        yr, rn = 0, 0
        for idx, race_data in enumerate(races):
            try:
                _ingest_race(con, race_data, idx)
                yr += 1
                rn += len(race_data.get("runners", []))
            except Exception:
                pass
            if (idx + 1) % 1000 == 0:
                logger.info(f"  {year_str}: {idx+1}/{len(races)}...")

        total_races += yr
        total_runners += rn
        logger.info(f"{year_str}: {yr} races, {rn} runners loaded")

    con.close()
    logger.info(f"DB load done: {total_races} races, {total_runners} runners")


def download_all(start_year=None, end_year=None):
    """Download, normalize, and load all years."""
    _ensure_dirs()

    if not RACING_API_USERNAME:
        logger.error("RACING_API_USERNAME not set in horse/.env")
        return

    today = date.today()
    if not start_year:
        start_year = today.year - TRAIN_YEARS
    if not end_year:
        end_year = today.year

    client = HistoricalAPIClient()

    logger.info("=" * 60)
    logger.info(f"  HISTORICAL DOWNLOAD: {start_year} to {end_year}")
    logger.info(f"  {end_year - start_year + 1} years")
    logger.info("=" * 60)

    years_with_data = []
    for year in range(start_year, end_year + 1):
        try:
            raw_path = download_year(client, year)
            if raw_path.exists() and raw_path.stat().st_size > 100:
                pq = normalize_year(year)
                if pq.exists() and pq.stat().st_size > 100:
                    years_with_data.append(year)
        except Exception as e:
            logger.error(f"{year}: FAILED - {e}")
            continue
        time.sleep(2)

    logger.info("=" * 60)
    logger.info(f"  Downloaded {len(years_with_data)} years with data")
    logger.info(f"  Loading into DuckDB...")
    logger.info("=" * 60)

    load_parquet_to_db()

    logger.info("=" * 60)
    logger.info("  ALL DONE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download historical racing data")
    parser.add_argument("--year", type=int, help="Download a single year")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--normalize-only", action="store_true")
    args = parser.parse_args()

    if args.load_only:
        _ensure_dirs()
        load_parquet_to_db()
        return

    if args.normalize_only:
        _ensure_dirs()
        today = date.today()
        s = args.start_year or (today.year - TRAIN_YEARS)
        e = args.end_year or today.year
        for y in range(s, e + 1):
            normalize_year(y)
        return

    if args.year:
        _ensure_dirs()
        client = HistoricalAPIClient()
        download_year(client, args.year)
        normalize_year(args.year)
        load_parquet_to_db()
        return

    download_all(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
