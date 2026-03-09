"""
RacingTV / RaceiQ Sectional Timing Scraper
===========================================

Fetches GPS-derived per-furlong split times for every UK/IRE race in our DB
and stores computed metrics in the `sectionals` table.

Zero data-leakage guarantee
----------------------------
- Only races whose results already exist in the DB are queried.
- end_date is hard-capped to yesterday -- future races are never touched.
- `scraped_at` timestamp is recorded on every row for full auditability.
- Features derived from this table in features.py always filter
  meeting_date < target_race_date (enforced there, not here).

API endpoint (public, no auth required):
  GET https://api.racingtv.com/racing/iq-results/{YYYY-MM-DD}/{course-slug}/{HHMM}/sectional

Confirmed response structure:
  d["race"]["runners"] -> list of runners
  runner["horse_name"], runner["id"], runner["finish_position"]
  runner["finishing_speed_percentage"]
  runner["sectionals"] -> list of {sector_number, sector_time,
                                    time_behind_leader, sector_position,
                                    sector_name, sector_pace_category}

Usage:
  python -m horse.scrapers.sectionals --days 7
  python -m horse.scrapers.sectionals --start 2026-01-01 --end 2026-03-05
  python -m horse.scrapers.sectionals --days 7 --dry-run
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
import random
import time
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional

try:
    import brotli as _brotli
    _BROTLI_AVAILABLE = True
except ImportError:
    _BROTLI_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from horse.db import get_connection, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sectionals_scrape.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_BASE_URL = "https://api.racingtv.com/racing/iq-results/{date}/{course}/{time}/sectional"
_HEADERS = {
    "x-requested-with": "racingtv-web/5.5.0",
    "origin": "https://www.racingtv.com",
    "referer": "https://www.racingtv.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-GB,en;q=0.9",
    "content-type": "application/json",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "authorization": "",
}
_DELAY_OK  = 4.0   # conservative -- avoids Cloudflare rate limiting
_DELAY_ERR = 5.0

_SLUG_OVERRIDES = {
    "ascot": "ascot", "bath": "bath", "beverley": "beverley",
    "brighton": "brighton", "carlisle": "carlisle",
    "catterick": "catterick-bridge",
    "chepstow": "chepstow", "chester": "chester",
    "chelmsford": "chelmsford-city", "chelmsford (aw)": "chelmsford-city",
    "doncaster": "doncaster", "epsom": "epsom-downs",
    "epsom downs": "epsom-downs", "exeter": "exeter",
    "ffos las": "ffos-las", "goodwood": "goodwood",
    "hamilton": "hamilton-park", "haydock": "haydock-park",
    "kempton": "kempton-park", "kempton (aw)": "kempton-park",
    "leicester": "leicester", "lingfield": "lingfield-park",
    "lingfield (aw)": "lingfield-park", "musselburgh": "musselburgh",
    "newbury": "newbury", "newcastle": "newcastle",
    "newcastle (aw)": "newcastle", "newmarket": "newmarket",
    "newmarket (july)": "newmarket", "nottingham": "nottingham",
    "pontefract": "pontefract", "redcar": "redcar", "ripon": "ripon",
    "salisbury": "salisbury", "sandown": "sandown-park",
    "southwell": "southwell", "southwell (aw)": "southwell",
    "thirsk": "thirsk", "windsor": "windsor",
    "wolverhampton": "wolverhampton", "wolverhampton (aw)": "wolverhampton",
    "yarmouth": "great-yarmouth", "york": "york",
    "curragh": "the-curragh", "the curragh": "the-curragh",
    "leopardstown": "leopardstown", "navan": "navan", "naas": "naas",
    "dundalk": "dundalk", "dundalk (aw)": "dundalk",
    "dundalk (aw) (ire)": "dundalk", "dundalk (ire)": "dundalk",
    "tipperary": "tipperary", "galway": "galway", "cork": "cork",
    "fairyhouse": "fairyhouse", "punchestown": "punchestown",
    "killarney": "killarney", "listowel": "listowel",
    "tramore": "tramore", "limerick": "limerick",
    "roscommon": "roscommon", "sligo": "sligo",
    "downpatrick": "downpatrick", "ballinrobe": "ballinrobe",
    "bellewstown": "bellewstown", "clonmel": "clonmel",
    "gowran park": "gowran-park", "kilbeggan": "kilbeggan",
    "laytown": "laytown", "wexford": "wexford",
    "thurles": "thurles",
    "wincanton": "wincanton", "worcester": "worcester",
    "hexham": "hexham", "huntingdon": "huntingdon",
    "kelso": "kelso", "ludlow": "ludlow",
    "market rasen": "market-rasen", "perth": "perth",
    "plumpton": "plumpton", "sedgefield": "sedgefield",
    "stratford": "stratford-upon-avon", "taunton": "taunton",
    "towcester": "towcester", "uttoxeter": "uttoxeter",
    "warwick": "warwick", "wetherby": "wetherby",
}


def _course_to_slug(course: str) -> str:
    key = course.strip().lower()
    return _SLUG_OVERRIDES.get(key, re.sub(r"[^a-z0-9]+", "-", key).strip("-"))


def _time_to_hhmm(off_dt: str) -> Optional[str]:
    match = re.search(r"(\d{1,2}):(\d{2})", str(off_dt or "").strip())
    return f"{int(match.group(1)):02d}{match.group(2)}" if match else None


def _decompress(raw: bytes, encoding: str) -> bytes:
    enc = (encoding or "").lower()
    if enc == "br" and _BROTLI_AVAILABLE:
        return _brotli.decompress(raw)
    if enc in ("gzip", "x-gzip"):
        return gzip.decompress(raw)
    # Fallback: try gzip then brotli
    try:
        return gzip.decompress(raw)
    except Exception:
        pass
    if _BROTLI_AVAILABLE:
        try:
            return _brotli.decompress(raw)
        except Exception:
            pass
    return raw


def _fetch_sectional(race_date: str, course_slug: str, hhmm: str) -> Optional[dict]:
    url = _BASE_URL.format(date=race_date, course=course_slug, time=hhmm)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            encoding = resp.headers.get("Content-Encoding", "")
            raw = resp.read()
            raw = _decompress(raw, encoding)
            return json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code not in (404, 406):
            logger.warning("HTTP %s for %s/%s/%s", exc.code, race_date, course_slug, hhmm)
        return None
    except Exception as exc:
        logger.warning("Fetch error %s/%s/%s: %s", race_date, course_slug, hhmm, exc)
        return None


def _compute_metrics(runner: dict, total_furlongs: int) -> dict:
    """
    Extract standardised metrics from one runner's sectional payload.
    Field mapping from confirmed API response:
      runner["finishing_speed_percentage"] -> finishing_speed_pct
      runner["sectionals"][i]["sector_time"] -> split time
      runner["sectionals"][i]["time_behind_leader"] -> gap to leader
      runner["sectionals"][i]["sector_position"] -> position
      runner["sectionals"][i]["sector_number"] -> furlong index (0-based)
    """
    sects = runner.get("sectionals") or []
    result = {
        "finishing_speed_pct": None,
        "pos_at_2f": None,
        "pos_at_halfway": None,
        "gap_at_2f_secs": None,
        "gap_at_halfway_secs": None,
        "early_split_avg": None,
        "late_split_avg": None,
        "pace_consistency": None,
        "position_gain": None,
        "splits_json": json.dumps(sects) if sects else None,
    }

    # finishing_speed_percentage is directly provided
    try:
        result["finishing_speed_pct"] = float(runner.get("finishing_speed_percentage") or 0) or None
    except (TypeError, ValueError):
        pass

    if not sects:
        return result

    sects_sorted = sorted(sects, key=lambda s: s.get("sector_number", 0))
    times = [s["sector_time"] for s in sects_sorted if isinstance(s.get("sector_time"), (int, float))]
    n = len(times)

    if n >= 2:
        mid = n // 2
        result["early_split_avg"] = sum(times[:mid]) / mid
        result["late_split_avg"] = sum(times[mid:]) / max(n - mid, 1)
        if n >= 3:
            avg = sum(times) / n
            variance = sum((t - avg) ** 2 for t in times) / n
            result["pace_consistency"] = round((variance ** 0.5) / avg, 4) if avg else None

    # Positions/gaps at 2f from finish and halfway
    # sector_number is 0-based (0=first furlong, n-1=last furlong)
    # "2f from finish" = sector_number == total_furlongs - 2 - 1 (0-based) or just take second-to-last
    if total_furlongs >= 2:
        target_2f   = total_furlongs - 2  # 0-based sector number
        target_half = total_furlongs // 2 - 1
    else:
        # fall back to positions in list
        target_2f   = max(n - 2, 0)
        target_half = n // 2 - 1

    for s in sects_sorted:
        sn = s.get("sector_number", -1)
        if sn == target_2f:
            result["pos_at_2f"]      = s.get("sector_position")
            result["gap_at_2f_secs"] = s.get("time_behind_leader")
        if sn == target_half:
            result["pos_at_halfway"]      = s.get("sector_position")
            result["gap_at_halfway_secs"] = s.get("time_behind_leader")

    # position gain: first sector position -> finish position
    start_pos  = sects_sorted[0].get("sector_position") if sects_sorted else None
    finish_pos = runner.get("finish_position")
    if isinstance(start_pos, int) and isinstance(finish_pos, int):
        result["position_gain"] = finish_pos - start_pos

    return result


def _process_race(con, race_id: int, race_date: str, course: str, off_dt: str, dry_run: bool) -> int:
    course_slug = _course_to_slug(course)
    hhmm = _time_to_hhmm(off_dt)
    if not hhmm:
        logger.warning("Cannot parse HHMM race_id=%s off_dt=%s", race_id, off_dt)
        return 0

    payload = _fetch_sectional(race_date, course_slug, hhmm)
    if not payload:
        time.sleep(_DELAY_ERR + random.uniform(0.5, 2.0))
        return 0

    # Correct path: payload["race"]["runners"]
    race_data = payload.get("race") or {}
    runners   = race_data.get("runners") or []
    total_furlongs = len(
        {s.get("sector_number") for r in runners for s in (r.get("sectionals") or [])}
    )

    if not runners:
        time.sleep(_DELAY_OK + random.uniform(0.5, 2.0))
        return 0

    result_rows = con.execute(
        "SELECT result_id, horse_name FROM results WHERE race_id = ?", [race_id]
    ).fetchall()
    result_map = {r[1].strip().lower(): r[0] for r in result_rows}

    written = 0
    for runner in runners:
        horse_name   = (runner.get("horse_name") or "").strip()
        if not horse_name:
            continue
        api_horse_id = str(runner.get("id") or "").strip() or None
        result_id    = result_map.get(horse_name.lower())
        metrics      = _compute_metrics(runner, total_furlongs)

        if dry_run:
            logger.info("[DRY-RUN] race_id=%s horse=%s fsp=%s pos_at_2f=%s",
                        race_id, horse_name,
                        metrics["finishing_speed_pct"], metrics["pos_at_2f"])
            written += 1
            continue

        try:
            new_id = con.execute("SELECT nextval('seq_sectional_id')").fetchone()[0]
            con.execute(
                """
                INSERT INTO sectionals (
                    sectional_id, race_id, result_id, horse_name, api_horse_id,
                    meeting_date, course,
                    finishing_speed_pct, total_furlongs,
                    pos_at_2f, pos_at_halfway,
                    gap_at_2f_secs, gap_at_halfway_secs,
                    early_split_avg, late_split_avg,
                    pace_consistency, position_gain,
                    splits_json, scraped_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (race_id, horse_name) DO UPDATE SET
                    finishing_speed_pct=excluded.finishing_speed_pct,
                    total_furlongs=excluded.total_furlongs,
                    pos_at_2f=excluded.pos_at_2f,
                    pos_at_halfway=excluded.pos_at_halfway,
                    gap_at_2f_secs=excluded.gap_at_2f_secs,
                    gap_at_halfway_secs=excluded.gap_at_halfway_secs,
                    early_split_avg=excluded.early_split_avg,
                    late_split_avg=excluded.late_split_avg,
                    pace_consistency=excluded.pace_consistency,
                    position_gain=excluded.position_gain,
                    splits_json=excluded.splits_json,
                    scraped_at=excluded.scraped_at
                """,
                [new_id, race_id, result_id, horse_name, api_horse_id,
                 race_date, course,
                 metrics["finishing_speed_pct"], total_furlongs,
                 metrics["pos_at_2f"], metrics["pos_at_halfway"],
                 metrics["gap_at_2f_secs"], metrics["gap_at_halfway_secs"],
                 metrics["early_split_avg"], metrics["late_split_avg"],
                 metrics["pace_consistency"], metrics["position_gain"],
                 metrics["splits_json"], datetime.now()],
            )
            written += 1
        except Exception as exc:
            logger.error("Insert failed race_id=%s horse=%s: %s", race_id, horse_name, exc)

    if written:
        logger.info("  -> %d runners written", written)
    time.sleep(_DELAY_OK + random.uniform(0.5, 2.0))
    return written


def scrape_sectionals(start_date: date, end_date: date, dry_run: bool = False) -> None:
    """
    Scrape sectionals for all settled races in [start_date, end_date].
    LEAKAGE GUARD: end_date capped to yesterday -- future races never touched.
    """
    today    = date.today()
    safe_end = min(end_date, today - timedelta(days=1))
    if safe_end < start_date:
        logger.info("No past races in window -- nothing to do.")
        return
    logger.info("Sectionals scrape: %s -> %s  dry_run=%s", start_date, safe_end, dry_run)
    init_database()
    con = get_connection()
    try:
        races = con.execute(
            """
            SELECT DISTINCT r.race_id,
                CAST(m.meeting_date AS VARCHAR) AS race_date,
                m.course, r.off_dt
            FROM races r
            JOIN meetings m ON r.meeting_id = m.meeting_id
            WHERE m.meeting_date >= ? AND m.meeting_date <= ?
              AND r.race_id IN (SELECT DISTINCT race_id FROM results)
              AND r.race_id NOT IN (SELECT DISTINCT race_id FROM sectionals)
            ORDER BY m.meeting_date, r.off_dt
            """,
            [str(start_date), str(safe_end)],
        ).fetchall()
        logger.info("Races to scrape: %d", len(races))
        total = 0
        for i, (race_id, race_date, course, off_dt) in enumerate(races, 1):
            logger.info("[%d/%d] race_id=%s %s %s %s",
                        i, len(races), race_id, race_date, course, off_dt)
            total += _process_race(con, race_id, race_date, course, off_dt, dry_run)
            if i % 25 == 0:
                logger.info("  CHECKPOINT: %d/%d races done, %d rows saved to DB", i, len(races), total)
        logger.info("Done. Total rows written: %d", total)
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="RacingTV sectional data scraper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=7)
    group.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    today = date.today()
    if args.start:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    else:
        end   = today - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)
    scrape_sectionals(start, end, dry_run=args.dry_run)


if __name__ == "__main__":
    main()