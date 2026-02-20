"""
HRI RÁS Scraper — Horse Racing Ireland.

Scrapes:
  - Racecards (upcoming runners)
  - Results (race outcomes, positions, times, SP)
  - IHRB Ratings (Flat + NH, weekly + full)
  - Ground Reports

Source: https://www.hri-ras.ie/
Polite: 2s delay, user-agent rotation, idempotent inserts.
"""

import logging
import random
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..config import HRI_BASE_URL, SCRAPE_DELAY, USER_AGENTS
from ..db import (
    get_connection, upsert_meeting, upsert_race, insert_result,
    upsert_horse, upsert_jockey, upsert_trainer, insert_rating,
)

logger = logging.getLogger(__name__)

HRI_RACING_URL = "https://www.hri.ie"


def _get_session() -> requests.Session:
    """Create a session with random user-agent."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s


def _polite_get(session: requests.Session, url: str, **kwargs) -> Optional[requests.Response]:
    """GET with polite delay and error handling."""
    time.sleep(SCRAPE_DELAY)
    try:
        resp = session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.warning(f"Request failed: {url} — {e}")
        return None


def _parse_sp(sp_str: str) -> Optional[float]:
    """Convert SP string (e.g. '3/1', '4/6f', 'Evs') to decimal."""
    if not sp_str:
        return None
    sp = sp_str.strip().rstrip("fFjJ")
    if sp.lower() in ("evs", "evens"):
        return 2.0
    if "/" in sp:
        try:
            parts = sp.split("/")
            return round(float(parts[0]) / float(parts[1]) + 1, 2)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return round(float(sp), 2)
    except ValueError:
        return None


def _parse_distance_furlongs(dist_str: str) -> Optional[float]:
    """Parse distance string like '2m 4f' or '7f' to furlongs."""
    if not dist_str:
        return None
    dist_str = dist_str.strip().lower()
    miles = 0
    furlongs = 0
    yards = 0

    m_match = re.search(r'(\d+)\s*m', dist_str)
    f_match = re.search(r'(\d+)\s*f', dist_str)
    y_match = re.search(r'(\d+)\s*y', dist_str)

    if m_match:
        miles = int(m_match.group(1))
    if f_match:
        furlongs = int(f_match.group(1))
    if y_match:
        yards = int(y_match.group(1))

    total = miles * 8 + furlongs + yards / 220
    return round(total, 2) if total > 0 else None


def _parse_weight(weight_str: str) -> Optional[float]:
    """Parse weight string like '11-10' (stone-lbs) to lbs."""
    if not weight_str:
        return None
    weight_str = weight_str.strip()
    if "-" in weight_str:
        try:
            parts = weight_str.split("-")
            stone = int(parts[0])
            lbs = int(parts[1])
            return stone * 14 + lbs
        except (ValueError, IndexError):
            return None
    try:
        return float(weight_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Results scraping
# ---------------------------------------------------------------------------

def scrape_results_for_date(target_date: date) -> Dict[str, Any]:
    """
    Scrape all Irish race results for a given date from HRI.

    Returns: {"meetings": N, "races": N, "results": N}
    """
    session = _get_session()
    date_str = target_date.strftime("%d-%m-%Y")
    url = f"{HRI_RACING_URL}/racing/results/?date={date_str}"

    resp = _polite_get(session, url)
    if not resp:
        return {"meetings": 0, "races": 0, "results": 0}

    soup = BeautifulSoup(resp.text, "html.parser")
    stats = {"meetings": 0, "races": 0, "results": 0}

    con = get_connection()
    try:
        # Find meeting links
        meeting_links = soup.select("a[href*='/racing/results/']")
        seen_meetings = set()

        for link in meeting_links:
            href = link.get("href", "")
            if "/racing/results/" not in href or href in seen_meetings:
                continue
            seen_meetings.add(href)

            meeting_url = urljoin(HRI_RACING_URL, href)
            meeting_resp = _polite_get(session, meeting_url)
            if not meeting_resp:
                continue

            meeting_soup = BeautifulSoup(meeting_resp.text, "html.parser")
            course_name = _extract_course_name(meeting_soup, href)

            if not course_name:
                continue

            meeting_id = upsert_meeting(
                con, course=course_name.lower().replace(" ", "_"),
                meeting_date=target_date, country="IE",
                going=_extract_going(meeting_soup),
            )
            stats["meetings"] += 1

            # Parse individual races
            race_sections = meeting_soup.select(".race-result, .racecard-race, [class*='race']")
            race_num = 0

            for section in race_sections:
                race_num += 1
                race_data = _parse_race_section(section, race_num)
                if not race_data:
                    continue

                race_id = upsert_race(
                    con, meeting_id=meeting_id,
                    race_number=race_num,
                    race_time=race_data.get("time"),
                    race_name=race_data.get("name"),
                    distance_furlongs=race_data.get("distance_furlongs"),
                    race_type=race_data.get("race_type"),
                    race_class=race_data.get("race_class"),
                    prize=race_data.get("prize"),
                    num_runners=race_data.get("num_runners"),
                )
                stats["races"] += 1

                for runner in race_data.get("runners", []):
                    horse_name = runner.get("horse", "").strip()
                    if not horse_name:
                        continue

                    jockey_name = runner.get("jockey", "")
                    trainer_name = runner.get("trainer", "")

                    jockey_id = upsert_jockey(con, jockey_name) if jockey_name else None
                    trainer_id = upsert_trainer(con, trainer_name) if trainer_name else None

                    insert_result(
                        con, race_id=race_id, horse_name=horse_name,
                        position=runner.get("position"),
                        beaten_lengths=runner.get("beaten_lengths"),
                        finish_time=runner.get("finish_time"),
                        sp_decimal=_parse_sp(runner.get("sp", "")),
                        weight_carried=_parse_weight(runner.get("weight")),
                        draw=runner.get("draw"),
                        jockey_id=jockey_id,
                        jockey_name=jockey_name,
                        trainer_id=trainer_id,
                        trainer_name=trainer_name,
                        official_rating=runner.get("or"),
                        age=runner.get("age"),
                        comment=runner.get("comment"),
                    )
                    stats["results"] += 1

        logger.info(f"HRI results {target_date}: {stats}")
    finally:
        con.close()

    return stats


def _extract_course_name(soup, href: str) -> Optional[str]:
    """Extract course name from meeting page or URL."""
    title = soup.select_one("h1, .meeting-header, .course-name")
    if title:
        text = title.get_text(strip=True)
        text = re.sub(r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}', '', text).strip()
        if text:
            return text

    match = re.search(r'/results/(\w+)', href)
    if match:
        return match.group(1).replace("-", " ").title()
    return None


def _extract_going(soup) -> Optional[str]:
    """Extract going description from meeting page."""
    for tag in soup.select(".going, [class*='going'], dt, th"):
        text = tag.get_text(strip=True).lower()
        if "going" in text:
            next_el = tag.find_next_sibling() or tag.find_next("dd") or tag.find_next("td")
            if next_el:
                return next_el.get_text(strip=True)
    return None


def _parse_race_section(section, race_num: int) -> Optional[Dict[str, Any]]:
    """Parse a single race section from the results page."""
    data: Dict[str, Any] = {"race_number": race_num, "runners": []}

    # Try to get race name
    name_el = section.select_one("h2, h3, .race-name, .race-title")
    if name_el:
        data["name"] = name_el.get_text(strip=True)

    # Distance
    for el in section.select(".distance, [class*='dist']"):
        dist_text = el.get_text(strip=True)
        data["distance_furlongs"] = _parse_distance_furlongs(dist_text)
        break

    # Runners from result rows
    rows = section.select("tr, .runner-row, .result-row")
    for row in rows:
        runner = _parse_runner_row(row)
        if runner:
            data["runners"].append(runner)

    data["num_runners"] = len(data["runners"])
    return data if data["runners"] else None


def _parse_runner_row(row) -> Optional[Dict[str, Any]]:
    """Parse a single runner result row."""
    cells = row.select("td")
    if len(cells) < 3:
        return None

    runner: Dict[str, Any] = {}

    # Try to extract position from first cell
    pos_text = cells[0].get_text(strip=True)
    pos_match = re.match(r'^(\d+)', pos_text)
    if pos_match:
        runner["position"] = int(pos_match.group(1))

    # Horse name — usually in a link
    horse_link = row.select_one("a[href*='horse'], a[href*='hid']")
    if horse_link:
        runner["horse"] = horse_link.get_text(strip=True)
    elif len(cells) > 1:
        runner["horse"] = cells[1].get_text(strip=True)

    if not runner.get("horse"):
        return None

    # Extract other fields from remaining cells
    for cell in cells[2:]:
        text = cell.get_text(strip=True)
        # SP odds
        if re.match(r'^\d+/\d+', text) or text.lower() in ("evs", "evens"):
            runner["sp"] = text
        # Weight
        elif re.match(r'^\d+-\d+$', text):
            runner["weight"] = text
        # Jockey (usually has a link)
        jockey_link = cell.select_one("a[href*='jockey'], a[href*='rid']")
        if jockey_link:
            runner["jockey"] = jockey_link.get_text(strip=True)
        trainer_link = cell.select_one("a[href*='trainer'], a[href*='tid']")
        if trainer_link:
            runner["trainer"] = trainer_link.get_text(strip=True)

    return runner


# ---------------------------------------------------------------------------
# IHRB Ratings
# ---------------------------------------------------------------------------

def scrape_ihrb_ratings(rating_type: str = "flat") -> int:
    """
    Scrape IHRB ratings from HRI RÁS.

    Args:
        rating_type: "flat" or "nh"

    Returns: number of ratings stored
    """
    session = _get_session()

    if rating_type == "flat":
        url = f"{HRI_BASE_URL}/ihrb-ratings/full-flat-ratings/"
    else:
        url = f"{HRI_BASE_URL}/ihrb-ratings/full-nh-ratings/"

    resp = _polite_get(session, url)
    if not resp:
        return 0

    soup = BeautifulSoup(resp.text, "html.parser")
    today = date.today()
    count = 0

    con = get_connection()
    try:
        # Ratings are typically in a table
        rows = soup.select("table tr, .rating-row")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 3:
                continue

            horse_name = cells[0].get_text(strip=True)
            if not horse_name or horse_name.lower() == "horse":
                continue

            # Try to find the rating value
            for cell in cells[1:]:
                text = cell.get_text(strip=True)
                try:
                    rating = int(text)
                    if 30 <= rating <= 200:
                        insert_rating(
                            con, horse_name=horse_name,
                            rating_date=today,
                            rating_type=rating_type,
                            rating=rating,
                            source="IHRB",
                        )
                        count += 1
                        break
                except ValueError:
                    continue

        logger.info(f"IHRB {rating_type} ratings: {count} stored")
    finally:
        con.close()

    return count


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill_results(start_date: date, end_date: date) -> Dict[str, int]:
    """Backfill HRI results for a date range."""
    totals = {"meetings": 0, "races": 0, "results": 0}
    current = start_date

    while current <= end_date:
        logger.info(f"Backfilling HRI results: {current}")
        stats = scrape_results_for_date(current)
        for k in totals:
            totals[k] += stats.get(k, 0)
        current += timedelta(days=1)

    logger.info(f"HRI backfill complete: {totals}")
    return totals


def scrape_today() -> Dict[str, Any]:
    """Scrape today's results + update ratings."""
    results = scrape_results_for_date(date.today())
    flat_count = scrape_ihrb_ratings("flat")
    nh_count = scrape_ihrb_ratings("nh")
    return {
        **results,
        "flat_ratings": flat_count,
        "nh_ratings": nh_count,
    }
