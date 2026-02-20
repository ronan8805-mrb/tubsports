"""
BHA Scraper — British Horseracing Authority.

Scrapes:
  - Official UK handicap ratings (Flat, AWT, Chase, Hurdle)
  - Updated weekly on Tuesday mornings

Source: https://www.britishhorseracing.com/regulation/official-ratings/ratings-database/
Polite: 2s delay, user-agent rotation, idempotent inserts.
"""

import logging
import random
import re
import time
from datetime import date
from typing import Optional, Dict, Any, List

import requests
from bs4 import BeautifulSoup

from ..config import BHA_RATINGS_URL, SCRAPE_DELAY, USER_AGENTS
from ..db import get_connection, insert_rating, upsert_horse

logger = logging.getLogger(__name__)


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
        logger.warning(f"BHA request failed: {url} — {e}")
        return None


def scrape_bha_ratings() -> Dict[str, int]:
    """
    Scrape BHA official ratings database.

    The BHA ratings page uses dynamic loading (Angular/Vue).
    We attempt to scrape the rendered table, or fall back to
    looking for an API endpoint that the frontend uses.

    Returns: {"flat": N, "awt": N, "chase": N, "hurdle": N}
    """
    session = _get_session()
    counts = {"flat": 0, "awt": 0, "chase": 0, "hurdle": 0}
    today = date.today()

    # The BHA ratings page is a dynamic Angular app.
    # It loads data via an internal API. Let's try to find it.
    # Known pattern: the page calls an API like:
    #   /umbraco/api/RatingsApi/GetRatings?searchTerm=...

    api_url = "https://www.britishhorseracing.com/umbraco/api/RatingsApi/GetRatings"

    # Try the API first (much more efficient than scraping)
    resp = _polite_get(session, api_url, params={"searchTerm": "", "pageSize": 5000, "page": 1})
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            counts = _process_bha_api_response(data, today)
            return counts
        except Exception as e:
            logger.warning(f"BHA API parse failed: {e}")

    # Fallback: scrape the HTML page
    resp = _polite_get(session, BHA_RATINGS_URL)
    if not resp:
        return counts

    soup = BeautifulSoup(resp.text, "html.parser")
    counts = _scrape_bha_html(soup, today)
    return counts


def _process_bha_api_response(data: Any, today: date) -> Dict[str, int]:
    """Process BHA API JSON response."""
    counts = {"flat": 0, "awt": 0, "chase": 0, "hurdle": 0}

    con = get_connection()
    try:
        items = data if isinstance(data, list) else data.get("items", data.get("ratings", []))

        for item in items:
            horse_name = item.get("horseName", item.get("horse", "")).strip()
            if not horse_name:
                continue

            trainer_name = item.get("trainerName", item.get("trainer", ""))

            # BHA provides separate ratings per category
            for cat, key in [("flat", "flat"), ("awt", "awt"), ("chase", "chase"), ("hurdle", "hurdle")]:
                rating_val = item.get(f"{cat}Rating", item.get(cat))
                if rating_val is not None:
                    try:
                        rating = int(rating_val)
                        if 30 <= rating <= 200:
                            insert_rating(
                                con, horse_name=horse_name,
                                rating_date=today,
                                rating_type=cat,
                                rating=rating,
                                source="BHA",
                            )
                            counts[cat] += 1
                    except (ValueError, TypeError):
                        continue

            # Also update horse record
            sex = item.get("sex", item.get("sexCode"))
            yof = item.get("yof", item.get("yearOfFoaling"))
            upsert_horse(con, horse_name, sex=sex)

        logger.info(f"BHA API ratings: {counts}")
    finally:
        con.close()

    return counts


def _scrape_bha_html(soup: BeautifulSoup, today: date) -> Dict[str, int]:
    """Scrape BHA ratings from HTML table (fallback)."""
    counts = {"flat": 0, "awt": 0, "chase": 0, "hurdle": 0}

    con = get_connection()
    try:
        rows = soup.select("table tr, .rating-row, li[class*='rating']")

        for row in rows:
            cells = row.select("td, span, div.cell")
            if len(cells) < 4:
                continue

            horse_name = cells[0].get_text(strip=True)
            if not horse_name or horse_name.lower() in ("horse", "horse/trainer"):
                continue

            # BHA table columns: Horse/Trainer, YOF, Sex, Flat, AWT, Chase, Hurdle
            col_map = {
                3: "flat",
                4: "awt",
                5: "chase",
                6: "hurdle",
            }

            for col_idx, cat in col_map.items():
                if col_idx >= len(cells):
                    continue
                text = cells[col_idx].get_text(strip=True)
                try:
                    rating = int(text)
                    if 30 <= rating <= 200:
                        insert_rating(
                            con, horse_name=horse_name,
                            rating_date=today,
                            rating_type=cat,
                            rating=rating,
                            source="BHA",
                        )
                        counts[cat] += 1
                except ValueError:
                    continue

        logger.info(f"BHA HTML ratings: {counts}")
    finally:
        con.close()

    return counts


def scrape_bha_weekly() -> Dict[str, int]:
    """Entry point for weekly BHA ratings update (Tuesdays)."""
    logger.info("Starting BHA weekly ratings scrape...")
    return scrape_bha_ratings()
