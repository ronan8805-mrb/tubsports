"""
Open-Meteo Weather Scraper — Free API, no key needed.

Fetches historical and forecast weather data for each course's
GPS coordinates. Data stored per (course, date).

API: https://api.open-meteo.com/v1/forecast
     https://archive-api.open-meteo.com/v1/archive

Parameters fetched:
  - temperature_2m (°C)
  - relative_humidity_2m (%)
  - precipitation (mm)
  - wind_speed_10m (km/h)
  - wind_direction_10m (degrees)
  - surface_pressure (hPa)
"""

import logging
import time
from datetime import date, timedelta
from typing import Optional, Dict, Any

import requests

from ..config import COURSES, SCRAPE_DELAY
from ..db import get_connection

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_for_course(
    course: str, target_date: date
) -> Optional[Dict[str, Any]]:
    """
    Fetch weather data for a course on a specific date.

    Uses forecast API for recent/future dates, archive API for historical.
    Returns dict with weather fields or None.
    """
    course_info = COURSES.get(course.lower())
    if not course_info:
        logger.debug(f"No GPS coords for course: {course}")
        return None

    lat = course_info["lat"]
    lon = course_info["lon"]
    date_str = target_date.isoformat()

    # Use archive for historical, forecast for recent/future
    days_ago = (date.today() - target_date).days
    if days_ago > 7:
        api_url = ARCHIVE_URL
    else:
        api_url = FORECAST_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant,surface_pressure_mean",
        "timezone": "Europe/London",
    }

    try:
        time.sleep(0.5)  # Polite — Open-Meteo is free
        resp = requests.get(api_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Weather fetch failed for {course} {date_str}: {e}")
        return None

    daily = data.get("daily", {})
    if not daily:
        return None

    def _first(key):
        vals = daily.get(key, [None])
        return vals[0] if vals else None

    return {
        "course": course,
        "date": target_date,
        "temperature": _first("temperature_2m_mean"),
        "humidity": _first("relative_humidity_2m_mean"),
        "precipitation": _first("precipitation_sum"),
        "wind_speed": _first("wind_speed_10m_max"),
        "wind_direction": _first("wind_direction_10m_dominant"),
        "pressure": _first("surface_pressure_mean"),
    }


def store_weather(course: str, target_date: date) -> bool:
    """Fetch and store weather for a course/date. Skips if already exists."""
    con = get_connection()
    try:
        existing = con.execute(
            "SELECT weather_id FROM weather WHERE course = ? AND weather_date = ?",
            [course, target_date]
        ).fetchone()
        if existing:
            return True

        weather = fetch_weather_for_course(course, target_date)
        if not weather:
            return False

        wid = con.execute("SELECT nextval('seq_weather_id')").fetchone()[0]
        con.execute(
            """INSERT INTO weather (weather_id, course, weather_date,
               temperature, humidity, precipitation,
               wind_speed, wind_direction, pressure)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [wid, course, target_date,
             weather["temperature"], weather["humidity"],
             weather["precipitation"], weather["wind_speed"],
             weather["wind_direction"], weather["pressure"]]
        )
        return True
    except Exception as e:
        logger.warning(f"Weather store failed: {course} {target_date} — {e}")
        return False
    finally:
        con.close()


def backfill_weather_for_meetings() -> int:
    """
    Fetch weather for all meetings that don't have weather data yet.
    Queries the meetings table and fills gaps.
    """
    con = get_connection(read_only=True)
    try:
        rows = con.execute("""
            SELECT DISTINCT m.course, m.meeting_date
            FROM meetings m
            LEFT JOIN weather w ON m.course = w.course AND m.meeting_date = w.weather_date
            WHERE w.weather_id IS NULL
            ORDER BY m.meeting_date DESC
            LIMIT 500
        """).fetchall()
    finally:
        con.close()

    count = 0
    for course, meeting_date in rows:
        if store_weather(course, meeting_date):
            count += 1
            if count % 50 == 0:
                logger.info(f"Weather backfill: {count}/{len(rows)}")

    logger.info(f"Weather backfill complete: {count} records stored")
    return count


def fetch_upcoming_weather(days_ahead: int = 3) -> int:
    """
    Fetch weather forecasts for all known courses for the next N days.
    Good for pre-race prediction.
    """
    count = 0
    today = date.today()

    for course in COURSES:
        for offset in range(days_ahead + 1):
            target = today + timedelta(days=offset)
            if store_weather(course, target):
                count += 1

    logger.info(f"Upcoming weather: {count} records for {len(COURSES)} courses x {days_ahead + 1} days")
    return count
