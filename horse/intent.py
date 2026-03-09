"""
Trainer intent signal detection.
Combines multiple signals to identify when a trainer is targeting a specific race.
"""

import logging
import math
from typing import Dict, Optional

import numpy as np

from .config import COURSES

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_intent_features(
    con,
    api_trainer_id: str,
    course: str,
    race_date,
    race_class_numeric: float,
    last_race_class_numeric: float,
    has_headgear: bool,
    last_had_headgear: bool,
    current_jockey_id: str,
    api_horse_id: str,
) -> Dict[str, float]:
    """Compute trainer intent signals.

    Returns:
        intent_score: composite 0-1 signal
        jockey_upgrade: 1 if jockey is better than horse's usual
        class_drop: 1 if dropping in class
        first_time_headgear: 1 if headgear added this run
        trainer_travel_km: distance from trainer's typical courses
    """
    feats = {}
    signals = 0.0
    max_signals = 5.0

    # --- Class drop ---
    if not np.isnan(race_class_numeric) and not np.isnan(last_race_class_numeric):
        class_diff = last_race_class_numeric - race_class_numeric
        feats["class_drop"] = 1.0 if class_diff > 0 else 0.0
        feats["class_drop_magnitude"] = max(class_diff, 0.0)
        feats["class_rise"] = 1.0 if class_diff < 0 else 0.0
        if class_diff > 0:
            signals += min(class_diff, 2.0) / 2.0
    else:
        feats["class_drop"] = np.nan
        feats["class_drop_magnitude"] = np.nan
        feats["class_rise"] = np.nan

    # --- First-time headgear ---
    feats["first_time_headgear"] = 1.0 if (has_headgear and not last_had_headgear) else 0.0
    if feats["first_time_headgear"]:
        signals += 1.0

    # --- Jockey upgrade ---
    jockey_upgrade = 0.0
    if api_trainer_id and current_jockey_id and api_horse_id:
        try:
            cutoff = str(race_date)
            usual = con.execute("""
                SELECT res.api_jockey_id, j.win_rate
                FROM results res
                JOIN jockeys j ON res.jockey_id = j.jockey_id
                WHERE res.api_horse_id = ? AND res.api_jockey_id IS NOT NULL
                ORDER BY res.result_id DESC
                LIMIT 3
            """, [api_horse_id]).fetchall()

            if usual:
                avg_usual_wr = np.mean([r[1] for r in usual if r[1] and r[1] > 0] or [0])
                current_wr = con.execute("""
                    SELECT win_rate FROM jockeys WHERE api_id = ?
                """, [current_jockey_id]).fetchone()
                if current_wr and current_wr[0]:
                    if current_wr[0] > avg_usual_wr * 1.3 and avg_usual_wr > 0:
                        jockey_upgrade = 1.0
                        signals += 1.0
        except Exception:
            pass
    feats["jockey_upgrade"] = jockey_upgrade

    # --- Travel distance ---
    travel_km = np.nan
    if api_trainer_id and course:
        try:
            course_lower = course.lower().replace(" ", "_").split("(")[0].strip().rstrip("_")
            target_info = COURSES.get(course_lower)
            if target_info:
                target_lat = target_info["lat"]
                target_lon = target_info["lon"]

                trainer_courses = con.execute("""
                    SELECT m.course, COUNT(*) as cnt
                    FROM results res
                    JOIN races ra ON res.race_id = ra.race_id
                    JOIN meetings m ON ra.meeting_id = m.meeting_id
                    WHERE res.api_trainer_id = ?
                    GROUP BY m.course
                    ORDER BY cnt DESC
                    LIMIT 5
                """, [api_trainer_id]).fetchall()

                if trainer_courses:
                    home_course = trainer_courses[0][0].lower().replace(" ", "_").split("(")[0].strip().rstrip("_")
                    home_info = COURSES.get(home_course)
                    if home_info:
                        travel_km = _haversine_km(
                            home_info["lat"], home_info["lon"],
                            target_lat, target_lon,
                        )
                        if travel_km > 200:
                            signals += 0.5
        except Exception:
            pass
    feats["trainer_travel_km"] = travel_km

    # --- Composite intent score ---
    feats["intent_score"] = min(signals / max_signals, 1.0)

    return feats
