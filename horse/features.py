"""
Feature engineering for UK/IRE Horse Racing predictions.
Completely separate from greyhound -- no shared imports.

Builds ~120 ML features from horse.duckdb tables:
  - results (167K+ runners)
  - horse_form (500K+ career entries)
  - races, meetings, courses
  - sire_stats, dam_stats

All features computed from data available BEFORE the race (no leakage).
Rolling windows use strict `< race_date` cutoffs.
"""

import logging
import re
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import duckdb

from .config import (
    GOING_ENCODE, RACE_TYPES, FORM_WINDOW, SHORT_FORM_WINDOW,
    MAX_DAYS_LOOKBACK,
)
from .db import get_connection

logger = logging.getLogger(__name__)


def _safe_num(val, default=None):
    """Safely convert a value that might be pd.NA/None/NaN to a Python number or default."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

# ===================================================================
# Constants
# ===================================================================

# Features that must NEVER be used for training (determined at/after race time)
LEAKAGE_COLS = frozenset({
    "position", "sp_decimal", "betfair_sp", "beaten_lengths", "ovr_btn",
    "finish_time", "comment", "prize", "tote_win", "tote_pl", "tote_ex",
    "tote_csf", "tote_tricast", "tote_trifecta", "winning_time_detail",
    "silk_url", "raw_json",
    "rpr", "tsr",
})

# ID / metadata columns (not features)
ID_COLS = [
    "race_id", "result_id", "api_horse_id", "horse_name",
    "meeting_date", "off_dt", "course",
]

TARGET_COL = "position"

# ===================================================================
# Encoding helpers
# ===================================================================

def encode_going(going_str: Optional[str]) -> float:
    """Convert going description to numeric (higher = softer ground)."""
    if not going_str:
        return 3.0  # default to 'good'
    g = going_str.strip().lower()
    if g in GOING_ENCODE:
        return float(GOING_ENCODE[g])
    # Partial match
    for key, val in GOING_ENCODE.items():
        if key in g:
            return float(val)
    return 3.0


def encode_race_class(class_str: Optional[str]) -> float:
    """Convert race_class VARCHAR to numeric (lower = better).
    'Group 1' -> 0.5, 'Class 1' -> 1, 'Class 7' -> 7, etc."""
    if not class_str:
        return 5.0  # default mid-class
    c = class_str.strip().lower()
    # Group races (top tier)
    if "group 1" in c or "grade 1" in c:
        return 0.5
    if "group 2" in c or "grade 2" in c:
        return 0.8
    if "group 3" in c or "grade 3" in c:
        return 1.0
    if "listed" in c:
        return 1.5
    # Class races
    m = re.search(r'class\s*(\d)', c)
    if m:
        return float(m.group(1))
    # Novice / maiden / other
    if "novice" in c:
        return 5.0
    if "maiden" in c:
        return 6.0
    if "handicap" in c:
        return 4.0
    return 5.0


def encode_race_type(type_str: Optional[str]) -> int:
    """Convert race type to numeric. flat=0, hurdle=1, chase=2, nh_flat=3."""
    if not type_str:
        return 0
    t = type_str.strip().lower()
    if "flat" in t and "hunt" not in t:
        return 0
    if "hurdle" in t:
        return 1
    if "chase" in t:
        return 2
    if "bumper" in t or "nhf" in t or "national hunt flat" in t:
        return 3
    mapped = RACE_TYPES.get(t)
    if mapped == "nh":
        return 1
    return 0


def encode_surface(surface_str: Optional[str]) -> int:
    """turf=0, all-weather=1."""
    if not surface_str:
        return 0
    return 1 if "aw" in surface_str.lower() or "all" in surface_str.lower() else 0


# ===================================================================
# Form feature computation (from horse_form table)
# ===================================================================

def _compute_form_features(form_df: pd.DataFrame, race_date,
                           distance_f: float, going_str: str,
                           course: str, race_type_str: str,
                           race_class_str: str) -> Dict[str, float]:
    """Compute form features for a single horse from their career history.

    form_df: horse_form rows for this horse, sorted by race_date ASC,
             filtered to race_date < current race date.
    """
    feats = {}

    n = len(form_df)
    feats["career_starts"] = float(n)
    feats["is_debutant"] = 1.0 if n == 0 else 0.0

    if n == 0:
        # Fill with defaults for debutants
        for key in [
            "form_avg_pos_6", "form_avg_pos_3", "form_win_rate",
            "form_place_rate", "form_momentum", "form_consistency",
            "form_best_pos", "form_worst_pos",
            "days_since_last_race", "races_last_30d", "races_last_90d",
            "dist_match_runs", "dist_match_win_rate", "dist_match_place_rate",
            "going_match_runs", "going_match_win_rate",
            "course_match_runs", "course_match_win_rate",
            "type_match_win_rate", "class_match_win_rate",
            "best_or_6", "avg_or_6", "or_trend",
            "best_rpr_6", "avg_rpr_6", "rpr_trend",
            "avg_beaten_lengths", "form_weight_avg",
            "best_speed_last_6", "avg_speed_last_3", "speed_trend",
        ]:
            feats[key] = np.nan
        return feats

    # Recent form (last FORM_WINDOW races)
    recent = form_df.tail(FORM_WINDOW)
    short = form_df.tail(SHORT_FORM_WINDOW)
    older = form_df.tail(FORM_WINDOW).head(FORM_WINDOW - SHORT_FORM_WINDOW)

    positions = recent["position"].dropna()
    short_pos = short["position"].dropna()
    older_pos = older["position"].dropna()

    feats["form_avg_pos_6"] = positions.mean() if len(positions) > 0 else np.nan
    feats["form_avg_pos_3"] = short_pos.mean() if len(short_pos) > 0 else np.nan
    feats["form_win_rate"] = (positions == 1).mean() if len(positions) > 0 else 0.0
    feats["form_place_rate"] = (positions <= 3).mean() if len(positions) > 0 else 0.0
    feats["form_consistency"] = positions.std() if len(positions) > 1 else np.nan
    feats["form_best_pos"] = positions.min() if len(positions) > 0 else np.nan
    feats["form_worst_pos"] = positions.max() if len(positions) > 0 else np.nan

    # Momentum: lower = improving (recent avg pos < older avg pos)
    if len(short_pos) > 0 and len(older_pos) > 0:
        feats["form_momentum"] = short_pos.mean() - older_pos.mean()
    else:
        feats["form_momentum"] = np.nan

    # Freshness
    last_date = form_df["race_date"].iloc[-1]
    if pd.notna(last_date):
        try:
            delta = (pd.Timestamp(race_date) - pd.Timestamp(last_date)).days
            feats["days_since_last_race"] = float(max(delta, 0))
        except Exception:
            feats["days_since_last_race"] = np.nan
    else:
        feats["days_since_last_race"] = np.nan

    cutoff_30 = pd.Timestamp(race_date) - pd.Timedelta(days=30)
    cutoff_90 = pd.Timestamp(race_date) - pd.Timedelta(days=90)
    dates = pd.to_datetime(form_df["race_date"])
    feats["races_last_30d"] = float((dates >= cutoff_30).sum())
    feats["races_last_90d"] = float((dates >= cutoff_90).sum())

    # --- Distance match ---
    distance_f = _safe_num(distance_f)
    if distance_f and distance_f > 0:
        dist_mask = (form_df["distance_furlongs"] - distance_f).abs() <= 1.0
        dist_rows = form_df[dist_mask]
        feats["dist_match_runs"] = float(len(dist_rows))
        dist_pos = dist_rows["position"].dropna()
        feats["dist_match_win_rate"] = (dist_pos == 1).mean() if len(dist_pos) > 0 else np.nan
        feats["dist_match_place_rate"] = (dist_pos <= 3).mean() if len(dist_pos) > 0 else np.nan
    else:
        feats["dist_match_runs"] = np.nan
        feats["dist_match_win_rate"] = np.nan
        feats["dist_match_place_rate"] = np.nan

    # --- Going match ---
    going_num = encode_going(going_str)
    form_going_nums = form_df["going"].apply(encode_going)
    going_mask = (form_going_nums - going_num).abs() <= 1.0
    going_rows = form_df[going_mask]
    feats["going_match_runs"] = float(len(going_rows))
    going_pos = going_rows["position"].dropna()
    feats["going_match_win_rate"] = (going_pos == 1).mean() if len(going_pos) > 0 else np.nan

    # --- Course match ---
    if course:
        course_rows = form_df[form_df["course"].str.lower() == course.lower()]
        feats["course_match_runs"] = float(len(course_rows))
        cpos = course_rows["position"].dropna()
        feats["course_match_win_rate"] = (cpos == 1).mean() if len(cpos) > 0 else np.nan
    else:
        feats["course_match_runs"] = np.nan
        feats["course_match_win_rate"] = np.nan

    # --- Race type match ---
    race_type_num = encode_race_type(race_type_str)
    type_nums = form_df["race_type"].apply(encode_race_type)
    type_rows = form_df[type_nums == race_type_num]
    type_pos = type_rows["position"].dropna()
    feats["type_match_win_rate"] = (type_pos == 1).mean() if len(type_pos) > 0 else np.nan

    # --- Class match ---
    class_num = encode_race_class(race_class_str)
    class_nums = form_df["race_class"].apply(encode_race_class)
    class_mask = (class_nums - class_num).abs() <= 1.0
    class_rows = form_df[class_mask]
    class_pos = class_rows["position"].dropna()
    feats["class_match_win_rate"] = (class_pos == 1).mean() if len(class_pos) > 0 else np.nan

    # --- Rating features (from horse_form official_rating) ---
    or_vals = recent["official_rating"].dropna()
    feats["best_or_6"] = or_vals.max() if len(or_vals) > 0 else np.nan
    feats["avg_or_6"] = or_vals.mean() if len(or_vals) > 0 else np.nan

    short_or = short["official_rating"].dropna()
    older_or = older["official_rating"].dropna()
    if len(short_or) > 0 and len(older_or) > 0:
        feats["or_trend"] = short_or.mean() - older_or.mean()
    else:
        feats["or_trend"] = np.nan

    # RPR from results (not in horse_form, use sp_decimal as proxy if needed)
    feats["best_rpr_6"] = np.nan
    feats["avg_rpr_6"] = np.nan
    feats["rpr_trend"] = np.nan

    # --- Beaten lengths ---
    btn = recent["beaten_lengths"].dropna()
    feats["avg_beaten_lengths"] = btn.mean() if len(btn) > 0 else np.nan

    # --- Weight ---
    feats["form_weight_avg"] = np.nan  # weight_carried is VARCHAR in form

    # --- Speed figures (from pre-computed speed_figure column) ---
    if "speed_figure" in form_df.columns:
        sf_recent = recent["speed_figure"].dropna()
        sf_short = short["speed_figure"].dropna()
        sf_older = older["speed_figure"].dropna()

        feats["best_speed_last_6"] = sf_recent.max() if len(sf_recent) > 0 else np.nan
        feats["avg_speed_last_3"] = sf_short.mean() if len(sf_short) > 0 else np.nan

        if len(sf_short) > 0 and len(sf_older) > 0:
            feats["speed_trend"] = sf_short.mean() - sf_older.mean()
        else:
            feats["speed_trend"] = np.nan
    else:
        feats["best_speed_last_6"] = np.nan
        feats["avg_speed_last_3"] = np.nan
        feats["speed_trend"] = np.nan

    return feats


# ===================================================================
# Jockey/Trainer features (from results table)
# ===================================================================

def _compute_jt_features(con: duckdb.DuckDBPyConnection,
                         api_jockey_id: str, api_trainer_id: str,
                         course: str, race_date) -> Dict[str, float]:
    """Compute jockey and trainer stats from historical results."""
    feats = {}
    cutoff = str(race_date)
    cutoff_365 = str(pd.Timestamp(race_date) - pd.Timedelta(days=365))

    # --- Jockey stats (last 365 days) ---
    if api_jockey_id:
        jrow = con.execute("""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS places
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.api_jockey_id = ?
              AND m.meeting_date >= ? AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
        """, [api_jockey_id, cutoff_365, cutoff]).fetchone()

        runs, wins, places = jrow if jrow else (0, 0, 0)
        runs = runs or 0
        feats["jockey_runs_365d"] = float(runs)
        feats["jockey_win_rate_365d"] = wins / runs if runs > 0 else np.nan
        feats["jockey_place_rate_365d"] = places / runs if runs > 0 else np.nan

        # Jockey at this course
        if course:
            jc = con.execute("""
                SELECT COUNT(*) AS runs,
                       SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
                FROM results res
                JOIN races ra ON res.race_id = ra.race_id
                JOIN meetings m ON ra.meeting_id = m.meeting_id
                WHERE res.api_jockey_id = ?
                  AND m.course = ? AND m.meeting_date < ?
                  AND res.position IS NOT NULL AND res.position > 0
            """, [api_jockey_id, course, cutoff]).fetchone()
            cr, cw = jc if jc else (0, 0)
            cr = cr or 0
            feats["jockey_course_runs"] = float(cr)
            feats["jockey_course_win_rate"] = cw / cr if cr > 0 else np.nan
        else:
            feats["jockey_course_runs"] = np.nan
            feats["jockey_course_win_rate"] = np.nan
    else:
        feats["jockey_runs_365d"] = np.nan
        feats["jockey_win_rate_365d"] = np.nan
        feats["jockey_place_rate_365d"] = np.nan
        feats["jockey_course_runs"] = np.nan
        feats["jockey_course_win_rate"] = np.nan

    # --- Trainer stats (last 365 days) ---
    if api_trainer_id:
        trow = con.execute("""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS places
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.api_trainer_id = ?
              AND m.meeting_date >= ? AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
        """, [api_trainer_id, cutoff_365, cutoff]).fetchone()

        runs, wins, places = trow if trow else (0, 0, 0)
        runs = runs or 0
        feats["trainer_runs_365d"] = float(runs)
        feats["trainer_win_rate_365d"] = wins / runs if runs > 0 else np.nan
        feats["trainer_place_rate_365d"] = places / runs if runs > 0 else np.nan

        if course:
            tc = con.execute("""
                SELECT COUNT(*) AS runs,
                       SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
                FROM results res
                JOIN races ra ON res.race_id = ra.race_id
                JOIN meetings m ON ra.meeting_id = m.meeting_id
                WHERE res.api_trainer_id = ?
                  AND m.course = ? AND m.meeting_date < ?
                  AND res.position IS NOT NULL AND res.position > 0
            """, [api_trainer_id, course, cutoff]).fetchone()
            cr, cw = tc if tc else (0, 0)
            cr = cr or 0
            feats["trainer_course_runs"] = float(cr)
            feats["trainer_course_win_rate"] = cw / cr if cr > 0 else np.nan
        else:
            feats["trainer_course_runs"] = np.nan
            feats["trainer_course_win_rate"] = np.nan
    else:
        feats["trainer_runs_365d"] = np.nan
        feats["trainer_win_rate_365d"] = np.nan
        feats["trainer_place_rate_365d"] = np.nan
        feats["trainer_course_runs"] = np.nan
        feats["trainer_course_win_rate"] = np.nan

    # --- Jockey-Trainer combo ---
    if api_jockey_id and api_trainer_id:
        combo = con.execute("""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.api_jockey_id = ? AND res.api_trainer_id = ?
              AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
        """, [api_jockey_id, api_trainer_id, cutoff]).fetchone()
        cr, cw = combo if combo else (0, 0)
        cr = cr or 0
        feats["jt_combo_runs"] = float(cr)
        feats["jt_combo_win_rate"] = cw / cr if cr > 0 else np.nan
    else:
        feats["jt_combo_runs"] = np.nan
        feats["jt_combo_win_rate"] = np.nan

    return feats


# ===================================================================
# Draw bias (computed from historical results at course)
# ===================================================================

def _compute_draw_features(con: duckdb.DuckDBPyConnection,
                           draw, course: str,
                           num_runners: int,
                           race_date) -> Dict[str, float]:
    """Draw-related features."""
    feats = {}
    draw_val = _safe_num(draw)
    nr = _safe_num(num_runners, 0)
    feats["draw"] = draw_val if draw_val and draw_val > 0 else np.nan
    feats["draw_pct_of_field"] = (draw_val / nr) if draw_val and nr and nr > 0 else np.nan

    if draw_val and draw_val > 0 and course:
        row = con.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.draw = ? AND m.course = ?
              AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
        """, [int(draw_val), course, str(race_date)]).fetchone()
        total, wins = row if row else (0, 0)
        total = total or 0
        feats["draw_hist_win_rate"] = wins / total if total >= 20 else np.nan
    else:
        feats["draw_hist_win_rate"] = np.nan

    return feats


# ===================================================================
# Main feature builder
# ===================================================================

def build_features_for_races(con: duckdb.DuckDBPyConnection,
                             race_ids: List[int]) -> pd.DataFrame:
    """Build feature vectors for all runners in the given races.
    Returns DataFrame with one row per runner."""
    if not race_ids:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(race_ids))

    # Fetch runners + race context
    runners = con.execute(f"""
        SELECT
            res.result_id, res.race_id, res.horse_name,
            res.api_horse_id, res.position, res.draw,
            res.rpr, res.tsr, res.official_rating,
            res.weight_lbs, res.age, res.sex,
            res.ovr_btn, res.jockey_claim_lbs,
            res.api_jockey_id, res.api_trainer_id,
            res.api_sire_id, res.api_dam_id,
            ra.off_dt, ra.distance_furlongs, ra.distance_yards,
            ra.race_type, ra.race_class, ra.going_description,
            ra.surface, ra.handicap_flag, ra.num_runners,
            ra.region_code,
            m.meeting_date, m.course, m.going AS meeting_going,
            c.direction AS course_direction
        FROM results res
        JOIN races ra ON res.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        LEFT JOIN courses c ON m.course = c.course
        WHERE res.race_id IN ({placeholders})
        ORDER BY res.race_id, res.draw
    """, race_ids).df()

    if runners.empty:
        return pd.DataFrame()

    # Batch-fetch weather data for all courses/dates in this batch
    weather_keys = runners[["course", "meeting_date"]].drop_duplicates()
    _weather_cache = {}
    for _, wk in weather_keys.iterrows():
        try:
            wrow = con.execute("""
                SELECT temperature, humidity, precipitation,
                       wind_speed, wind_direction, pressure
                FROM weather
                WHERE course = ? AND weather_date = ?
            """, [wk["course"], wk["meeting_date"]]).fetchone()
            if wrow:
                _weather_cache[(wk["course"], str(wk["meeting_date"]))] = {
                    "weather_temp": wrow[0],
                    "weather_humidity": wrow[1],
                    "weather_precipitation": wrow[2],
                    "weather_wind_speed": wrow[3],
                    "weather_wind_direction": wrow[4],
                    "weather_pressure": wrow[5],
                }
        except Exception:
            pass

    all_rows = []

    for _, row in runners.iterrows():
        feat = {}

        # --- ID columns (kept for grouping, not for training) ---
        feat["race_id"] = row["race_id"]
        feat["result_id"] = row["result_id"]
        feat["api_horse_id"] = row["api_horse_id"]
        feat["horse_name"] = row["horse_name"]
        feat["meeting_date"] = row["meeting_date"]
        feat["off_dt"] = row["off_dt"]
        feat["course"] = row["course"]

        # --- Target ---
        feat["position"] = row["position"]

        # --- Race context features ---
        _df = _safe_num(row["distance_furlongs"])
        _dy = _safe_num(row["distance_yards"])
        feat["distance_furlongs"] = _df if _df else (_dy / 220.0 if _dy else np.nan)
        feat["going_numeric"] = encode_going(
            row["going_description"] or row["meeting_going"]
        )
        feat["race_class_numeric"] = encode_race_class(row["race_class"])
        feat["race_type_encoded"] = encode_race_type(row["race_type"])
        feat["surface_encoded"] = encode_surface(row["surface"])
        feat["num_runners"] = _safe_num(row["num_runners"], 0)
        feat["handicap_flag"] = 1.0 if pd.notna(row["handicap_flag"]) and row["handicap_flag"] else 0.0
        feat["course_direction_encoded"] = (
            1.0 if row.get("course_direction") and "right" in str(row["course_direction"]).lower()
            else 0.0
        )

        # --- Runner-level features (from results row) ---
        # NOTE: rpr and tsr are POST-RACE ratings (leakage) -- excluded
        feat["official_rating"] = float(row["official_rating"]) if pd.notna(row["official_rating"]) else np.nan
        feat["weight_lbs"] = float(row["weight_lbs"]) if pd.notna(row["weight_lbs"]) else np.nan
        feat["age"] = float(row["age"]) if pd.notna(row["age"]) else np.nan
        feat["sex_encoded"] = 1.0 if row.get("sex") and "f" in str(row["sex"]).lower() else 0.0
        feat["jockey_claim_lbs"] = float(row["jockey_claim_lbs"]) if pd.notna(row["jockey_claim_lbs"]) else 0.0

        # --- Horse form features (from horse_form table) ---
        race_date = row["meeting_date"]
        api_horse_id = row["api_horse_id"]
        if api_horse_id and race_date:
            lookback = str(pd.Timestamp(race_date) - pd.Timedelta(days=MAX_DAYS_LOOKBACK))
            form_df = con.execute("""
                SELECT race_date, course, distance_furlongs, race_type,
                       race_class, going, position, beaten_lengths,
                       official_rating, sp_decimal, draw, num_runners,
                       speed_figure
                FROM horse_form
                WHERE api_horse_id = ? AND race_date < ? AND race_date >= ?
                ORDER BY race_date ASC
            """, [api_horse_id, str(race_date), lookback]).df()

            form_feats = _compute_form_features(
                form_df, race_date,
                feat["distance_furlongs"],
                row["going_description"] or row["meeting_going"] or "",
                row["course"] or "",
                row["race_type"] or "",
                row["race_class"] or "",
            )
            feat.update(form_feats)
        else:
            feat["is_debutant"] = 1.0
            feat["career_starts"] = 0.0

        # --- Jockey / Trainer features ---
        jt_feats = _compute_jt_features(
            con, row["api_jockey_id"], row["api_trainer_id"],
            row["course"], race_date,
        )
        feat.update(jt_feats)

        # --- Draw features ---
        draw_feats = _compute_draw_features(
            con, row["draw"], row["course"],
            _safe_num(row["num_runners"], 0), race_date,
        )
        feat.update(draw_feats)

        # --- Weather features (per course/date from Open-Meteo) ---
        w_key = (row["course"], str(row["meeting_date"]))
        w_data = _weather_cache.get(w_key)
        if w_data:
            feat["weather_temp"] = w_data["weather_temp"]
            feat["weather_precipitation"] = w_data["weather_precipitation"]
            feat["weather_wind_speed"] = w_data["weather_wind_speed"]
            feat["weather_humidity"] = w_data["weather_humidity"]
            feat["weather_pressure"] = w_data["weather_pressure"]
            precip = w_data["weather_precipitation"]
            going = feat.get("going_numeric", 3.0)
            feat["precip_x_going"] = (precip or 0) * (going or 3.0)
        else:
            feat["weather_temp"] = np.nan
            feat["weather_precipitation"] = np.nan
            feat["weather_wind_speed"] = np.nan
            feat["weather_humidity"] = np.nan
            feat["weather_pressure"] = np.nan
            feat["precip_x_going"] = np.nan

        all_rows.append(feat)

    df = pd.DataFrame(all_rows)

    # --- Field-relative features (computed per race) ---
    for race_id, group in df.groupby("race_id"):
        idx = group.index
        # OR rank in field
        if group["official_rating"].notna().sum() > 0:
            df.loc[idx, "or_rank_in_field"] = group["official_rating"].rank(
                ascending=False, method="min"
            )
            field_avg_or = group["official_rating"].mean()
            df.loc[idx, "field_avg_or"] = field_avg_or
            df.loc[idx, "or_vs_field_avg"] = group["official_rating"] - field_avg_or
        else:
            df.loc[idx, "or_rank_in_field"] = np.nan
            df.loc[idx, "field_avg_or"] = np.nan
            df.loc[idx, "or_vs_field_avg"] = np.nan

        # Weight relative to field median
        if group["weight_lbs"].notna().sum() > 0:
            median_wt = group["weight_lbs"].median()
            df.loc[idx, "weight_diff_from_median"] = group["weight_lbs"] - median_wt
        else:
            df.loc[idx, "weight_diff_from_median"] = np.nan

        # Speed relative to field
        if "avg_speed_last_3" in group.columns and group["avg_speed_last_3"].notna().sum() > 0:
            field_avg_speed = group["avg_speed_last_3"].mean()
            df.loc[idx, "speed_vs_field_avg"] = group["avg_speed_last_3"] - field_avg_speed
            df.loc[idx, "speed_rank_in_field"] = group["avg_speed_last_3"].rank(
                ascending=False, method="min"
            )
        else:
            df.loc[idx, "speed_vs_field_avg"] = np.nan
            df.loc[idx, "speed_rank_in_field"] = np.nan

    # --- Interaction features ---
    df["or_vs_class"] = df["official_rating"] / df["race_class_numeric"].replace(0, np.nan)
    df["form_x_freshness"] = df["form_win_rate"] * (
        1.0 / np.log1p(df["days_since_last_race"].fillna(365))
    )
    df["jockey_x_trainer"] = df["jockey_win_rate_365d"] * df["trainer_win_rate_365d"]

    logger.info(
        f"Built features: {len(df)} runners, "
        f"{len([c for c in df.columns if c not in ID_COLS and c != TARGET_COL])} features"
    )

    return df


def build_training_dataset(con: duckdb.DuckDBPyConnection,
                           start_date: str = None,
                           end_date: str = None,
                           batch_size: int = 200) -> pd.DataFrame:
    """Build features for all races in date range. Returns full DataFrame."""
    if not start_date:
        start_date = "2025-02-01"
    if not end_date:
        end_date = str(date.today())

    race_ids = con.execute("""
        SELECT ra.race_id
        FROM races ra
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE m.meeting_date >= ? AND m.meeting_date <= ?
          AND ra.num_runners >= 2
        ORDER BY m.meeting_date
    """, [start_date, end_date]).fetchall()
    race_ids = [r[0] for r in race_ids]

    logger.info(f"Building features for {len(race_ids)} races ({start_date} to {end_date})")

    all_dfs = []
    for i in range(0, len(race_ids), batch_size):
        batch = race_ids[i:i + batch_size]
        batch_df = build_features_for_races(con, batch)
        if not batch_df.empty:
            all_dfs.append(batch_df)
        if (i // batch_size + 1) % 10 == 0:
            total = sum(len(d) for d in all_dfs)
            logger.info(f"  Progress: {i + len(batch)}/{len(race_ids)} races, {total:,} runners")

    if not all_dfs:
        logger.warning("No features built -- empty dataset")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Training dataset: {len(result):,} runners, {len(result.columns)} columns")
    return result


# ===================================================================
# Feature column lists (for model training)
# ===================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature columns suitable for ML training."""
    exclude = set(ID_COLS) | LEAKAGE_COLS | {TARGET_COL, "sp_decimal", "betfair_sp"}
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ("float64", "float32", "int64", "int32", "Float64", "Int64")
    ]
