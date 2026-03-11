"""
Feature engineering for UK/IRE Horse Racing predictions.

Builds ~130 ML features from horse.duckdb tables:
  - results, horse_form, races, meetings, courses
  - sire_stats, dam_stats, damsire_stats, sire_distance_stats
  - ratings_history, market_data, weather

All features computed from data available BEFORE the race (no leakage).
Rolling windows use strict `< race_date` cutoffs.
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import duckdb

from .config import (
    GOING_ENCODE, RACE_TYPES, FORM_WINDOW, SHORT_FORM_WINDOW,
    MAX_DAYS_LOOKBACK, TRAIN_YEARS, COURSES,
)
from .nlp import compute_comment_features, compute_run_style_features

logger = logging.getLogger(__name__)

# ===================================================================
# Module-level worker for ProcessPoolExecutor (must be pickleable)
# ===================================================================
_DB_PATH_FOR_WORKER = str(Path(__file__).parent / "data" / "horse.duckdb")


def _build_features_worker(batch_race_ids):
    """Worker function for parallel feature building. Opens its own DB connection."""
    import duckdb as _ddb
    _con = _ddb.connect(_DB_PATH_FOR_WORKER, read_only=True)
    try:
        return build_features_for_races(_con, batch_race_ids)
    except Exception as _e:
        return None
    finally:
        _con.close()


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

    if n == 0:
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
            "avg_beaten_lengths",
            "best_speed_last_6", "avg_speed_last_3", "speed_trend",
            "implied_prob_avg_6", "implied_prob_avg_3", "best_sp_last_6",
            "sp_momentum", "sp_consistency", "sp_vs_result",
            "comment_sentiment_avg", "comment_sentiment_trend",
            "trip_trouble_count", "positive_comment_count",
            "negative_comment_count", "had_excuse_count",
            "run_style", "last_race_class",
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
    n_runs = len(positions)
    feats["form_win_rate"] = (float((positions == 1).sum()) + 2) / (n_runs + 5) if n_runs > 0 else 0.0
    feats["form_place_rate"] = (float((positions <= 3).sum()) + 2) / (n_runs + 5) if n_runs > 0 else 0.0
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
        n_dist = len(dist_pos)
        feats["dist_match_win_rate"] = (float((dist_pos == 1).sum()) + 2) / (n_dist + 5) if n_dist > 0 else np.nan
        feats["dist_match_place_rate"] = (float((dist_pos <= 3).sum()) + 2) / (n_dist + 5) if n_dist > 0 else np.nan
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
    n_going = len(going_pos)
    feats["going_match_win_rate"] = (float((going_pos == 1).sum()) + 2) / (n_going + 5) if n_going > 0 else np.nan

    # --- Course match ---
    if course:
        course_rows = form_df[form_df["course"].str.lower() == course.lower()]
        feats["course_match_runs"] = float(len(course_rows))
        cpos = course_rows["position"].dropna()
        n_course = len(cpos)
        feats["course_match_win_rate"] = (float((cpos == 1).sum()) + 2) / (n_course + 5) if n_course > 0 else np.nan
    else:
        feats["course_match_runs"] = np.nan
        feats["course_match_win_rate"] = np.nan

    # --- Race type match ---
    race_type_num = encode_race_type(race_type_str)
    type_nums = form_df["race_type"].apply(encode_race_type)
    type_rows = form_df[type_nums == race_type_num]
    type_pos = type_rows["position"].dropna()
    n_type = len(type_pos)
    feats["type_match_win_rate"] = (float((type_pos == 1).sum()) + 2) / (n_type + 5) if n_type > 0 else np.nan

    # --- Class match ---
    class_num = encode_race_class(race_class_str)
    class_nums = form_df["race_class"].apply(encode_race_class)
    class_mask = (class_nums - class_num).abs() <= 1.0
    class_rows = form_df[class_mask]
    class_pos = class_rows["position"].dropna()
    n_class = len(class_pos)
    feats["class_match_win_rate"] = (float((class_pos == 1).sum()) + 2) / (n_class + 5) if n_class > 0 else np.nan

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

    # --- Beaten lengths ---
    btn = recent["beaten_lengths"].dropna()
    feats["avg_beaten_lengths"] = btn.mean() if len(btn) > 0 else np.nan

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

    # --- Market / SP features (from horse_form.sp_decimal) ---
    if "sp_decimal" in form_df.columns:
        sp_recent = recent["sp_decimal"].dropna()
        sp_short = short["sp_decimal"].dropna()
        sp_older = older["sp_decimal"].dropna()

        valid_sp = sp_recent[sp_recent > 1.0]
        valid_sp_short = sp_short[sp_short > 1.0]
        valid_sp_older = sp_older[sp_older > 1.0]

        if len(valid_sp) > 0:
            implied = 1.0 / valid_sp
            feats["implied_prob_avg_6"] = implied.mean()
            feats["best_sp_last_6"] = valid_sp.min()
        else:
            feats["implied_prob_avg_6"] = np.nan
            feats["best_sp_last_6"] = np.nan

        if len(valid_sp_short) > 0:
            feats["implied_prob_avg_3"] = (1.0 / valid_sp_short).mean()
        else:
            feats["implied_prob_avg_3"] = np.nan

        if len(valid_sp_short) > 0 and len(valid_sp_older) > 0:
            feats["sp_momentum"] = (1.0 / valid_sp_short).mean() - (1.0 / valid_sp_older).mean()
        else:
            feats["sp_momentum"] = np.nan

        all_sp = form_df["sp_decimal"].dropna()
        valid_all_sp = all_sp[all_sp > 1.0]
        if len(valid_all_sp) >= 3:
            feats["sp_consistency"] = (1.0 / valid_all_sp).std()
        else:
            feats["sp_consistency"] = np.nan

        sp_pos_pairs = recent[["sp_decimal", "position"]].dropna()
        sp_pos_pairs = sp_pos_pairs[sp_pos_pairs["sp_decimal"] > 1.0]
        if len(sp_pos_pairs) >= 3:
            implied_rank = sp_pos_pairs["sp_decimal"].rank(method="min")
            actual_rank = sp_pos_pairs["position"].rank(method="min")
            feats["sp_vs_result"] = (implied_rank - actual_rank).mean()
        else:
            feats["sp_vs_result"] = np.nan
    else:
        feats["implied_prob_avg_6"] = np.nan
        feats["implied_prob_avg_3"] = np.nan
        feats["best_sp_last_6"] = np.nan
        feats["sp_momentum"] = np.nan
        feats["sp_consistency"] = np.nan
        feats["sp_vs_result"] = np.nan

    # --- NLP comment features ---
    if "comment" in form_df.columns:
        recent_comments = recent["comment"].tolist()
        comment_feats = compute_comment_features(recent_comments)
        feats.update(comment_feats)

        all_comments = form_df["comment"].tolist()
        all_positions = form_df["position"].tolist()
        style_feats = compute_run_style_features(all_comments, all_positions)
        feats.update(style_feats)
    else:
        feats["comment_sentiment_avg"] = np.nan
        feats["comment_sentiment_trend"] = np.nan
        feats["trip_trouble_count"] = np.nan
        feats["positive_comment_count"] = np.nan
        feats["negative_comment_count"] = np.nan
        feats["had_excuse_count"] = np.nan
        feats["run_style"] = np.nan

    # --- Class movement (last race class vs current) ---
    if n > 0 and "race_class" in form_df.columns:
        last_class_str = form_df["race_class"].iloc[-1]
        feats["last_race_class"] = encode_race_class(last_class_str)
    else:
        feats["last_race_class"] = np.nan

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
    cutoff_14 = str(pd.Timestamp(race_date) - pd.Timedelta(days=14))
    cutoff_30 = str(pd.Timestamp(race_date) - pd.Timedelta(days=30))

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

        # Trainer short-term form (14 days)
        t14 = con.execute("""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.api_trainer_id = ?
              AND m.meeting_date >= ? AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
        """, [api_trainer_id, cutoff_14, cutoff]).fetchone()
        r14, w14 = t14 if t14 else (0, 0)
        r14 = r14 or 0
        feats["trainer_14day_runs"] = float(r14)
        feats["trainer_14day_win_rate"] = w14 / r14 if r14 > 0 else np.nan

        # Trainer 30-day level-stakes ROI (backing every runner at SP)
        t30 = con.execute("""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN res.position = 1 THEN res.sp_decimal ELSE 0 END) AS returns
            FROM results res
            JOIN races ra ON res.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE res.api_trainer_id = ?
              AND m.meeting_date >= ? AND m.meeting_date < ?
              AND res.position IS NOT NULL AND res.position > 0
              AND res.sp_decimal IS NOT NULL AND res.sp_decimal > 0
        """, [api_trainer_id, cutoff_30, cutoff]).fetchone()
        r30, ret30 = t30 if t30 else (0, 0)
        r30 = r30 or 0
        ret30 = ret30 or 0.0
        feats["trainer_30day_roi"] = (ret30 / r30 - 1.0) if r30 > 0 else np.nan

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
        feats["trainer_14day_runs"] = np.nan
        feats["trainer_14day_win_rate"] = np.nan
        feats["trainer_30day_roi"] = np.nan
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
            res.headgear,
            res.api_jockey_id, res.api_trainer_id,
            res.api_sire_id, res.api_dam_id, res.api_damsire_id,
            ra.off_dt, ra.distance_furlongs, ra.distance_yards,
            ra.race_type, ra.race_class, ra.going_description,
            ra.surface, ra.num_runners,
            ra.region_code, ra.rating_band,
            m.meeting_date, m.course, m.going AS meeting_going
        FROM results res
        JOIN races ra ON res.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
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

    # Batch-fetch breeding data (sire/dam/damsire stats)
    _sire_cache = {}
    _sire_dist_cache = {}
    _dam_cache = {}
    _damsire_cache = {}
    sire_ids = runners["api_sire_id"].dropna().unique().tolist()
    dam_ids = runners["api_dam_id"].dropna().unique().tolist()
    damsire_ids = runners["api_damsire_id"].dropna().unique().tolist() if "api_damsire_id" in runners.columns else []

    if sire_ids:
        ph = ",".join(["?"] * len(sire_ids))
        try:
            sire_rows = con.execute(f"""
                SELECT api_sire_id, progeny_win_rate, progeny_count,
                       avg_distance_furlongs, going_preference,
                       flat_win_rate, nh_win_rate
                FROM sire_stats WHERE api_sire_id IN ({ph})
            """, sire_ids).fetchall()
            for sr in sire_rows:
                _sire_cache[sr[0]] = {
                    "sire_win_rate": sr[1], "sire_progeny_count": sr[2],
                    "sire_avg_distance": sr[3], "sire_going_pref": sr[4],
                    "sire_flat_wr": sr[5], "sire_nh_wr": sr[6],
                }
            sire_dist_rows = con.execute(f"""
                SELECT api_sire_id, distance_f, win_rate, runners
                FROM sire_distance_stats WHERE api_sire_id IN ({ph})
            """, sire_ids).fetchall()
            for sdr in sire_dist_rows:
                _sire_dist_cache.setdefault(sdr[0], []).append(
                    {"dist_f": sdr[1], "win_rate": sdr[2], "runners": sdr[3]}
                )
        except Exception:
            pass

    if dam_ids:
        ph = ",".join(["?"] * len(dam_ids))
        try:
            dam_rows = con.execute(f"""
                SELECT api_dam_id, progeny_win_rate, progeny_count,
                       avg_distance_furlongs
                FROM dam_stats WHERE api_dam_id IN ({ph})
            """, dam_ids).fetchall()
            for dr in dam_rows:
                _dam_cache[dr[0]] = {
                    "dam_win_rate": dr[1], "dam_progeny_count": dr[2],
                    "dam_avg_distance": dr[3],
                }
        except Exception:
            pass

    if damsire_ids:
        ph = ",".join(["?"] * len(damsire_ids))
        try:
            ds_rows = con.execute(f"""
                SELECT api_damsire_id, grandoffspring_win_rate
                FROM damsire_stats WHERE api_damsire_id IN ({ph})
            """, damsire_ids).fetchall()
            for dsr in ds_rows:
                _damsire_cache[dsr[0]] = {"damsire_win_rate": dsr[1]}
        except Exception:
            pass

    # Batch-fetch ALL headgear + ratings history for this batch upfront (fast)
    # Filter to each horse's race_date in Python â€” avoids N*2 individual SQL calls
    horse_ids_for_hg = runners["api_horse_id"].dropna().unique().tolist()
    _headgear_history = {}   # horse_id -> [(date_str, headgear), ...] sorted date DESC
    _ratings_history_map = {}  # horse_id -> [(date_str, rating), ...] sorted date DESC
    _headgear_cache = {}     # (horse_id, race_date) -> prev headgear (populated lazily from _headgear_history)
    _ratings_cache = {}      # (horse_id, race_date) -> [{date, rating}, ...] (populated lazily)

    if horse_ids_for_hg:
        ph_hg = ",".join(["?"] * len(horse_ids_for_hg))
        try:
            hg_rows = con.execute(f"""
                SELECT res.api_horse_id, CAST(m.meeting_date AS VARCHAR), res.headgear
                FROM results res
                JOIN races ra ON res.race_id = ra.race_id
                JOIN meetings m ON ra.meeting_id = m.meeting_id
                WHERE res.api_horse_id IN ({ph_hg})
                  AND res.headgear IS NOT NULL AND res.headgear != ''
                ORDER BY res.api_horse_id, m.meeting_date DESC
            """, horse_ids_for_hg).fetchall()
            for hid, hdate, hg in hg_rows:
                if hid not in _headgear_history:
                    _headgear_history[hid] = []
                _headgear_history[hid].append((hdate, hg))
        except Exception:
            pass

        try:
            rat_rows = con.execute(f"""
                SELECT h.api_horse_id, CAST(rh.rating_date AS VARCHAR), rh.rating
                FROM ratings_history rh
                JOIN horses h ON rh.horse_id = h.horse_id
                WHERE h.api_horse_id IN ({ph_hg})
                ORDER BY h.api_horse_id, rh.rating_date DESC
            """, horse_ids_for_hg).fetchall()
            for hid, rdate, rating in rat_rows:
                if hid not in _ratings_history_map:
                    _ratings_history_map[hid] = []
                _ratings_history_map[hid].append((rdate, rating))
        except Exception:
            pass

    # Batch-fetch BSP history for all horses in this batch
    _bsp_history = {}  # horse_id -> [(date_str, betfair_sp, sp_decimal), ...] sorted DESC
    if horse_ids_for_hg:
        try:
            ph_bsp = ",".join(["?"] * len(horse_ids_for_hg))
            bsp_all = con.execute(f"""
                SELECT res.api_horse_id, CAST(m.meeting_date AS VARCHAR),
                       res.betfair_sp, res.sp_decimal
                FROM results res
                JOIN races ra ON res.race_id = ra.race_id
                JOIN meetings m ON ra.meeting_id = m.meeting_id
                WHERE res.api_horse_id IN ({ph_bsp})
                  AND res.betfair_sp IS NOT NULL AND res.betfair_sp > 1.0
                ORDER BY res.api_horse_id, m.meeting_date DESC
            """, horse_ids_for_hg).fetchall()
            for hid, hdate, bsp, sp in bsp_all:
                if hid not in _bsp_history:
                    _bsp_history[hid] = []
                _bsp_history[hid].append((hdate, bsp, sp))
        except Exception:
            pass

    # Batch-fetch ALL horse form history for this batch upfront (fast)
    # Avoids ~2000 individual SQL calls per batch
    _form_history = {}  # horse_id -> DataFrame of form rows sorted ASC by race_date
    if horse_ids_for_hg:
        try:
            ph_form = ",".join(["?"] * len(horse_ids_for_hg))
            all_form_rows = con.execute(f"""
                SELECT api_horse_id, race_date, course, distance_furlongs, race_type,
                       race_class, going, position, beaten_lengths,
                       official_rating, sp_decimal, draw, num_runners,
                       speed_figure, comment
                FROM horse_form
                WHERE api_horse_id IN ({ph_form})
                ORDER BY api_horse_id, race_date ASC
            """, horse_ids_for_hg).df()
            if not all_form_rows.empty:
                for hid, grp in all_form_rows.groupby("api_horse_id"):
                    _form_history[hid] = grp.drop(columns=["api_horse_id"]).reset_index(drop=True)
        except Exception:
            pass

    # Batch-fetch course profiles from DB (enriched with draw_bias, pace_bias)
    _course_db_cache = {}
    try:
        course_rows = con.execute("""
            SELECT course, draw_bias_score, pace_bias, direction
            FROM courses
            WHERE draw_bias_score IS NOT NULL OR pace_bias IS NOT NULL
        """).fetchall()
        for cr in course_rows:
            _course_db_cache[cr[0].lower()] = {
                "draw_bias_score": cr[1] or 0.0,
                "pace_bias": cr[2] or "neutral",
                "direction": cr[3],
            }
    except Exception:
        pass

    # Batch-fetch market data
    _market_cache = {}
    try:
        market_rows = con.execute(f"""
            SELECT race_id, horse_name, morning_price, back_odds, lay_odds,
                   market_move_pct, steam_flag, exchange_volume
            FROM market_data WHERE race_id IN ({",".join(["?"] * len(race_ids))})
        """, race_ids).fetchall()
        for mr in market_rows:
            _market_cache[(mr[0], mr[1])] = {
                "morning_price": mr[2], "back_odds": mr[3], "lay_odds": mr[4],
                "market_move_pct": mr[5], "steam_flag": mr[6],
                "exchange_volume": mr[7],
            }
    except Exception:
        pass

    # ---------------------------------------------------------------------------
    # Batch-fetch sectional history (RacingTV GPS data)
    # ---------------------------------------------------------------------------
    # LEAKAGE GUARD 1: SQL excludes ALL race_ids in this batch -- current races
    #                  can never appear in a horse's own history.
    # LEAKAGE GUARD 2: Keyed by api_horse_id first, horse_name second.
    #                  Prevents same-name-different-horse confusion.
    # LEAKAGE GUARD 3: Per-runner filter enforces date < race_date (strict less-than).
    #                  Same-day sectional data is always excluded.
    _sectionals_history = {}  # api_horse_id -> list of dicts sorted DESC by meeting_date
    _sect_name_history  = {}  # horse_name.lower() -> list (fallback when no api_horse_id)
    try:
        ph_sect = ",".join(["?"] * len(race_ids))
        sect_rows = con.execute(f"""
            SELECT
                s.api_horse_id,
                LOWER(TRIM(s.horse_name))        AS hname,
                CAST(s.meeting_date AS VARCHAR)  AS sect_date,
                s.race_id                        AS sect_race_id,
                s.course                         AS sect_course,
                s.finishing_speed_pct,
                s.pos_at_2f,
                s.pos_at_halfway,
                s.gap_at_2f_secs,
                s.gap_at_halfway_secs,
                s.pace_consistency,
                s.position_gain
            FROM sectionals s
            WHERE s.race_id NOT IN ({ph_sect})
            ORDER BY s.api_horse_id, s.meeting_date DESC
        """, race_ids).fetchall()
        for (ahid, hname, sdate, sect_race_id, sect_course,
             fsp, p2f, phalf, g2f, ghalf, pace_cons, pos_gain) in sect_rows:
            row_dict = {
                "date": sdate,
                "race_id": sect_race_id,
                "course": sect_course,
                "finishing_speed_pct": fsp,
                "pos_at_2f": p2f,
                "pos_at_halfway": phalf,
                "gap_at_2f_secs": g2f,
                "gap_at_halfway_secs": ghalf,
                "pace_consistency": pace_cons,
                "position_gain": pos_gain,
            }
            if ahid:
                if ahid not in _sectionals_history:
                    _sectionals_history[ahid] = []
                _sectionals_history[ahid].append(row_dict)
            if hname:
                if hname not in _sect_name_history:
                    _sect_name_history[hname] = []
                _sect_name_history[hname].append(row_dict)
    except Exception:
        _sectionals_history = {}
        _sect_name_history  = {}

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

        # --- Runner-level features (from results row) ---
        # NOTE: rpr and tsr are POST-RACE ratings (leakage) -- excluded
        feat["official_rating"] = float(row["official_rating"]) if pd.notna(row["official_rating"]) else np.nan
        feat["weight_lbs"] = float(row["weight_lbs"]) if pd.notna(row["weight_lbs"]) else np.nan
        feat["age"] = float(row["age"]) if pd.notna(row["age"]) else np.nan
        feat["sex_encoded"] = 1.0 if row.get("sex") and "f" in str(row["sex"]).lower() else 0.0
        feat["jockey_claim_lbs"] = float(row["jockey_claim_lbs"]) if pd.notna(row["jockey_claim_lbs"]) else 0.0

        # --- Breeding features (from batch-fetched sire/dam/damsire stats) ---
        sire_id = row.get("api_sire_id")
        dam_id = row.get("api_dam_id")
        damsire_id = row.get("api_damsire_id")
        s_data = _sire_cache.get(sire_id, {})
        d_data = _dam_cache.get(dam_id, {})
        ds_data = _damsire_cache.get(damsire_id, {})

        feat["sire_win_rate"] = s_data.get("sire_win_rate", np.nan)
        feat["sire_progeny_count"] = float(s_data.get("sire_progeny_count", 0)) if s_data.get("sire_progeny_count") else np.nan
        feat["dam_win_rate"] = d_data.get("dam_win_rate", np.nan)
        feat["damsire_win_rate"] = ds_data.get("damsire_win_rate", np.nan)

        sire_avg_dist = s_data.get("sire_avg_distance")
        dist_f = feat.get("distance_furlongs")
        if sire_avg_dist and dist_f and not np.isnan(dist_f):
            feat["breeding_distance_fit"] = abs(sire_avg_dist - dist_f)
        else:
            feat["breeding_distance_fit"] = np.nan

        sire_dist_entries = _sire_dist_cache.get(sire_id, [])
        feat["sire_dist_match_win_rate"] = np.nan
        if sire_dist_entries and dist_f and not np.isnan(dist_f):
            best_match = None
            best_diff = 999
            for sde in sire_dist_entries:
                if sde["dist_f"] and sde["runners"] and sde["runners"] >= 5:
                    diff = abs(sde["dist_f"] - dist_f)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = sde
            if best_match and best_diff <= 2.0:
                feat["sire_dist_match_win_rate"] = best_match["win_rate"]

        race_type_num = feat["race_type_encoded"]
        sire_flat = s_data.get("sire_flat_wr")
        sire_nh = s_data.get("sire_nh_wr")
        if sire_flat is not None and sire_nh is not None:
            feat["sire_type_match_wr"] = sire_flat if race_type_num == 0 else sire_nh
        else:
            feat["sire_type_match_wr"] = np.nan

        # --- Race date (needed by headgear/ratings date-filtered lookups below) ---
        race_date = row["meeting_date"]
        api_horse_id = row["api_horse_id"]

        # --- Headgear features (per-runner with date filter to avoid leakage) ---
        current_hg = row.get("headgear")
        has_headgear = bool(current_hg and str(current_hg).strip())
        feat["has_headgear"] = 1.0 if has_headgear else 0.0

        hg_key = (api_horse_id, str(race_date))
        if hg_key not in _headgear_cache:
            race_date_str = str(race_date)
            hg_hist = _headgear_history.get(api_horse_id, [])
            prev_hg_found = next((hg for d, hg in hg_hist if d < race_date_str), None)
            _headgear_cache[hg_key] = prev_hg_found

        prev_hg = _headgear_cache.get(hg_key)
        had_prev_headgear = bool(prev_hg and str(prev_hg).strip())
        feat["first_time_headgear"] = 1.0 if (has_headgear and not had_prev_headgear) else 0.0
        feat["headgear_changed"] = 1.0 if (has_headgear and had_prev_headgear and str(current_hg).strip().lower() != str(prev_hg).strip().lower()) else 0.0

        # --- Course profile features (from enriched DB data) ---
        course_name_raw = row.get("course") or ""
        course_db = _course_db_cache.get(course_name_raw.lower(), {})
        feat["course_draw_bias"] = course_db.get("draw_bias_score", np.nan)
        pace_str = course_db.get("pace_bias", "neutral")
        feat["course_pace_bias"] = 1.0 if pace_str == "front" else (0.0 if pace_str == "hold_up" else 0.5)

        # --- Ratings history features (date-filtered via Python lookup) ---
        rat_key = (api_horse_id, str(race_date))
        if rat_key not in _ratings_cache:
            race_date_str = str(race_date)
            all_ratings = _ratings_history_map.get(api_horse_id, [])
            filtered = [{"date": d, "rating": r} for d, r in all_ratings if d < race_date_str][:10]
            _ratings_cache[rat_key] = filtered

        rat_entries = _ratings_cache.get(rat_key, [])
        if rat_entries:
            ratings_vals = [r["rating"] for r in rat_entries if r["rating"]]
            if ratings_vals:
                or_val_for_peak = row.get("official_rating")
                feat["rating_peak_diff"] = float(or_val_for_peak) - max(ratings_vals) if pd.notna(or_val_for_peak) else np.nan
                recent_90 = [r["rating"] for r in rat_entries[:3] if r["rating"]]
                older_90 = [r["rating"] for r in rat_entries[3:6] if r["rating"]]
                if recent_90 and older_90:
                    feat["rating_trend_90d"] = np.mean(recent_90) - np.mean(older_90)
                else:
                    feat["rating_trend_90d"] = np.nan
            else:
                feat["rating_peak_diff"] = np.nan
                feat["rating_trend_90d"] = np.nan
        else:
            feat["rating_peak_diff"] = np.nan
            feat["rating_trend_90d"] = np.nan

        # --- Market data features (from batch-fetched market_data) ---
        m_key = (row["race_id"], row["horse_name"])
        m_data = _market_cache.get(m_key, {})
        feat["morning_price"] = m_data.get("morning_price", np.nan)
        feat["back_odds"] = m_data.get("back_odds", np.nan)
        feat["lay_odds"] = m_data.get("lay_odds", np.nan)
        feat["market_move_pct"] = m_data.get("market_move_pct", np.nan)
        feat["steam_flag"] = 1.0 if m_data.get("steam_flag") else 0.0
        feat["exchange_volume"] = m_data.get("exchange_volume", np.nan)

        bo = m_data.get("back_odds")
        lo = m_data.get("lay_odds")
        if bo and lo and bo > 1.0 and lo > 1.0:
            feat["back_lay_spread"] = lo - bo
        else:
            feat["back_lay_spread"] = np.nan

        # --- Horse form features (from pre-fetched batch data â€” no SQL per runner) ---
        if api_horse_id and race_date:
            race_date_str = str(race_date)
            lookback_ts = pd.Timestamp(race_date) - pd.Timedelta(days=MAX_DAYS_LOOKBACK)
            lookback_str = str(lookback_ts.date())
            all_form = _form_history.get(api_horse_id)
            if all_form is not None and not all_form.empty:
                rd_col = all_form["race_date"].astype(str)
                form_df = all_form[(rd_col < race_date_str) & (rd_col >= lookback_str)].copy()
            else:
                form_df = pd.DataFrame()

            form_feats = _compute_form_features(
                form_df, race_date,
                feat["distance_furlongs"],
                row["going_description"] or row["meeting_going"] or "",
                row["course"] or "",
                row["race_type"] or "",
                row["race_class"] or "",
            )
            feat.update(form_feats)

            # Class movement: current race class vs last race class from form
            last_rc = form_feats.get("last_race_class", np.nan)
            current_rc = feat["race_class_numeric"]
            if not np.isnan(last_rc) and not np.isnan(current_rc):
                class_diff = last_rc - current_rc
                feat["class_drop"] = 1.0 if class_diff > 0 else 0.0
                feat["class_drop_magnitude"] = max(class_diff, 0.0)
                feat["class_rise"] = 1.0 if class_diff < 0 else 0.0
            else:
                feat["class_drop"] = np.nan
                feat["class_drop_magnitude"] = np.nan
                feat["class_rise"] = np.nan
        else:
            feat["career_starts"] = 0.0
            feat["class_drop"] = np.nan
            feat["class_drop_magnitude"] = np.nan
            feat["class_rise"] = np.nan

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

        # --- Trainer intent features ---
        try:
            from .intent import compute_intent_features
            last_rc = feat.get("last_race_class", np.nan)
            current_rc = feat.get("race_class_numeric", np.nan)
            has_hg = feat.get("has_headgear", 0) == 1.0
            last_hg = bool(prev_hg and str(prev_hg).strip()) if row.get("api_horse_id") else False
            intent_feats = compute_intent_features(
                con,
                api_trainer_id=row.get("api_trainer_id", ""),
                course=row.get("course", ""),
                race_date=race_date,
                race_class_numeric=current_rc if current_rc is not None else np.nan,
                last_race_class_numeric=last_rc if last_rc is not None else np.nan,
                has_headgear=has_hg,
                last_had_headgear=last_hg,
                current_jockey_id=row.get("api_jockey_id", ""),
                api_horse_id=row.get("api_horse_id", ""),
            )
            already_computed = {"class_drop", "class_drop_magnitude", "class_rise", "first_time_headgear"}
            for k, v in intent_feats.items():
                if k not in already_computed:
                    feat[k] = v
        except Exception:
            feat["intent_score"] = np.nan
            feat["jockey_upgrade"] = np.nan
            feat["trainer_travel_km"] = np.nan
            feat["first_time_headgear"] = feat.get("first_time_headgear", np.nan)

        # --- BSP features (from pre-fetched batch data â€” no SQL per runner) ---
        bsp_hist = _bsp_history.get(api_horse_id, []) if api_horse_id else []
        race_date_str_bsp = str(race_date)
        bsp_rows = [(bsp, sp) for d, bsp, sp in bsp_hist if d < race_date_str_bsp][:6]
        if bsp_rows:
            bsp_vals = [r[0] for r in bsp_rows if r[0] and r[0] > 1.0]
            sp_vals = [r[1] for r in bsp_rows if r[1] and r[1] > 1.0]
            feat["bsp_implied_prob_avg_3"] = np.mean([1.0 / b for b in bsp_vals[:3]]) if bsp_vals else np.nan
            if bsp_vals and sp_vals:
                ratios = [b / s for b, s in zip(bsp_vals, sp_vals) if s > 1.0]
                feat["bsp_vs_sp"] = np.mean(ratios) if ratios else np.nan
            else:
                feat["bsp_vs_sp"] = np.nan
        else:
            feat["bsp_implied_prob_avg_3"] = np.nan
            feat["bsp_vs_sp"] = np.nan

        # --- Sectional features (RacingTV GPS splits) ---
        # LEAKAGE GUARD A: lookup by api_horse_id first (exact horse identity).
        #                  Fall back to horse_name only if api_horse_id missing.
        #                  Prevents same-name-different-horse confusion.
        # LEAKAGE GUARD B: date < race_date (strict less-than) -- same-day excluded.
        # LEAKAGE GUARD C: race_id != current race_id -- belt-and-braces check.
        _curr_race_id      = row.get("race_id")
        race_date_str_sect = str(race_date)
        _sect_ahid = row.get("api_horse_id")
        if _sect_ahid and _sect_ahid in _sectionals_history:
            sect_hist = _sectionals_history[_sect_ahid]
        else:
            sect_key  = str(row.get("horse_name", "")).strip().lower()
            sect_hist = _sect_name_history.get(sect_key, [])
        past_sect = [
            s for s in sect_hist
            if s["date"] < race_date_str_sect        # strictly before race day
            and s["race_id"] != _curr_race_id        # never the current race
        ][:3]
        if past_sect:
            fsp_vals   = [s["finishing_speed_pct"] for s in past_sect if s["finishing_speed_pct"] is not None]
            p2f_vals   = [s["pos_at_2f"]           for s in past_sect if s["pos_at_2f"] is not None]
            g2f_vals   = [s["gap_at_2f_secs"]      for s in past_sect if s["gap_at_2f_secs"] is not None]
            pgain_vals = [s["position_gain"]        for s in past_sect if s["position_gain"] is not None]
            pace_vals  = [s["pace_consistency"]     for s in past_sect if s["pace_consistency"] is not None]
            feat["sect_fsp_avg3"]              = np.mean(fsp_vals)   if fsp_vals   else np.nan
            feat["sect_early_pos_avg3"]        = np.mean(p2f_vals)   if p2f_vals   else np.nan
            feat["sect_gap_early_avg3"]        = np.mean(g2f_vals)   if g2f_vals   else np.nan
            feat["sect_position_gain_avg3"]    = np.mean(pgain_vals) if pgain_vals else np.nan
            feat["sect_pace_consistency_avg3"] = np.mean(pace_vals)  if pace_vals  else np.nan
            feat["sect_coverage"]              = len(past_sect) / 3.0
        else:
            feat["sect_fsp_avg3"]              = np.nan
            feat["sect_early_pos_avg3"]        = np.nan
            feat["sect_gap_early_avg3"]        = np.nan
            feat["sect_position_gain_avg3"]    = np.nan
            feat["sect_pace_consistency_avg3"] = np.nan
            feat["sect_coverage"]              = 0.0

        # --- Rating vs race band ---
        rating_band = row.get("rating_band")
        or_val = feat.get("official_rating", np.nan)
        if rating_band and pd.notna(or_val):
            try:
                nums = re.findall(r'\d+', str(rating_band))
                if len(nums) >= 2:
                    band_mid = (int(nums[0]) + int(nums[1])) / 2.0
                    feat["rating_vs_race_band"] = or_val - band_mid
                elif len(nums) == 1:
                    feat["rating_vs_race_band"] = or_val - int(nums[0])
                else:
                    feat["rating_vs_race_band"] = np.nan
            except Exception:
                feat["rating_vs_race_band"] = np.nan
        else:
            feat["rating_vs_race_band"] = np.nan

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
            field_std_speed = group["avg_speed_last_3"].std()
            if pd.isna(field_std_speed) or field_std_speed == 0:
                field_std_speed = 1.0
            df.loc[idx, "speed_vs_field_avg"] = group["avg_speed_last_3"] - field_avg_speed
            df.loc[idx, "speed_rank_in_field"] = group["avg_speed_last_3"].rank(
                ascending=False, method="min"
            )
            df.loc[idx, "relative_speed_z"] = (
                (group["avg_speed_last_3"] - field_avg_speed) / field_std_speed
            ).clip(-4, 4)
        else:
            df.loc[idx, "speed_vs_field_avg"] = np.nan
            df.loc[idx, "speed_rank_in_field"] = np.nan
            df.loc[idx, "relative_speed_z"] = np.nan

    # Interaction features using relative speed z-score
    df["relative_speed_x_form"] = df["relative_speed_z"] * df.get("form_win_rate", pd.Series(np.nan, index=df.index))
    df["relative_speed_x_distance"] = df["relative_speed_z"] * df.get("dist_match_win_rate", pd.Series(np.nan, index=df.index))

    # --- Field-relative market features ---
    for race_id, group in df.groupby("race_id"):
        idx = group.index
        if "implied_prob_avg_3" in group.columns and group["implied_prob_avg_3"].notna().sum() > 0:
            field_avg_ip = group["implied_prob_avg_3"].mean()
            df.loc[idx, "market_vs_field"] = group["implied_prob_avg_3"] - field_avg_ip
            df.loc[idx, "market_rank_in_field"] = group["implied_prob_avg_3"].rank(
                ascending=False, method="min"
            )
        else:
            df.loc[idx, "market_vs_field"] = np.nan
            df.loc[idx, "market_rank_in_field"] = np.nan

    # --- Field-relative live market features ---
    if "back_odds" in df.columns:
        for race_id, group in df.groupby("race_id"):
            idx = group.index
            valid_odds = group["back_odds"].dropna()
            valid_odds = valid_odds[valid_odds > 1.0]
            if len(valid_odds) > 0:
                implied = 1.0 / valid_odds
                total_implied = implied.sum()
                fair = implied / total_implied if total_implied > 0 else implied
                df.loc[valid_odds.index, "live_market_prob"] = fair.values
                df.loc[valid_odds.index, "live_market_rank"] = fair.rank(ascending=False, method="min").values
            else:
                df.loc[idx, "live_market_prob"] = np.nan
                df.loc[idx, "live_market_rank"] = np.nan

    # --- Field-relative exchange volume ---
    if "exchange_volume" in df.columns:
        for race_id, group in df.groupby("race_id"):
            idx = group.index
            vol = group["exchange_volume"].dropna()
            if len(vol) > 0:
                df.loc[idx, "exchange_volume_rank"] = group["exchange_volume"].rank(ascending=False, method="min")
            else:
                df.loc[idx, "exchange_volume_rank"] = np.nan

    # --- Pace features (race-level dynamics) ---
    if "run_style" in df.columns:
        for race_id, group in df.groupby("race_id"):
            idx = group.index
            n_front = (group["run_style"] == 1.0).sum()
            n_runners = len(group)
            df.loc[idx, "pace_scenario"] = float(n_front)

            # Early speed rank within the race
            has_sect = "sect_early_pos_avg3" in group.columns and group["sect_early_pos_avg3"].notna().sum() > 0
            if has_sect:
                df.loc[idx, "early_speed_rank"] = group["sect_early_pos_avg3"].rank(
                    ascending=True, method="min"
                )
                pace_leaders = (group["sect_early_pos_avg3"].rank(ascending=True, method="min") <= 3).sum()
                df.loc[idx, "pace_pressure"] = float(pace_leaders)
            else:
                df.loc[idx, "early_speed_rank"] = np.nan
                df.loc[idx, "pace_pressure"] = float(n_front)

            df.loc[idx, "front_runner_count"] = float(n_front)

            # Lead probability: P(this horse leads) from run style + sectionals
            course_pace = group.get("course_pace_bias")
            cpb = course_pace.iloc[0] if course_pace is not None and len(course_pace) > 0 and not pd.isna(course_pace.iloc[0]) else 0.5
            for i_row in idx:
                style = df.at[i_row, "run_style"] if "run_style" in df.columns else np.nan
                early_rank = df.at[i_row, "early_speed_rank"] if has_sect else np.nan
                if pd.isna(style):
                    df.at[i_row, "lead_probability"] = np.nan
                    continue
                base = 0.6 if style == 1.0 else (0.2 if style == 2.0 else 0.05)
                if not pd.isna(early_rank):
                    if early_rank == 1:
                        base = min(base + 0.25, 0.95)
                    elif early_rank <= 3:
                        base = min(base + 0.10, 0.80)
                    else:
                        base = max(base - 0.05, 0.01)
                if n_runners > 0:
                    base *= (1.0 / max(n_front, 1))
                base = max(0.01, min(0.95, base))
                df.at[i_row, "lead_probability"] = base

            # Expected pace categorical: 0=slow, 1=even, 2=fast
            if n_front <= 1:
                df.loc[idx, "expected_pace"] = 0.0
            elif n_front <= 3:
                df.loc[idx, "expected_pace"] = 1.0
            else:
                df.loc[idx, "expected_pace"] = 2.0

            styles = group["run_style"].dropna()
            if len(styles) > 0:
                df.loc[idx, "pace_advantage"] = group["run_style"].apply(
                    lambda s: 1.0 if (n_front >= 3 and s == 3.0) else
                              (1.0 if (n_front <= 1 and s == 1.0) else 0.5)
                    if not pd.isna(s) else np.nan
                )
            else:
                df.loc[idx, "pace_advantage"] = np.nan

    # --- Finish speed ratio (horse FSP / race avg FSP) ---
    if "sect_fsp_avg3" in df.columns:
        for race_id, group in df.groupby("race_id"):
            idx = group.index
            fsp_vals = group["sect_fsp_avg3"].dropna()
            if len(fsp_vals) >= 2:
                race_avg_fsp = fsp_vals.mean()
                df.loc[idx, "finish_speed_ratio"] = (
                    group["sect_fsp_avg3"] / race_avg_fsp
                ).clip(0.5, 2.0)
            else:
                df.loc[idx, "finish_speed_ratio"] = np.nan

    # --- Field strength features ---
    for race_id, group in df.groupby("race_id"):
        idx = group.index
        or_vals = group["official_rating"].dropna()
        if len(or_vals) >= 2:
            df.loc[idx, "field_rating_mean"] = or_vals.mean()
            df.loc[idx, "field_rating_std"] = or_vals.std()
            df.loc[idx, "rating_vs_field_mean"] = group["official_rating"] - or_vals.mean()
        else:
            df.loc[idx, "field_rating_mean"] = np.nan
            df.loc[idx, "field_rating_std"] = np.nan
            df.loc[idx, "rating_vs_field_mean"] = np.nan

    # --- Market entropy (how spread is the betting market) ---
    if "back_odds" in df.columns:
        for race_id, group in df.groupby("race_id"):
            idx = group.index
            valid_odds = group["back_odds"].dropna()
            valid_odds = valid_odds[valid_odds > 1.0]
            if len(valid_odds) >= 2:
                implied = 1.0 / valid_odds
                probs = implied / implied.sum()
                entropy = -(probs * np.log(probs + 1e-10)).sum()
                df.loc[idx, "market_entropy"] = entropy
            else:
                df.loc[idx, "market_entropy"] = np.nan

    # --- Interaction features ---
    df["or_vs_class"] = df["official_rating"] / df["race_class_numeric"].replace(0, np.nan)
    df["form_x_freshness"] = df["form_win_rate"] * (
        1.0 / np.log1p(df["days_since_last_race"].fillna(365).clip(lower=1))
    )
    df["jockey_x_trainer"] = df["jockey_win_rate_365d"] * df["trainer_win_rate_365d"]
    df["market_x_form"] = df["implied_prob_avg_3"].fillna(0) * df["form_win_rate"].fillna(0)
    df["class_x_headgear"] = df.get("class_drop", pd.Series(np.nan, index=df.index)).fillna(0) * df.get("first_time_headgear", pd.Series(np.nan, index=df.index)).fillna(0)
    df["sire_x_distance"] = df.get("sire_dist_match_win_rate", pd.Series(np.nan, index=df.index)).fillna(0) * df.get("dist_match_win_rate", pd.Series(np.nan, index=df.index)).fillna(0)

    # --- High-value interaction features ---
    fwr = df.get("form_win_rate", pd.Series(np.nan, index=df.index))
    dwr = df.get("dist_match_win_rate", pd.Series(np.nan, index=df.index))
    gwr = df.get("going_match_win_rate", pd.Series(np.nan, index=df.index))
    cwr = df.get("course_match_win_rate", pd.Series(np.nan, index=df.index))
    clwr = df.get("class_match_win_rate", pd.Series(np.nan, index=df.index))

    df["form_x_distance"] = fwr * dwr
    df["form_x_going"] = fwr * gwr
    df["course_x_distance"] = cwr * dwr
    df["class_x_form"] = clwr * fwr

    # Sectional interaction features (where available)
    sect_fsp = df.get("sect_fsp_avg3", pd.Series(np.nan, index=df.index))
    sect_early = df.get("sect_early_pos_avg3", pd.Series(np.nan, index=df.index))
    sect_gain = df.get("sect_position_gain_avg3", pd.Series(np.nan, index=df.index))
    sect_pace = df.get("sect_pace_consistency_avg3", pd.Series(np.nan, index=df.index))

    df["finish_strength"] = (sect_fsp / sect_pace.replace(0, np.nan)).clip(-5, 5)
    df["sect_gain_x_form"] = sect_gain * fwr
    df["early_pos_x_distance"] = sect_early * dwr

    # pace_vs_course_bias: does the horse's run style suit this course's pace profile
    if "run_style" in df.columns and "course_pace_bias" in df.columns:
        def _pace_course_match(row):
            style = row.get("run_style")
            bias = row.get("course_pace_bias")
            if pd.isna(style) or pd.isna(bias):
                return np.nan
            if style == 1.0 and bias == 0.0:
                return 1.0
            if style == 3.0 and bias == 1.0:
                return 1.0
            if style == 1.0 and bias == 1.0:
                return 0.0
            if style == 3.0 and bias == 0.0:
                return 0.0
            return 0.5
        df["pace_vs_course_bias"] = df.apply(_pace_course_match, axis=1)

    # intent_x_class: intent signal amplified by class drop
    if "intent_score" in df.columns and "class_drop" in df.columns:
        df["intent_x_class"] = df["intent_score"].fillna(0) * df["class_drop"].fillna(0)

    logger.info(
        f"Built features: {len(df)} runners, "
        f"{len([c for c in df.columns if c not in ID_COLS and c != TARGET_COL])} features"
    )

    return df


FEATURE_MATRIX_PATH = Path(__file__).parent / "data" / "feature_matrix.parquet"


def build_training_dataset(con: duckdb.DuckDBPyConnection,
                           start_date: str = None,
                           end_date: str = None,
                           batch_size: int = 200,
                           full_rebuild: bool = False) -> pd.DataFrame:
    """Build features for all races in date range.

    Uses saved feature matrix for incremental updates:
    - If feature_matrix.parquet exists and full_rebuild=False, loads it
      and only builds features for new races since last train.
    - Saves the final matrix for next time.
    """
    if not start_date:
        start_date = str(date.today() - timedelta(days=TRAIN_YEARS * 365))
    if not end_date:
        end_date = str(date.today())

    # --- Incremental mode: load saved matrix, add only new races ---
    # Auto-detect if feature columns changed (new features added) -> force rebuild
    _EXPECTED_NEW_COLS = {"sire_win_rate", "has_headgear", "comment_sentiment_avg",
                          "run_style", "intent_score", "rating_peak_diff",
                          "course_draw_bias", "class_drop",
                          "sect_fsp_avg3", "sect_coverage",
                          "form_x_distance", "form_x_going",
                          "course_x_distance", "class_x_form",
                          "finish_strength", "sect_gain_x_form",
                          "early_pos_x_distance",
                          "relative_speed_z", "relative_speed_x_form",
                          "relative_speed_x_distance",
                          "trainer_14day_runs", "trainer_14day_win_rate",
                          "trainer_30day_roi", "lead_probability",
                          "finish_speed_ratio"}
    if not full_rebuild and FEATURE_MATRIX_PATH.exists():
        saved_df = pd.read_parquet(FEATURE_MATRIX_PATH)
        missing_cols = _EXPECTED_NEW_COLS - set(saved_df.columns)
        if missing_cols:
            logger.info(f"  Feature columns changed ({len(missing_cols)} new cols detected) â€” forcing FULL REBUILD")
            full_rebuild = True

    # Detect races with new sectional data — refresh only those, not full rebuild
    _sect_refresh_race_ids: set = set()
    if not full_rebuild and FEATURE_MATRIX_PATH.exists():
        try:
            import datetime as _dt
            matrix_mtime = FEATURE_MATRIX_PATH.stat().st_mtime
            row = con.execute("SELECT MAX(scraped_at) FROM sectionals").fetchone()
            last_sect_ts = row[0] if row and row[0] else None
            if last_sect_ts is not None:
                if hasattr(last_sect_ts, "timestamp"):
                    sect_mtime = last_sect_ts.timestamp()
                else:
                    sect_mtime = _dt.datetime.fromisoformat(str(last_sect_ts)).timestamp()
                if sect_mtime > matrix_mtime:
                    matrix_dt = _dt.datetime.fromtimestamp(matrix_mtime)
                    sect_rows = con.execute(
                        "SELECT DISTINCT race_id FROM sectionals WHERE scraped_at > ?",
                        [matrix_dt]
                    ).fetchall()
                    _sect_refresh_race_ids = {r[0] for r in sect_rows}
                    logger.info(
                        f"  Sectionals updated: {len(_sect_refresh_race_ids)} races need refresh (incremental)"
                    )
        except Exception as _e:
            logger.debug(f"  Sectionals freshness check skipped: {_e}")

    if not full_rebuild and FEATURE_MATRIX_PATH.exists():
        logger.info("  Loading saved feature matrix for incremental update...")
        saved_df = pd.read_parquet(FEATURE_MATRIX_PATH)
        saved_race_ids = set(saved_df["race_id"].unique())
        logger.info(f"  Saved matrix: {len(saved_df):,} runners, {len(saved_race_ids):,} races")

        if _sect_refresh_race_ids:
            stale_ids = _sect_refresh_race_ids & saved_race_ids
            if stale_ids:
                saved_df = saved_df[~saved_df["race_id"].isin(stale_ids)]
                saved_race_ids -= stale_ids
                logger.info(f"  Dropped {len(stale_ids)} stale races for sectional refresh")

        all_race_ids = [r[0] for r in con.execute("""
            SELECT ra.race_id
            FROM races ra
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE m.meeting_date >= ? AND m.meeting_date <= ?
              AND ra.num_runners >= 2
            ORDER BY m.meeting_date
        """, [start_date, end_date]).fetchall()]

        new_race_ids = [rid for rid in all_race_ids if rid not in saved_race_ids]

        if not new_race_ids:
            logger.info("  No new races â€” using saved matrix as-is")
            return saved_df

        logger.info(f"  Building features for {len(new_race_ids)} NEW races...")
        new_dfs = []
        for i in range(0, len(new_race_ids), batch_size):
            batch = new_race_ids[i:i + batch_size]
            batch_df = build_features_for_races(con, batch)
            if not batch_df.empty:
                new_dfs.append(batch_df)
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(new_race_ids):
                logger.info(f"  Progress: {min(i + batch_size, len(new_race_ids))}/{len(new_race_ids)} new races")

        if new_dfs:
            new_df = pd.concat(new_dfs, ignore_index=True)
            result = pd.concat([saved_df, new_df], ignore_index=True)
            logger.info(f"  Added {len(new_df):,} new runners")
        else:
            result = saved_df

        # Drop races older than training window
        old_race_ids = set(all_race_ids)
        result = result[result["race_id"].isin(old_race_ids)]

        before_dedup = len(result)
        result = result.drop_duplicates(subset=["race_id", "horse_name"], keep="last")
        if len(result) < before_dedup:
            logger.info(f"  Deduped: {before_dedup:,} -> {len(result):,} (removed {before_dedup - len(result):,})")

        result.to_parquet(FEATURE_MATRIX_PATH, index=False)
        logger.info(f"Training dataset: {len(result):,} runners, {len(result.columns)} columns (saved)")
        return result

    # --- Full rebuild mode (parallel) ---
    race_ids = [r[0] for r in con.execute("""
        SELECT ra.race_id
        FROM races ra
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE m.meeting_date >= ? AND m.meeting_date <= ?
          AND ra.num_runners >= 2
        ORDER BY m.meeting_date
    """, [start_date, end_date]).fetchall()]

    cache_dir = Path(__file__).parent / "data" / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building features for {len(race_ids)} races ({start_date} to {end_date})")

    existing = sorted(cache_dir.glob("batch_*.parquet"))
    covered_race_ids = set()
    if existing:
        for p in existing:
            try:
                covered_race_ids.update(pd.read_parquet(p, columns=["race_id"])["race_id"].unique())
            except Exception:
                pass
        logger.info(f"  Resumed from cache: {len(existing)} chunks, {len(covered_race_ids)} races already built")

    pending_race_ids = [rid for rid in race_ids if rid not in covered_race_ids]
    pending_batches = [
        pending_race_ids[i:i + batch_size]
        for i in range(0, len(pending_race_ids), batch_size)
    ]

    if not pending_batches:
        logger.info("  All races already cached")
    else:
        logger.info(f"  Sequential build: {len(pending_batches)} batches")

        completed = 0
        chunk_dfs = []

        for batch_ids in pending_batches:
            try:
                batch_df = build_features_for_races(con, batch_ids)
                if batch_df is not None and not batch_df.empty:
                    chunk_dfs.append(batch_df)
            except Exception as e:
                logger.warning(f"  Batch failed: {e}")
            completed += 1
            if completed % 5 == 0 or completed == len(pending_batches):
                done_races = len(covered_race_ids) + completed * batch_size
                logger.info(f"  Progress: {min(done_races, len(race_ids))}/{len(race_ids)} races")
                if chunk_dfs:
                    chunk_path = cache_dir / f"batch_{len(existing):04d}.parquet"
                    pd.concat(chunk_dfs, ignore_index=True).to_parquet(chunk_path, index=False)
                    existing.append(chunk_path)
                    chunk_dfs.clear()
                    logger.info(f"  Checkpoint saved: {chunk_path.name}")

        if chunk_dfs:
            chunk_path = cache_dir / f"batch_{len(existing):04d}.parquet"
            pd.concat(chunk_dfs, ignore_index=True).to_parquet(chunk_path, index=False)
            existing.append(chunk_path)
            logger.info(f"  Final checkpoint saved: {chunk_path.name}")

    actual_files = sorted(cache_dir.glob("batch_*.parquet"))
    if not actual_files:
        logger.warning("No features built -- empty dataset")
        return pd.DataFrame()

    logger.info(f"  Loading {len(actual_files)} checkpoint files...")
    result = pd.concat(
        [pd.read_parquet(p) for p in actual_files],
        ignore_index=True,
    )

    before_dedup = len(result)
    result = result.drop_duplicates(subset=["race_id", "horse_name"], keep="last")
    if len(result) < before_dedup:
        logger.info(f"  Deduped: {before_dedup:,} -> {len(result):,} (removed {before_dedup - len(result):,})")

    result.to_parquet(FEATURE_MATRIX_PATH, index=False)
    logger.info(f"Training dataset: {len(result):,} runners, {len(result.columns)} columns (saved)")

    for f in cache_dir.glob("batch_*.parquet"):
        f.unlink()

    return result


# ===================================================================
# Feature column lists (for model training)
# ===================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature columns suitable for ML training."""
    exclude = set(ID_COLS) | LEAKAGE_COLS | {TARGET_COL, "sp_decimal", "betfair_sp", "last_race_class"}
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ("float64", "float32", "int64", "int32", "Float64", "Int64")
    ]
