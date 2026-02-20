"""
Speed figure computation for UK/IRE Horse Racing.

Computes normalized speed_figure = (standard_time / horse_time) * 100
where standard_time is the median of top-3 finishers grouped by
(course, distance_bucket, surface, going_group).

Hierarchical fallback when sample size < 20 or variance too high:
  1. course + distance_bucket + surface + going_group  (>=20, CV<8%)
  2. course + distance_bucket + surface                (>=20, CV<8%)
  3. distance_bucket + surface                         (>=20, CV<8%)
  4. NaN (no standard available)

Leakage protection: standard times only use races BEFORE the target date.
"""

import json
import logging
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import duckdb

from .config import GOING_ENCODE
from .db import get_connection

logger = logging.getLogger(__name__)

MIN_BUCKET_SIZE = 20
MAX_CV = 0.08  # 8% coefficient of variation threshold


def parse_time_str(time_str: str) -> Optional[float]:
    """Parse time like '4:7.00' or '1:38.09' into seconds."""
    if not time_str or not isinstance(time_str, str):
        return None
    time_str = time_str.strip()
    m = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', time_str)
    if m:
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        return minutes * 60.0 + seconds
    m2 = re.match(r'^(\d+(?:\.\d+)?)$', time_str)
    if m2:
        return float(m2.group(1))
    return None


def going_group(going_str: Optional[str]) -> str:
    """Map going description to one of: firm, good, soft, heavy."""
    if not going_str:
        return "good"
    g = going_str.strip().lower()
    score = GOING_ENCODE.get(g)
    if score is None:
        for key, val in GOING_ENCODE.items():
            if key in g:
                score = val
                break
    if score is None:
        score = 3.0
    if score <= 1.5:
        return "firm"
    elif score <= 3.5:
        return "good"
    elif score <= 5.0:
        return "soft"
    else:
        return "heavy"


def distance_bucket(dist_f: float) -> Optional[float]:
    """Round distance to nearest 0.5 furlongs for sample stability."""
    if dist_f is None or np.isnan(dist_f) or dist_f <= 0:
        return None
    return round(dist_f * 2) / 2


def normalize_course(course_str: Optional[str]) -> str:
    """Strip country suffixes and normalize course name."""
    if not course_str:
        return ""
    c = re.sub(r'\s*\((?:IRE|GB|FR|USA|GER|AUS|HK|UAE|JPN)\)\s*$', '', course_str, flags=re.IGNORECASE)
    return c.strip().lower()


def normalize_surface(surface_str: Optional[str]) -> str:
    """Normalize surface to 'turf' or 'aw'."""
    if not surface_str:
        return "turf"
    s = surface_str.strip().lower()
    if s in ("aw", "all weather", "all-weather", "dirt", "polytrack", "tapeta", "fibresand"):
        return "aw"
    return "turf"


def backfill_finish_times(con: duckdb.DuckDBPyConnection) -> int:
    """Extract 'time' from results.raw_json and write to results.finish_time.
    Returns count of rows updated."""
    todo = con.execute("""
        SELECT result_id, raw_json
        FROM results
        WHERE raw_json IS NOT NULL
          AND (finish_time IS NULL OR finish_time = 0)
    """).fetchall()

    updated = 0
    batch = []
    for result_id, raw in todo:
        try:
            d = json.loads(raw)
            t = d.get("time")
            secs = parse_time_str(t) if t else None
            if secs and secs > 0:
                batch.append((secs, result_id))
        except (json.JSONDecodeError, TypeError):
            continue

        if len(batch) >= 5000:
            con.executemany(
                "UPDATE results SET finish_time = ? WHERE result_id = ?",
                batch,
            )
            updated += len(batch)
            batch = []

    if batch:
        con.executemany(
            "UPDATE results SET finish_time = ? WHERE result_id = ?",
            batch,
        )
        updated += len(batch)

    logger.info(f"Backfilled finish_time for {updated:,} results")
    return updated


def _build_times_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all results with valid finish times, enriched with race metadata."""
    df = con.execute("""
        SELECT
            r.result_id, r.race_id, r.finish_time, r.position,
            r.horse_name, r.api_horse_id,
            ra.distance_furlongs, ra.going_description, ra.surface,
            m.course, m.meeting_date
        FROM results r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE r.finish_time IS NOT NULL AND r.finish_time > 0
          AND r.position IS NOT NULL AND r.position > 0
          AND ra.distance_furlongs IS NOT NULL AND ra.distance_furlongs > 0
    """).df()

    if df.empty:
        return df

    df["course_norm"] = df["course"].apply(normalize_course)
    df["surface_norm"] = df["surface"].apply(normalize_surface)
    df["going_grp"] = df["going_description"].apply(going_group)
    df["dist_bucket"] = df["distance_furlongs"].apply(distance_bucket)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df


def _stable_median(times: pd.Series) -> Tuple[Optional[float], bool]:
    """Compute median and check stability (CV < threshold).
    Returns (median, is_stable)."""
    if len(times) < MIN_BUCKET_SIZE:
        return None, False
    med = times.median()
    if med <= 0:
        return None, False
    std = times.std()
    cv = std / med
    if cv > MAX_CV:
        return None, False
    return med, True


def compute_standard_time(
    times_df: pd.DataFrame,
    course: str,
    dist_bucket_val: float,
    surface: str,
    going_grp: str,
    before_date: pd.Timestamp,
) -> Optional[float]:
    """Compute standard time with hierarchical fallback + variance guard.

    Priority:
      1. course + dist_bucket + surface + going_group
      2. course + dist_bucket + surface
      3. dist_bucket + surface
    """
    historical = times_df[times_df["meeting_date"] < before_date]
    if historical.empty:
        return None

    top3 = historical[historical["position"] <= 3]
    if top3.empty:
        return None

    # Level 1: course + dist_bucket + surface + going_group
    mask = (
        (top3["course_norm"] == course) &
        (top3["dist_bucket"] == dist_bucket_val) &
        (top3["surface_norm"] == surface) &
        (top3["going_grp"] == going_grp)
    )
    med, ok = _stable_median(top3.loc[mask, "finish_time"])
    if ok:
        return med

    # Level 2: course + dist_bucket + surface (drop going)
    mask = (
        (top3["course_norm"] == course) &
        (top3["dist_bucket"] == dist_bucket_val) &
        (top3["surface_norm"] == surface)
    )
    med, ok = _stable_median(top3.loc[mask, "finish_time"])
    if ok:
        return med

    # Level 3: dist_bucket + surface (drop course)
    mask = (
        (top3["dist_bucket"] == dist_bucket_val) &
        (top3["surface_norm"] == surface)
    )
    med, ok = _stable_median(top3.loc[mask, "finish_time"])
    if ok:
        return med

    return None


def compute_all_speed_figures(con: duckdb.DuckDBPyConnection) -> int:
    """Compute speed_figure for all results that have finish_time.
    Also writes speed_figure back to horse_form rows where matched.
    Returns count of results with speed figures."""
    logger.info("Loading times data for speed figure computation...")
    times_df = _build_times_df(con)
    if times_df.empty:
        logger.warning("No valid finish times found")
        return 0

    logger.info(f"Computing speed figures for {len(times_df):,} results...")

    # Pre-compute standard times by unique (race_id, date) to avoid
    # recomputing for every runner in the same race
    race_meta = times_df.drop_duplicates("race_id")[
        ["race_id", "course_norm", "dist_bucket", "surface_norm",
         "going_grp", "meeting_date"]
    ].set_index("race_id")

    std_cache = {}
    computed = 0
    updates_results = []
    updates_form = []

    for race_id, group in times_df.groupby("race_id"):
        meta = race_meta.loc[race_id]
        cache_key = (meta["course_norm"], meta["dist_bucket"],
                     meta["surface_norm"], meta["going_grp"],
                     meta["meeting_date"])

        if cache_key not in std_cache:
            std_cache[cache_key] = compute_standard_time(
                times_df,
                meta["course_norm"],
                meta["dist_bucket"],
                meta["surface_norm"],
                meta["going_grp"],
                meta["meeting_date"],
            )

        std_time = std_cache[cache_key]
        if std_time is None or std_time <= 0:
            continue

        for _, row in group.iterrows():
            horse_time = row["finish_time"]
            if horse_time <= 0:
                continue
            sf = (std_time / horse_time) * 100.0
            sf = round(sf, 2)
            updates_results.append((sf, row["result_id"]))
            if row["api_horse_id"] and pd.notna(row["meeting_date"]):
                updates_form.append((
                    sf, row["api_horse_id"],
                    row["meeting_date"].strftime("%Y-%m-%d"),
                    meta["course_norm"],
                ))
            computed += 1

        if len(updates_results) >= 10000:
            _flush_speed_updates(con, updates_results, updates_form)
            updates_results = []
            updates_form = []
            logger.info(f"  Speed figures: {computed:,} computed so far")

    if updates_results:
        _flush_speed_updates(con, updates_results, updates_form)

    logger.info(
        f"Speed figures complete: {computed:,} computed, "
        f"{len(std_cache)} unique standard-time buckets"
    )
    return computed


def _flush_speed_updates(con, updates_results, updates_form):
    """Batch write speed figures to results and horse_form tables."""
    if updates_results:
        con.executemany(
            "UPDATE results SET speed_figure = ? WHERE result_id = ?",
            updates_results,
        )
    if updates_form:
        for sf, api_id, race_date, course in updates_form:
            try:
                con.execute("""
                    UPDATE horse_form SET speed_figure = ?
                    WHERE api_horse_id = ?
                      AND race_date = ?
                      AND LOWER(REPLACE(REPLACE(course, ' (IRE)', ''), ' (GB)', '')) = ?
                """, [sf, api_id, race_date, course])
            except Exception:
                pass
