"""
Pre-compute predictions locally and save to JSON cache.
Render serves these cached results instead of running ML models.

Usage:
    python -m horse.precompute              # today + tomorrow
    python -m horse.precompute 2026-03-11   # specific date
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DB_PATH
from .features import build_features_for_races
from .models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "predictions_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NR_JOCKEY_PATTERNS = {"NON-RUNNER", "NON RUNNER", "NR", "N/R", "NONRUNNER"}


def _is_non_runner(jockey_name: str) -> bool:
    if not jockey_name:
        return False
    return jockey_name.strip().upper() in NR_JOCKEY_PATTERNS or "NON-RUNNER" in jockey_name.upper()


def _is_undeclared_entry(jockey_name: str) -> bool:
    """Horse is just an entry, not a confirmed declaration (no jockey assigned)."""
    if not jockey_name or not jockey_name.strip():
        return True
    upper = jockey_name.strip().upper()
    if upper in ("TBC", "TBA", "TO BE CONFIRMED", "TO BE ANNOUNCED", "RESERVE", "—", "-"):
        return True
    return False


def _num_paying_places(num_runners: int, is_handicap: bool = False) -> int:
    if num_runners <= 4:
        return 1
    if num_runners <= 7:
        return 2
    if is_handicap and num_runners >= 16:
        return 4
    return 3


def _normalize_probs(probs: np.ndarray, target_sum: float) -> np.ndarray:
    """Normalize probabilities to sum to target_sum, preserving relative ordering."""
    s = probs.sum()
    if s <= 0:
        return np.full_like(probs, target_sum / len(probs))
    return probs * (target_sum / s)


def _predict_stacked(model_data, X, feature_names):
    import xgboost as xgb

    lgb_model = model_data["lgb_model"]
    xgb_model = model_data["xgb_model"]
    catboost_model = model_data.get("catboost_model")
    rf_model = model_data.get("rf_model")
    ensemble_weights = model_data.get("ensemble_weights", {"lgb": 0.30, "xgb": 0.20})
    calibrator = model_data.get("calibrator")

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_names))

    base_preds = {"lgb": lgb_probs, "xgb": xgb_probs}

    if catboost_model is not None:
        try:
            base_preds["catboost"] = catboost_model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if rf_model is not None:
        try:
            base_preds["rf"] = rf_model.predict_proba(X.fillna(0))[:, 1]
        except Exception:
            pass

    probs = np.zeros(len(X))
    total_w = 0.0
    for name, preds in base_preds.items():
        w = ensemble_weights.get(name, 0)
        if w > 0:
            probs += w * preds
            total_w += w
    if total_w > 0:
        probs /= total_w

    if calibrator is not None:
        try:
            raw_probs = probs.copy()
            if hasattr(calibrator, "predict_proba"):
                cal_probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            else:
                cal_probs = calibrator.predict(probs.reshape(-1, 1))

            # If calibration destroys differentiation, blend with raw to preserve ordering
            raw_unique = len(set(np.round(raw_probs, 6)))
            cal_unique = len(set(np.round(cal_probs, 6)))
            if cal_unique < max(3, raw_unique // 2) and len(probs) > 3:
                probs = 0.5 * cal_probs + 0.5 * raw_probs
            else:
                probs = cal_probs
        except Exception:
            pass

    return probs


def precompute_for_date(target_date: str):
    import duckdb
    from .api.thirteen_d import compute_thirteen_d_scores

    logger.info(f"Pre-computing predictions for {target_date}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        race_rows = con.execute("""
            SELECT ra.race_id, m.course, ra.race_name,
                   COALESCE(ra.race_time, strftime(ra.off_dt, '%H:%M')) as race_time,
                   ra.num_runners, ra.api_race_id,
                   m.meeting_date, ra.race_number, ra.distance_furlongs,
                   ra.race_type, ra.race_class, ra.going_description,
                   ra.surface, ra.region_code, ra.handicap_flag
            FROM races ra
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE m.meeting_date = ?
            ORDER BY ra.off_dt
        """, [target_date]).fetchall()

        if not race_rows:
            logger.warning(f"No races found for {target_date}")
            return

        logger.info(f"Found {len(race_rows)} races")
        race_ids = [r[0] for r in race_rows]

        race_meta = {}
        for r in race_rows:
            race_meta[r[0]] = {
                "race_id": r[0], "course": r[1], "race_name": r[2],
                "race_time": r[3], "num_runners": r[4], "api_race_id": r[5],
                "meeting_date": str(r[6]), "race_number": r[7],
                "distance_furlongs": float(r[8]) if r[8] else None,
                "race_type": r[9], "race_class": r[10], "going": r[11],
                "surface": r[12], "region_code": r[13],
                "handicap_flag": bool(r[14]) if r[14] is not None else False,
            }

        logger.info("Building features...")
        features_df = build_features_for_races(con, race_ids)

        jt_rows = con.execute("""
            SELECT horse_name, jockey_name, trainer_name, race_id
            FROM results WHERE race_id IN ({})
        """.format(",".join(str(r) for r in race_ids))).fetchall()
        jt_map = {(r[0], r[3]): (r[1], r[2]) for r in jt_rows}

        runner_api_rows = con.execute("""
            SELECT race_id, horse_name, api_horse_id
            FROM results WHERE race_id IN ({}) AND api_horse_id IS NOT NULL
        """.format(",".join(str(r) for r in race_ids))).fetchall()
        runner_api_map = {}
        for r in runner_api_rows:
            if r[0] not in runner_api_map:
                runner_api_map[r[0]] = {}
            runner_api_map[r[0]][r[1]] = r[2]
    finally:
        con.close()

    if features_df.empty:
        logger.warning("No features built")
        return

    logger.info("Loading models...")
    win_model = load_model("win")
    place_model = load_model("place")
    feature_names = win_model["feature_names"]

    X = features_df.reindex(columns=feature_names).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    logger.info("Running predictions...")
    win_probs = _predict_stacked(win_model, X, feature_names)
    place_probs = _predict_stacked(place_model, X, feature_names)

    features_df = features_df.copy()
    features_df["_win_prob_raw"] = win_probs
    features_df["_place_prob_raw"] = place_probs

    # --- Per-race: filter NRs, normalize, quality-check ---
    nr_total = 0
    quality_warnings = []

    for race_id in race_ids:
        mask = features_df["race_id"] == race_id
        idx = features_df.index[mask]
        if len(idx) == 0:
            continue

        meta = race_meta.get(race_id, {})
        is_hcap = meta.get("handicap_flag", False)

        # Filter only explicit non-runners (jockey field says NON-RUNNER)
        exclude_mask = []
        declared_count = 0
        for i in idx:
            name = features_df.at[i, "horse_name"]
            jt = jt_map.get((name, race_id), (None, None))
            jockey = jt[0] or ""
            is_nr = _is_non_runner(jockey)
            exclude_mask.append(is_nr)
            if jockey.strip() and not is_nr:
                declared_count += 1
        exclude_mask = np.array(exclude_mask)

        nr_count = exclude_mask.sum()
        nr_total += nr_count

        active_idx = idx[~exclude_mask]

        # Flag pre-declaration races (entries without jockeys)
        if declared_count == 0:
            meta["_pre_declaration"] = True
            logger.info(
                f"  PRE-DECLARATION: {meta.get('race_time','')} {meta.get('course','')}: "
                f"{len(active_idx)} entries (jockeys not yet declared)"
            )

        if len(active_idx) < 2:
            meta["_entries_only"] = True
            features_df.loc[idx, "_win_prob_raw"] = 0.0
            features_df.loc[idx, "_place_prob_raw"] = 0.0
            features_df.loc[idx, "_win_prob"] = 0.0
            features_df.loc[idx, "_place_prob"] = 0.0
            features_df.loc[idx, "_ew_score"] = 0.0
            logger.warning(
                f"  SKIPPING {meta.get('race_time','')} {meta.get('course','')}: "
                f"only {len(active_idx)} active runner(s)"
            )
            continue

        # Mark excluded runners with zero probability
        if nr_count > 0:
            features_df.loc[idx[exclude_mask], "_win_prob_raw"] = 0.0
            features_df.loc[idx[exclude_mask], "_place_prob_raw"] = 0.0

        # Normalize win probs to sum to 1.0 across active runners
        raw_win = features_df.loc[active_idx, "_win_prob_raw"].values.copy()
        norm_win = _normalize_probs(raw_win, 1.0)

        MAX_WIN_PROB = 0.70
        if norm_win.max() > MAX_WIN_PROB and len(active_idx) > 2:
            excess = norm_win - MAX_WIN_PROB
            excess[excess < 0] = 0
            norm_win = norm_win - excess
            remainder = excess.sum()
            below_cap = norm_win < MAX_WIN_PROB
            if below_cap.sum() > 0:
                norm_win[below_cap] += remainder / below_cap.sum()
            norm_win = _normalize_probs(norm_win, 1.0)

        features_df.loc[active_idx, "_win_prob"] = norm_win

        # Normalize place probs to sum to num_paying_places
        n_active = len(active_idx)
        n_places = _num_paying_places(n_active, is_hcap)
        raw_place = features_df.loc[active_idx, "_place_prob_raw"].values.copy()
        norm_place = _normalize_probs(raw_place, float(n_places))
        norm_place = np.clip(norm_place, 0.0, 0.85)

        # Enforce place_prob >= win_prob (a winner always places)
        norm_place = np.maximum(norm_place, norm_win)

        features_df.loc[active_idx, "_place_prob"] = norm_place

        # Excluded runners get 0
        if nr_count > 0:
            features_df.loc[idx[exclude_mask], "_win_prob"] = 0.0
            features_df.loc[idx[exclude_mask], "_place_prob"] = 0.0

        # E/W score from normalized probs
        features_df.loc[active_idx, "_ew_score"] = (
            0.3 * features_df.loc[active_idx, "_win_prob"].values +
            0.7 * features_df.loc[active_idx, "_place_prob"].values
        )
        if nr_count > 0:
            features_df.loc[idx[exclude_mask], "_ew_score"] = 0.0

        # Quality check
        unique_win = len(set(np.round(norm_win, 4)))
        if unique_win < 3 and n_active > 3:
            quality_warnings.append(
                f"  LOW CONFIDENCE: {meta.get('race_time','')} {meta.get('course','')} "
                f"({n_active} runners, only {unique_win} distinct probs)"
            )

        # Update meta with actual active runner count
        meta["num_runners_active"] = int(n_active)
        meta["num_non_runners"] = int(nr_count)

    if nr_total > 0:
        logger.info(f"Filtered {nr_total} non-runners/undeclared entries across all races")

    from .api.thirteen_d import compute_thirteen_d_scores, compute_drivers

    # Collect race IDs that were skipped (entries only, no declarations)
    skipped_race_ids = [
        rid for rid in race_ids
        if race_meta.get(rid, {}).get("_entries_only")
    ]

    cache = {"date": target_date, "generated_at": datetime.now().isoformat(), "races": {}, "best_bets": {}, "skipped_race_ids": skipped_race_ids}

    for race_id in race_ids:
        meta = race_meta.get(race_id, {})

        # Skip races with no declarations
        if meta.get("_entries_only"):
            continue

        race_features = features_df[features_df["race_id"] == race_id]
        if race_features.empty:
            continue

        # Only include active runners (not NRs)
        active_features = race_features[race_features["_win_prob"] > 0]
        nr_features = race_features[race_features["_win_prob"] <= 0]

        dim_scores = compute_thirteen_d_scores(active_features, race_id) if not active_features.empty else {}

        runners = []
        for _, row in active_features.iterrows():
            name = row.get("horse_name", "")
            jt = jt_map.get((name, race_id), (None, None))
            dims = dim_scores.get(name, {})

            pos_drivers, neg_drivers = compute_drivers(active_features, name, race_id)

            runner = {
                "horse_name": name,
                "win_prob": round(float(row["_win_prob"]), 4),
                "place_prob": round(float(row["_place_prob"]), 4),
                "ew_score": round(float(row["_ew_score"]), 4),
                "jockey": jt[0],
                "trainer": jt[1],
                "official_rating": int(row["official_rating"]) if pd.notna(row.get("official_rating")) else None,
                "form_string": str(row.get("form_string", "")) if pd.notna(row.get("form_string")) else None,
                "dimensions": dims,
                "positive_drivers": pos_drivers,
                "negative_drivers": neg_drivers,
            }

            for feat in ["form_win_rate", "dist_match_win_rate", "going_match_win_rate",
                         "course_match_win_rate", "form_avg_pos_3", "speed_vs_field_avg"]:
                val = row.get(feat)
                runner[feat] = round(float(val), 4) if pd.notna(val) else None

            runners.append(runner)

        runners.sort(key=lambda r: r["win_prob"], reverse=True)

        cache["races"][str(race_id)] = {
            "meta": meta,
            "runners": runners,
            "runner_api_ids": runner_api_map.get(race_id, {}),
        }

    # --- Best bets (from normalized, NR-filtered data) ---
    active_df = features_df[features_df["_win_prob"] > 0]
    candidates = []
    for race_id, group in active_df.groupby("race_id"):
        meta = race_meta.get(race_id, {})
        n_active = meta.get("num_runners_active", len(group))
        if n_active < 4:
            continue

        sorted_group = group.sort_values("_ew_score", ascending=False)
        top = sorted_group.iloc[0]
        wp = float(top["_win_prob"])
        pp = float(top["_place_prob"])
        ew = float(top["_ew_score"])
        gap = pp - wp
        field_avg_pp = group["_place_prob"].mean()
        edge = pp - field_avg_pp

        jt = jt_map.get((top["horse_name"], race_id), (None, None))

        reason_parts = []
        if pp > 0.45:
            reason_parts.append("strong place chance")
        if wp > 0.2:
            reason_parts.append("genuine win contender")
        if gap > 0.25:
            reason_parts.append(f"{gap*100:.0f}% place safety net")
        if edge > 0.15:
            reason_parts.append("big edge over field")
        if top.get("form_avg_pos_3") and pd.notna(top.get("form_avg_pos_3")) and float(top.get("form_avg_pos_3", 99)) <= 3.0:
            reason_parts.append("consistent recent form")
        if not reason_parts:
            reason_parts.append("best E/W value in race")

        candidates.append({
            "horse_name": str(top["horse_name"]),
            "course": meta.get("course", ""),
            "race_time": meta.get("race_time"),
            "race_name": meta.get("race_name"),
            "race_id": int(race_id),
            "win_prob": round(wp, 4),
            "place_prob": round(pp, 4),
            "pct_gap": round(gap, 4),
            "fair_odds": round(1.0 / max(wp, 0.001), 2),
            "jockey": jt[0],
            "trainer": jt[1],
            "official_rating": int(top["official_rating"]) if pd.notna(top.get("official_rating")) else None,
            "confidence": round(ew, 4),
            "reason": " + ".join(reason_parts),
        })

    candidates.sort(key=lambda c: (c["confidence"], c["pct_gap"]), reverse=True)

    seen = set()
    top_picks = []
    for c in candidates:
        if c["horse_name"] in seen:
            continue
        seen.add(c["horse_name"])
        top_picks.append(c)
        if len(top_picks) == 4:
            break

    cache["best_bets"] = {
        "date": target_date,
        "picks": top_picks,
        "total_races_scanned": len(race_ids),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = CACHE_DIR / f"{target_date}.json"
    out_path.write_text(json.dumps(cache, default=str), encoding="utf-8")
    logger.info(f"Cache saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"  {len(cache['races'])} races, {len(top_picks)} best bets")

    # --- Quality audit ---
    logger.info("QUALITY AUDIT:")
    ok_count = 0
    warn_count = 0
    for rid, race_data in cache["races"].items():
        runners = race_data["runners"]
        if not runners:
            continue
        win_sum = sum(r["win_prob"] for r in runners)
        n_unique = len(set(r["win_prob"] for r in runners))
        m = race_data["meta"]
        status = "OK"
        if abs(win_sum - 1.0) > 0.02:
            status = f"WARN sum={win_sum:.2f}"
            warn_count += 1
        elif n_unique < 3 and len(runners) > 3:
            status = f"WARN only {n_unique} distinct"
            warn_count += 1
        else:
            ok_count += 1
        if status != "OK":
            logger.warning(f"  {m.get('race_time','')} {m.get('course','')}: {len(runners)} runners, win_sum={win_sum:.3f}, {n_unique} unique probs - {status}")

    logger.info(f"  {ok_count} races OK, {warn_count} warnings")
    if quality_warnings:
        for w in quality_warnings:
            logger.warning(w)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        precompute_for_date(sys.argv[1])
    else:
        today = date.today()
        tomorrow = today + timedelta(days=1)
        precompute_for_date(str(today))
        precompute_for_date(str(tomorrow))
