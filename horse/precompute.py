"""
Pre-compute predictions locally and save to JSON cache.
Render serves these cached results instead of running ML models.

Usage:
    python -m horse.precompute              # today
    python -m horse.precompute 2026-03-11   # specific date
"""

import json
import logging
import sys
from datetime import date, datetime
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
            probs = calibrator.predict(probs.reshape(-1, 1))
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
                   ra.surface, ra.region_code
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
    features_df["_win_prob"] = win_probs
    features_df["_place_prob"] = place_probs
    features_df["_ew_score"] = 0.3 * win_probs + 0.7 * place_probs

    from .api.thirteen_d import compute_thirteen_d_scores

    cache = {"date": target_date, "generated_at": datetime.now().isoformat(), "races": {}, "best_bets": {}}

    for race_id in race_ids:
        meta = race_meta.get(race_id, {})
        race_features = features_df[features_df["race_id"] == race_id]
        if race_features.empty:
            continue

        dim_scores = compute_thirteen_d_scores(race_features, race_id)

        runners = []
        for _, row in race_features.iterrows():
            name = row.get("horse_name", "")
            jt = jt_map.get((name, race_id), (None, None))
            dims = dim_scores.get(name, {})

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

    candidates = []
    for race_id, group in features_df.groupby("race_id"):
        meta = race_meta.get(race_id, {})
        nr = meta.get("num_runners", len(group))
        if nr and nr < 4:
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

    return out_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    precompute_for_date(target)
