"""
Prediction Reconciliation for Horse Racing.

Reads predictions from predictions.duckdb and actual results from horse.duckdb,
joins them in Python to measure:
  - Top-pick win hit rate
  - Top-3 place hit rate
  - ROI simulation (flat-stake on top pick at SP)
  - Calibration (predicted % vs actual win %)
  - Per-day and per-course breakdown

Run after scraping results:
    python -m horse.reconcile
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .db import get_connection
from .prediction_db import get_pred_connection

logger = logging.getLogger(__name__)


def copy_actual_results() -> int:
    """Copy finished race results from horse.duckdb into predictions.duckdb.
    Only copies results for races that have predictions saved.
    Returns number of new rows inserted."""
    pred_con = get_pred_connection()
    try:
        predicted_race_ids = pred_con.execute(
            "SELECT DISTINCT race_id FROM predictions"
        ).fetchall()
        race_ids = [r[0] for r in predicted_race_ids]
    finally:
        pred_con.close()

    if not race_ids:
        return 0

    main_con = get_connection(read_only=True)
    try:
        placeholders = ",".join(["?"] * len(race_ids))
        actuals = main_con.execute(f"""
            SELECT r.race_id, r.horse_name, r.position, r.sp_decimal,
                   ra.num_runners, m.meeting_date, m.course
            FROM results r
            JOIN races ra ON r.race_id = ra.race_id
            JOIN meetings m ON ra.meeting_id = m.meeting_id
            WHERE r.race_id IN ({placeholders})
              AND r.position IS NOT NULL
              AND r.position > 0
        """, race_ids).fetchall()
    finally:
        main_con.close()

    if not actuals:
        return 0

    pred_con = get_pred_connection()
    inserted = 0
    try:
        for row in actuals:
            try:
                rid = pred_con.execute(
                    "SELECT nextval('seq_recon_id')"
                ).fetchone()[0]
                pred_con.execute("""
                    INSERT INTO reconciled_results
                        (id, race_id, horse_name, actual_position, sp_decimal,
                         num_runners, meeting_date, course, reconciled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                    ON CONFLICT (race_id, horse_name) DO UPDATE SET
                        actual_position = excluded.actual_position,
                        sp_decimal = excluded.sp_decimal,
                        reconciled_at = excluded.reconciled_at
                """, [rid, *row])
                inserted += 1
            except Exception:
                pass
        logger.info(f"Copied {inserted} actual results to predictions.duckdb")
    finally:
        pred_con.close()
    return inserted


def reconcile_predictions(days_back: int = 30) -> dict:
    """Join predictions against actual results for the last N days."""

    copy_actual_results()

    pred_con = get_pred_connection(read_only=True)
    try:
        cutoff = (datetime.now() - pd.Timedelta(days=days_back)).isoformat()
        preds_df = pred_con.execute("""
            SELECT race_id, horse_name, win_prob, place_prob,
                   win_rank, place_rank, model_type, predicted_at
            FROM predictions
            WHERE predicted_at >= ?
        """, [cutoff]).fetchdf()

        recon_df = pred_con.execute("""
            SELECT race_id, horse_name, actual_position, sp_decimal,
                   num_runners, meeting_date, course
            FROM reconciled_results
        """).fetchdf()
    finally:
        pred_con.close()

    if preds_df.empty:
        return {
            "total_races": 0,
            "total_predictions": 0,
            "message": "No predictions saved yet. View races in the UI to generate predictions.",
            "timestamp": datetime.now().isoformat(),
        }

    if recon_df.empty:
        return {
            "total_races": 0,
            "total_predictions": len(preds_df),
            "message": "Predictions saved but no results yet. Scrape results after races finish.",
            "timestamp": datetime.now().isoformat(),
        }

    matched = pd.merge(
        preds_df, recon_df,
        on=["race_id", "horse_name"],
        how="inner",
    )

    if matched.empty:
        return {
            "total_races": 0,
            "total_predictions": len(preds_df),
            "message": "Predictions saved but results haven't been matched yet. Run reconciliation after scraping results.",
            "timestamp": datetime.now().isoformat(),
        }

    race_ids = matched["race_id"].unique()
    total_races = len(race_ids)
    total_preds = len(matched)

    # --- Top Pick Win Rate ---
    top_picks = matched[matched["win_rank"] == 1]
    top_pick_wins = int((top_picks["actual_position"] == 1).sum())
    top_pick_places = int((top_picks["actual_position"] <= 3).sum())
    top_pick_total = len(top_picks)
    top_pick_win_rate = top_pick_wins / max(top_pick_total, 1)
    top_pick_place_rate = top_pick_places / max(top_pick_total, 1)

    # --- Top 3 Picks Place Rate ---
    top3_picks = matched[matched["win_rank"] <= 3]
    top3_actual_place = int((top3_picks["actual_position"] <= 3).sum())
    top3_total = len(top3_picks)
    top3_place_rate = top3_actual_place / max(top3_total, 1)

    # --- ROI (flat stake on top pick at SP) ---
    roi_bets = 0
    roi_returns = 0.0
    for _, row in top_picks.iterrows():
        sp = row.get("sp_decimal")
        if sp and sp > 0:
            roi_bets += 1
            if row["actual_position"] == 1:
                roi_returns += float(sp)
    roi_pct = ((roi_returns - roi_bets) / max(roi_bets, 1)) * 100

    # --- Calibration buckets ---
    calibration = []
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    for low, high in bins:
        bucket = matched[(matched["win_prob"] >= low) & (matched["win_prob"] < high)]
        if len(bucket) > 0:
            actual_win_rate = float((bucket["actual_position"] == 1).mean())
            predicted_avg = float(bucket["win_prob"].mean())
            calibration.append({
                "bucket": f"{int(low*100)}-{int(high*100)}%",
                "count": int(len(bucket)),
                "predicted_avg": round(predicted_avg, 4),
                "actual_win_rate": round(actual_win_rate, 4),
                "gap": round(actual_win_rate - predicted_avg, 4),
            })

    # --- Per-day breakdown ---
    daily = []
    if "meeting_date" in matched.columns:
        for dt in sorted(matched["meeting_date"].unique(), reverse=True)[:14]:
            day_data = matched[matched["meeting_date"] == dt]
            day_top = day_data[day_data["win_rank"] == 1]
            day_races = len(day_data["race_id"].unique())
            day_wins = int((day_top["actual_position"] == 1).sum())
            day_places = int((day_top["actual_position"] <= 3).sum())

            day_roi_bets = 0
            day_roi_returns = 0.0
            for _, row in day_top.iterrows():
                sp = row.get("sp_decimal")
                if sp and sp > 0:
                    day_roi_bets += 1
                    if row["actual_position"] == 1:
                        day_roi_returns += float(sp)

            daily.append({
                "date": str(dt),
                "races": day_races,
                "top_pick_wins": day_wins,
                "top_pick_places": day_places,
                "top_pick_total": len(day_top),
                "win_rate": round(day_wins / max(len(day_top), 1), 4),
                "roi_pct": round(((day_roi_returns - day_roi_bets) / max(day_roi_bets, 1)) * 100, 1),
            })

    # --- Top courses ---
    course_stats = []
    if "course" in matched.columns:
        for course in matched["course"].unique():
            c_data = matched[matched["course"] == course]
            c_top = c_data[c_data["win_rank"] == 1]
            if len(c_top) >= 3:
                c_wins = int((c_top["actual_position"] == 1).sum())
                course_stats.append({
                    "course": str(course),
                    "races": len(c_top),
                    "wins": c_wins,
                    "win_rate": round(c_wins / len(c_top), 4),
                })
        course_stats.sort(key=lambda x: x["win_rate"], reverse=True)

    return {
        "total_races": total_races,
        "total_predictions": total_preds,
        "days_back": days_back,
        "top_pick": {
            "total": top_pick_total,
            "wins": top_pick_wins,
            "places": top_pick_places,
            "win_rate": round(top_pick_win_rate, 4),
            "place_rate": round(top_pick_place_rate, 4),
        },
        "top_3_picks": {
            "total": top3_total,
            "placed": top3_actual_place,
            "place_rate": round(top3_place_rate, 4),
        },
        "roi": {
            "bets": roi_bets,
            "returns": round(roi_returns, 2),
            "roi_pct": round(roi_pct, 1),
            "profit_loss": round(roi_returns - roi_bets, 2),
        },
        "calibration": calibration,
        "daily": daily,
        "courses": course_stats[:10],
        "timestamp": datetime.now().isoformat(),
    }


def get_prediction_count() -> dict:
    """Quick stats: how many predictions are saved, how many reconciled."""
    pred_con = get_pred_connection(read_only=True)
    try:
        total = pred_con.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        reconciled = pred_con.execute("""
            SELECT COUNT(*)
            FROM predictions p
            JOIN reconciled_results r
              ON p.race_id = r.race_id AND p.horse_name = r.horse_name
        """).fetchone()[0]

        unresolved = total - reconciled
        races_predicted = pred_con.execute(
            "SELECT COUNT(DISTINCT race_id) FROM predictions"
        ).fetchone()[0]

        return {
            "total_predictions": total,
            "reconciled": reconciled,
            "unresolved": unresolved,
            "races_predicted": races_predicted,
        }
    finally:
        pred_con.close()


def print_report(days_back: int = 30):
    """Print a human-readable performance report."""
    summary = reconcile_predictions(days_back)

    print("\n" + "=" * 60)
    print("  HORSE RACING 13D -- PREDICTION PERFORMANCE")
    print("=" * 60)

    if summary["total_races"] == 0:
        print(f"\n  {summary.get('message', 'No data yet.')}")
        print("=" * 60 + "\n")
        return summary

    tp = summary["top_pick"]
    t3 = summary["top_3_picks"]
    roi = summary["roi"]

    print(f"\n  Period: last {days_back} days")
    print(f"  Races predicted: {summary['total_races']}")
    print(f"  Total predictions: {summary['total_predictions']}")

    print(f"\n  TOP PICK (rank #1):")
    print(f"    Win rate:   {tp['wins']}/{tp['total']} = {tp['win_rate']*100:.1f}%")
    print(f"    Place rate: {tp['places']}/{tp['total']} = {tp['place_rate']*100:.1f}%")

    print(f"\n  TOP 3 PICKS:")
    print(f"    Placed (top 3): {t3['placed']}/{t3['total']} = {t3['place_rate']*100:.1f}%")

    print(f"\n  ROI (flat stake on top pick at SP):")
    print(f"    Bets: {roi['bets']}  |  Returns: {roi['returns']:.2f}")
    print(f"    P/L: {roi['profit_loss']:+.2f}  |  ROI: {roi['roi_pct']:+.1f}%")

    if summary["calibration"]:
        print(f"\n  CALIBRATION:")
        for c in summary["calibration"]:
            gap_str = f"{c['gap']:+.1%}"
            print(f"    {c['bucket']:>8s}: predicted {c['predicted_avg']:.1%} "
                  f"vs actual {c['actual_win_rate']:.1%} ({gap_str}) [{c['count']} runners]")

    if summary["daily"]:
        print(f"\n  DAILY BREAKDOWN:")
        for d in summary["daily"][:7]:
            print(f"    {d['date']}: {d['top_pick_wins']}/{d['top_pick_total']} wins "
                  f"({d['win_rate']*100:.0f}%) | ROI {d['roi_pct']:+.1f}%")

    if summary["courses"]:
        print(f"\n  TOP COURSES:")
        for c in summary["courses"][:5]:
            print(f"    {c['course']}: {c['wins']}/{c['races']} = {c['win_rate']*100:.0f}%")

    print("\n" + "=" * 60 + "\n")
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print_report(days)
