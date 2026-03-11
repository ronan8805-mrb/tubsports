"""
RETRAIN HORSE MODEL - Full feature pipeline + LightGBM/XGBoost ensemble.

Run:  python -m horse.retrain
      python -m horse.retrain --full-rebuild
      python -m horse.retrain --full-rebuild --precompute-predictions
      python -m horse.retrain --full-rebuild --train-models CatBoost,LightGBM,XGBoost --target WIN,PLACE
Or:   RETRAIN_HORSE.bat
"""

import sys
import os
import argparse
import logging
import json
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("horse_retrain.log", mode="w"),
    ],
)
logger = logging.getLogger("horse.retrain")

ALL_MODELS = {"CatBoost", "LightGBM", "XGBoost", "RandomForest"}
ALL_TARGETS = {"WIN", "PLACE"}


def _parse_args():
    p = argparse.ArgumentParser(description="Retrain horse racing ML models")
    p.add_argument("--full-rebuild", action="store_true",
                   help="Force full feature rebuild instead of incremental")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature names to verify/log after training")
    p.add_argument("--train-models", type=str, default="CatBoost,LightGBM,XGBoost,RandomForest",
                   help="Comma-separated models to train (CatBoost,LightGBM,XGBoost,RandomForest)")
    p.add_argument("--target", type=str, default="WIN,PLACE",
                   help="Comma-separated targets: WIN, PLACE, or both")
    p.add_argument("--validate", type=str, default="full",
                   choices=["full", "calibration_set_only"],
                   help="Validation strategy: full or calibration_set_only")
    p.add_argument("--historical-only", action="store_true",
                   help="Skip Phase 0 live scraping/weather/reconciliation")
    p.add_argument("--log-feature-importances", action="store_true", default=True,
                   help="Log top features after training (default: True)")
    p.add_argument("--save-models", action="store_true", default=True,
                   help="Save trained models to disk (default: True)")
    p.add_argument("--precompute-predictions", action="store_true",
                   help="Run precompute for today+tomorrow after training")
    p.add_argument("--tune", action="store_true",
                   help="Run Optuna hyperparameter tuning")
    return p.parse_args()


def main():
    args = _parse_args()

    requested_models = {m.strip() for m in args.train_models.split(",")} & ALL_MODELS
    requested_targets = {t.strip().upper() for t in args.target.split(",")} & ALL_TARGETS
    tracked_extra = [f.strip() for f in args.features.split(",")] if args.features else []

    from horse.db import get_connection, get_db_stats
    from horse.features import build_training_dataset, get_feature_columns
    from horse.models import (
        train_model, save_model, audit_leakage, time_based_cv,
        purged_walk_forward_cv, run_optuna_tuning,
        load_hard_examples, MODEL_DIR,
    )

    logger.info("=" * 60)
    logger.info("  HORSE RACING AI - RETRAIN")
    logger.info("  LightGBM (CPU) + XGBoost (GPU/CUDA)")
    logger.info(f"  Models: {', '.join(sorted(requested_models))}")
    logger.info(f"  Targets: {', '.join(sorted(requested_targets))}")
    logger.info(f"  Validate: {args.validate}")
    logger.info(f"  Historical-only: {args.historical_only}")
    logger.info("=" * 60)

    # ── PHASE 0: Pre-train data refresh ──────────────────────────
    if not args.historical_only:
        logger.info("PHASE 0a: Scraping recent results...")
        try:
            from horse.scrapers.racing_api import fetch_recent_results
            res = fetch_recent_results(days_back=3)
            logger.info(f"  Results ingested: {res['races']} races, {res['runners']} runners")
        except Exception as e:
            logger.warning(f"  Results scrape skipped: {e}")

        logger.info("PHASE 0b: Backfilling weather...")
        try:
            from horse.scrapers.weather import backfill_weather_for_meetings
            wcount = backfill_weather_for_meetings()
            logger.info(f"  Weather records filled: {wcount}")
        except Exception as e:
            logger.warning(f"  Weather backfill skipped: {e}")

        logger.info("PHASE 0c: Reconciling predictions vs results...")
        try:
            from horse.reconcile import copy_actual_results, reconcile_predictions
            copied = copy_actual_results()
            logger.info(f"  Actual results copied to predictions.duckdb: {copied}")
            summary = reconcile_predictions(days_back=30)
            tp = summary.get("top_pick", {})
            if tp.get("total", 0) > 0:
                logger.info(
                    f"  Reconciliation: {tp['wins']}/{tp['total']} top-pick wins "
                    f"({tp['win_rate']*100:.1f}%), "
                    f"ROI: {summary.get('roi', {}).get('roi_pct', 0):+.1f}%"
                )
            else:
                logger.info(f"  Reconciliation: {summary.get('message', 'No matched data yet')}")
        except Exception as e:
            logger.warning(f"  Reconciliation skipped: {e}")

        logger.info("PHASE 0 complete — data refreshed")
    else:
        logger.info("PHASE 0 SKIPPED (--historical-only)")

    logger.info("-" * 60)

    # Check for previous model
    prev_path = MODEL_DIR / "win_model" / "meta.json"
    prev = None
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        ens = prev.get("metrics", {}).get("ensemble", {})
        logger.info(
            f"Previous: AUC={ens.get('auc')}, "
            f"features={len(prev.get('feature_names', []))}"
        )

    # Connect and show stats
    con = get_connection(read_only=True)
    stats = get_db_stats(con)
    logger.info(f"DB: {stats.get('results', 0):,} results, "
                f"{stats.get('horse_form', 0):,} form entries, "
                f"{stats.get('races', 0):,} races")

    if stats.get("results", 0) < 100:
        logger.error("Not enough data. Run backfill first (BACKFILL_HORSE.bat)")
        con.close()
        return

    # Build features
    mode = "FULL REBUILD" if args.full_rebuild else "INCREMENTAL"
    logger.info(f"Building feature matrix ({mode})...")
    t0 = time.time()
    df = build_training_dataset(con, full_rebuild=args.full_rebuild)
    con.close()

    if df.empty:
        logger.error("Feature matrix is empty. Check data.")
        return

    feat_cols = get_feature_columns(df)
    logger.info(
        f"Feature matrix: {len(df):,} rows, {len(feat_cols)} features "
        f"({time.time() - t0:.0f}s)"
    )

    # Verify requested features are present
    if tracked_extra:
        present = [f for f in tracked_extra if f in feat_cols]
        missing = [f for f in tracked_extra if f not in feat_cols]
        logger.info(f"Requested features present: {len(present)}/{len(tracked_extra)}")
        if missing:
            logger.warning(f"Requested features NOT found in matrix: {missing}")

    # Filter valid positions
    df = df[df["position"].notna() & (df["position"] > 0)]
    logger.info(f"Valid rows (pos>0): {len(df):,}")

    if len(df) < 100:
        logger.error("Not enough valid rows for training.")
        return

    # Leakage audit
    logger.info("LEAKAGE AUDIT...")
    flagged = audit_leakage(df, target="win")
    if flagged:
        logger.warning(f"Dropping {len(flagged)} leakage features")
        df = df.drop(columns=[c for c in flagged if c in df.columns], errors="ignore")

    # Purged walk-forward CV
    logger.info("PURGED WALK-FORWARD CV...")
    cv_result = purged_walk_forward_cv(df, target="win", n_folds=5, gap_days=7)
    if "avg_auc" in cv_result:
        logger.info(f"WF-CV: AUC={cv_result['avg_auc']}, Brier={cv_result['avg_brier']}")

    # Optuna tuning (optional)
    if args.tune:
        logger.info("OPTUNA HYPERPARAMETER TUNING...")
        n_trials = 30
        try:
            best_params = run_optuna_tuning(df, target="win", n_trials=n_trials)
            logger.info(f"Optuna best LGB: {best_params.get('lgb', {})}")
            logger.info(f"Optuna best XGB: {best_params.get('xgb', {})}")
        except Exception as e:
            logger.warning(f"Optuna tuning failed: {e}")

    # Load hard examples from previous cycle
    hard_ids = load_hard_examples()
    if hard_ids:
        logger.info(f"Loaded {len(hard_ids)} hard examples from previous cycle")

    # Train models for each requested target
    results = {}

    if "WIN" in requested_targets:
        logger.info("Training WIN model...")
        t0 = time.time()
        win = train_model(df, target="win", hard_example_ids=hard_ids,
                          models_to_train=requested_models)
        logger.info(f"WIN done in {time.time() - t0:.0f}s")
        if args.save_models:
            save_model(win)
        results["win"] = win

    if "PLACE" in requested_targets:
        logger.info("Training PLACE model...")
        t0 = time.time()
        place = train_model(df, target="place", hard_example_ids=hard_ids,
                            models_to_train=requested_models)
        logger.info(f"PLACE done in {time.time() - t0:.0f}s")
        if args.save_models:
            save_model(place)
        results["place"] = place

    # Summary
    logger.info("=" * 60)
    logger.info("  RESULTS")
    logger.info("=" * 60)
    for name, res in results.items():
        m = res["metrics"]
        if "ensemble" in m:
            ens = m["ensemble"]
            logger.info(
                f"{name.upper()}: AUC={ens['auc']}, Brier={ens['brier']}, "
                f"Acc={ens['accuracy']}"
            )

    # Feature importance logging
    if args.log_feature_importances and "win" in results:
        win = results["win"]
        logger.info("TOP 30 WIN FEATURES:")
        imp_items = list(win["feature_importance"].items())
        for i, (f, v) in enumerate(imp_items[:30]):
            logger.info(f"  {i + 1}. {f}: {v:.1f}")

        tracked_features = [
            "form_win_rate", "dist_match_win_rate", "going_match_win_rate",
            "course_match_win_rate", "type_match_win_rate", "class_match_win_rate",
            "form_x_freshness", "form_x_distance", "form_x_going",
            "course_x_distance", "class_x_form", "finish_strength",
            "relative_speed_z", "relative_speed_x_form", "relative_speed_x_distance",
            "jockey_win_rate_365d", "trainer_win_rate_365d",
            "front_runner_count", "pace_pressure", "expected_pace",
            "field_rating_mean", "field_rating_std", "rating_vs_field_mean",
            "market_entropy",
            "trainer_14day_win_rate", "trainer_14day_runs", "trainer_30day_roi",
            "lead_probability", "finish_speed_ratio",
        ] + tracked_extra

        logger.info("TRACKED FEATURE RANKINGS:")
        imp_ranked = {feat: rank + 1 for rank, (feat, _) in enumerate(
            sorted(imp_items, key=lambda x: -x[1])
        )}
        seen = set()
        for tf in tracked_features:
            if tf in seen:
                continue
            seen.add(tf)
            rank = imp_ranked.get(tf, "N/A")
            logger.info(f"  {tf}: rank #{rank}")

        if prev and "ensemble" in prev.get("metrics", {}):
            old_auc = prev["metrics"]["ensemble"].get("auc")
            new_auc = win["metrics"].get("ensemble", {}).get("auc")
            if old_auc and new_auc:
                logger.info(f"AUC: {old_auc} -> {new_auc}")

        logger.info(f"Features used: {len(win['feature_names'])}")

    # ── Save extended outputs ──────────────────────────────────
    if args.save_models and "win" in results:
        import csv
        win = results["win"]
        output_dir = MODEL_DIR / "model_15yr_decay365"
        output_dir.mkdir(parents=True, exist_ok=True)

        imp_items = list(win["feature_importance"].items())
        imp_path = output_dir / "feature_importance.csv"
        with open(imp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "importance", "rank"])
            for rank, (feat, val) in enumerate(
                sorted(imp_items, key=lambda x: -x[1]), 1
            ):
                writer.writerow([feat, f"{val:.2f}", rank])
        logger.info(f"Feature importance saved: {imp_path}")

        # Calibration report
        try:
            test_preds = win.get("test_predictions")
            test_actuals = win.get("test_actuals")
            if test_preds is not None and test_actuals is not None:
                import numpy as np
                cal_bins = []
                for lo in range(0, 100, 10):
                    hi = lo + 10
                    mask = (test_preds >= lo / 100) & (test_preds < hi / 100)
                    n = int(mask.sum())
                    if n > 0:
                        predicted = float(test_preds[mask].mean())
                        actual = float(test_actuals[mask].mean())
                    else:
                        predicted = (lo + hi) / 200
                        actual = 0.0
                    cal_bins.append({
                        "bin_low": lo / 100,
                        "bin_high": hi / 100,
                        "predicted": round(predicted, 4),
                        "actual": round(actual, 4),
                        "n": n,
                    })
                cal_path = output_dir / "calibration_report.json"
                with open(cal_path, "w", encoding="utf-8") as f:
                    json.dump(cal_bins, f, indent=2)
                logger.info(f"Calibration report saved: {cal_path}")

                try:
                    import pandas as pd
                    test_df_out = pd.DataFrame({
                        "predicted_win_prob": test_preds,
                        "actual_win": test_actuals,
                    })
                    pred_parquet = output_dir / "test_predictions.parquet"
                    test_df_out.to_parquet(pred_parquet, index=False)
                    logger.info(f"Test predictions saved: {pred_parquet}")
                except Exception as e:
                    logger.warning(f"Could not save test predictions parquet: {e}")
        except Exception as e:
            logger.warning(f"Calibration report skipped: {e}")

    # Log model versions to predictions.duckdb
    if args.save_models and results:
        try:
            from horse.prediction_db import get_pred_connection
            pred_con = get_pred_connection()
            try:
                for target_name, result in results.items():
                    ens = result["metrics"].get("ensemble", {})
                    mid = pred_con.execute(
                        "SELECT nextval('seq_model_id')"
                    ).fetchone()[0]
                    pred_con.execute("""
                        INSERT INTO model_versions
                            (id, model_version, target, trained_at,
                             auc, brier, logloss, accuracy,
                             feature_count, train_size, test_size)
                        VALUES (?, 'v1', ?, current_date, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (model_version, target) DO UPDATE SET
                            trained_at = excluded.trained_at,
                            auc = excluded.auc,
                            brier = excluded.brier,
                            logloss = excluded.logloss,
                            accuracy = excluded.accuracy,
                            feature_count = excluded.feature_count,
                            train_size = excluded.train_size,
                            test_size = excluded.test_size
                    """, [
                        mid, target_name,
                        ens.get("auc"), ens.get("brier"),
                        ens.get("logloss"), ens.get("accuracy"),
                        len(result["feature_names"]),
                        result["train_size"], result["test_size"],
                    ])
                logger.info("Model versions logged to predictions.duckdb")
            finally:
                pred_con.close()
        except Exception as e:
            logger.warning(f"Could not log model version: {e}")

    # Save monitoring snapshot
    if "win" in results:
        try:
            from horse.online import save_monitoring_snapshot, detect_drift
            win = results["win"]
            win_ens = win["metrics"].get("ensemble", {})
            snapshot = {
                "auc": win_ens.get("auc"),
                "brier": win_ens.get("brier"),
                "logloss": win_ens.get("logloss"),
                "accuracy": win_ens.get("accuracy"),
                "feature_count": len(win["feature_names"]),
                "train_size": win["train_size"],
                "test_size": win["test_size"],
                "weights": win_ens.get("weights", {}),
            }
            save_monitoring_snapshot(snapshot)
            logger.info("Monitoring snapshot saved")

            if prev and "ensemble" in prev.get("metrics", {}):
                drift = detect_drift(
                    {"rolling_auc": win_ens.get("auc"), "rolling_brier": win_ens.get("brier")},
                    prev["metrics"]["ensemble"],
                )
                if drift.get("needs_retrain"):
                    logger.warning("DRIFT DETECTED - model performance degraded")
                else:
                    logger.info("No drift detected - model stable")
        except Exception as e:
            logger.warning(f"Monitoring snapshot failed: {e}")

    # ── Precompute predictions if requested ────────────────────
    if args.precompute_predictions:
        logger.info("-" * 60)
        logger.info("PRECOMPUTING PREDICTIONS...")
        try:
            from horse.precompute import precompute_for_date
            today = str(date.today())
            tomorrow = str(date.today() + timedelta(days=1))
            logger.info(f"Precomputing for {today}...")
            precompute_for_date(today)
            logger.info(f"Precomputing for {tomorrow}...")
            precompute_for_date(tomorrow)
            logger.info("Precompute complete")
        except Exception as e:
            logger.error(f"Precompute failed: {e}")

    logger.info("DONE. Restart horse API to use new models.")


if __name__ == "__main__":
    main()
