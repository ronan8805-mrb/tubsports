"""
ML model training for UK/IRE Horse Racing predictions.
Completely separate from greyhound -- no shared imports.

Architecture (RTX 5090 Accelerated):
  - LightGBM classifier  (CPU -- GPU hangs on Windows/Blackwell)
  - XGBoost classifier    (GPU via CUDA)
  - Ensemble: weighted average of predicted probabilities
  - Optuna hyperparameter tuning (optional, for XGBoost GPU)
  - Leakage audit + time-based CV + calibration + hard-example boosting

Two prediction targets:
  1. WIN model    -- P(position == 1)  (binary classification)
  2. PLACE model  -- P(position <= 3)  (binary classification)

Models saved to horse/data/models/ as native format files.
"""

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import DATA_DIR, MAX_DAYS_LOOKBACK
from .features import get_feature_columns, ID_COLS, LEAKAGE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
TIME_DECAY_HALF_LIFE_DAYS = 120
HARD_EXAMPLE_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> bool:
    """Check if CUDA GPU is available for XGBoost training."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"GPU detected: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    return False


_GPU_AVAILABLE = _detect_gpu()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_xy(
    df: pd.DataFrame, target: str = "win"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and binary target y."""
    valid = df.dropna(subset=[TARGET_COL]).copy()
    valid = valid[valid[TARGET_COL] > 0]

    if target == "win":
        y = (valid[TARGET_COL] == 1).astype(int)
    elif target == "place":
        y = (valid[TARGET_COL] <= 3).astype(int)
    else:
        raise ValueError(f"Unknown target: {target}")

    feature_cols = get_feature_columns(valid)
    X = valid[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    logger.info(
        f"Prepared {target} target: {len(X):,} samples, "
        f"{y.sum():,} positives ({100 * y.mean():.1f}%), "
        f"{len(feature_cols)} features"
    )
    return X, y


def _temporal_train_test_split(
    df: pd.DataFrame, test_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date: last test_days go to test, rest to train."""
    dates = sorted(df["meeting_date"].dropna().unique())
    if len(dates) == 0:
        return df, pd.DataFrame()

    max_date = dates[-1]
    min_date = dates[0]
    date_range = (max_date - min_date).days

    if date_range < test_days * 2:
        split_idx = max(1, int(len(dates) * 0.8))
        cutoff = dates[split_idx - 1]
        logger.info(f"Short data range ({date_range}d). 80/20 split at {cutoff}")
    else:
        cutoff = max_date - timedelta(days=test_days)

    train = df[df["meeting_date"] <= cutoff]
    test = df[df["meeting_date"] > cutoff]

    if len(train) == 0:
        logger.warning("Train set empty. Using all data, no test set.")
        return df, pd.DataFrame()

    logger.info(
        f"Temporal split: train={len(train):,} (to {cutoff}), "
        f"test={len(test):,} (after {cutoff})"
    )
    return train, test


def _compute_sample_weights(
    df: pd.DataFrame,
    half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS,
    hard_example_ids: Optional[set] = None,
) -> np.ndarray:
    """Time-decay weighting with hard-example boosting."""
    if "meeting_date" not in df.columns:
        return np.ones(len(df))

    today = date.today()
    dates = pd.to_datetime(df["meeting_date"]).dt.date
    age_days = np.array(
        [(today - d).days if d is not None else 365 for d in dates],
        dtype=float,
    )

    decay = np.log(2) / max(half_life_days, 1)
    weights = np.exp(-age_days * decay)
    weights = np.clip(weights, 0.01, 1.0)

    if hard_example_ids and "race_id" in df.columns:
        for i, rid in enumerate(df["race_id"].values):
            if rid in hard_example_ids:
                weights[i] *= HARD_EXAMPLE_WEIGHT

    logger.info(
        f"Sample weights: mean={weights.mean():.3f}, "
        f"min={weights.min():.3f}, max={weights.max():.3f}"
    )
    return weights


# ---------------------------------------------------------------------------
# Leakage Audit
# ---------------------------------------------------------------------------

def audit_leakage(features_df: pd.DataFrame, target: str = "win") -> List[str]:
    """Detect features that leak future/current race info."""
    flagged = []
    X, y = _prepare_xy(features_df, target)

    for col in X.columns:
        if col in LEAKAGE_COLS:
            flagged.append(col)
            logger.warning(f"LEAKAGE: {col} in LEAKAGE_COLS")

    for col in X.columns:
        if col in flagged:
            continue
        valid_mask = X[col].notna()
        if valid_mask.sum() < 100:
            continue
        corr = X.loc[valid_mask, col].corr(y[valid_mask].astype(float))
        if abs(corr) > 0.5:
            flagged.append(col)
            logger.warning(f"LEAKAGE: {col} corr={corr:.3f}")

    post_keywords = ["result_", "actual_", "final_", "outcome_"]
    for col in X.columns:
        if col in flagged:
            continue
        if any(kw in col.lower() for kw in post_keywords):
            flagged.append(col)
            logger.warning(f"LEAKAGE: {col} post-race keyword")

    if not flagged:
        logger.info("LEAKAGE AUDIT: CLEAN")
    else:
        logger.warning(f"LEAKAGE AUDIT: {len(flagged)} flagged: {flagged}")
    return flagged


# ---------------------------------------------------------------------------
# Time-based cross-validation
# ---------------------------------------------------------------------------

def time_based_cv(
    features_df: pd.DataFrame,
    target: str = "win",
    n_folds: int = 3,
    fold_gap_days: int = 7,
) -> Dict[str, Any]:
    """Temporal CV with fold metrics and feature stability analysis."""
    if "meeting_date" not in features_df.columns:
        return {"error": "No meeting_date column"}

    dates = sorted(features_df["meeting_date"].dropna().unique())
    if len(dates) < 2:
        return {"error": "Not enough dates for CV"}

    total_days = (dates[-1] - dates[0]).days
    fold_days = total_days // (n_folds + 1)

    fold_metrics = []
    fold_importances = []

    for i in range(n_folds):
        test_start = dates[0] + timedelta(days=fold_days * (i + 1))
        test_end = test_start + timedelta(days=fold_days)
        train_end = test_start - timedelta(days=fold_gap_days)

        train_df = features_df[features_df["meeting_date"] <= train_end]
        test_df = features_df[
            (features_df["meeting_date"] >= test_start)
            & (features_df["meeting_date"] <= test_end)
        ]

        if len(train_df) < 500 or len(test_df) < 50:
            logger.info(f"  CV fold {i + 1}: skipped (too small)")
            continue

        X_train, y_train = _prepare_xy(train_df, target)
        X_test, y_test = _prepare_xy(test_df, target)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        feature_names = list(X_train.columns)
        lgb_params = _get_lgb_params()
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(
            lgb_params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        probs = model.predict(X_test)

        try:
            auc = roc_auc_score(y_test, probs)
            ll = log_loss(y_test, probs)
            brier = brier_score_loss(y_test, probs)
        except ValueError:
            continue

        fold_metrics.append({
            "fold": i + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "auc": round(auc, 4),
            "logloss": round(ll, 4),
            "brier": round(brier, 4),
        })

        imp = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
        fold_importances.append(imp)
        logger.info(
            f"  CV fold {i + 1}: AUC={auc:.4f}, Brier={brier:.4f}"
        )

    if not fold_metrics:
        return {"error": "No valid CV folds"}

    top20_counts: Dict[str, int] = {}
    for imp in fold_importances:
        top20 = sorted(imp.items(), key=lambda x: -x[1])[:20]
        for feat, _ in top20:
            top20_counts[feat] = top20_counts.get(feat, 0) + 1

    stable = [f for f, c in top20_counts.items() if c >= max(1, len(fold_metrics) - 1)]
    avg_auc = np.mean([m["auc"] for m in fold_metrics])
    avg_brier = np.mean([m["brier"] for m in fold_metrics])

    logger.info(f"CV Summary: avg_AUC={avg_auc:.4f}, avg_Brier={avg_brier:.4f}")
    return {
        "fold_metrics": fold_metrics,
        "avg_auc": round(avg_auc, 4),
        "avg_brier": round(avg_brier, 4),
        "stable_features": stable,
    }


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------

def _get_lgb_params() -> dict:
    """LightGBM params for horse racing (CPU -- GPU crashes on Windows/RTX 5090)."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.3,
        "lambda_l2": 0.5,
        "path_smooth": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }


def _get_xgb_params() -> dict:
    """XGBoost params for horse racing (GPU via CUDA on RTX 5090)."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 50,
        "reg_alpha": 0.3,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }
    if _GPU_AVAILABLE:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
        logger.info("XGBoost: GPU enabled (CUDA)")
    else:
        params["tree_method"] = "hist"
    return params


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning (XGBoost GPU)
# ---------------------------------------------------------------------------

def optuna_tune_xgb(
    features_df: pd.DataFrame,
    target: str = "win",
    n_trials: int = 50,
    test_days: int = 30,
) -> dict:
    """Run Optuna to find best XGBoost hyperparameters.
    Uses GPU acceleration for fast trial evaluation."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return {}

    train_df, test_df = _temporal_train_test_split(features_df, test_days)
    X_train, y_train = _prepare_xy(train_df, target)
    X_test, y_test = _prepare_xy(test_df, target)

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Not enough data for Optuna tuning")
        return {}

    feature_names = list(X_train.columns)
    weights = _compute_sample_weights(train_df.loc[X_train.index])

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "verbosity": 0,
            "random_state": 42,
        }
        if _GPU_AVAILABLE:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "hist"

        pos = y_train.sum()
        neg = len(y_train) - pos
        params["scale_pos_weight"] = neg / max(pos, 1)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights,
                             feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dtest, "test")],
            early_stopping_rounds=50, verbose_eval=0,
        )
        preds = model.predict(dtest)
        return brier_score_loss(y_test, preds)

    study = optuna.create_study(direction="minimize")
    logger.info(f"Starting Optuna: {n_trials} trials ({target} target)")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    elapsed = time.time() - t0

    logger.info(
        f"Optuna done in {elapsed:.0f}s. Best Brier: {study.best_value:.4f}"
    )
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    features_df: pd.DataFrame,
    target: str = "win",
    n_rounds: int = 800,
    early_stopping: int = 80,
    test_days: int = 14,
    hard_example_ids: Optional[set] = None,
    xgb_overrides: Optional[dict] = None,
) -> Dict[str, Any]:
    """Train LightGBM + XGBoost ensemble for horse racing."""
    logger.info(f"=== Training {target.upper()} model ===")

    train_df, test_df = _temporal_train_test_split(features_df, test_days)

    if len(train_df) < 50:
        raise ValueError(f"Only {len(train_df)} training rows. Need more data.")

    X_train, y_train = _prepare_xy(train_df, target)
    X_test, y_test = (
        _prepare_xy(test_df, target) if len(test_df) > 0
        else (pd.DataFrame(), pd.Series(dtype=int))
    )

    feature_names = list(X_train.columns)
    if len(X_train) == 0:
        raise ValueError("Empty training set after filtering")

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = neg / max(pos, 1)
    logger.info(f"Pos weight: {pos_weight:.2f} ({pos} pos, {neg} neg)")

    weights = _compute_sample_weights(
        train_df.loc[X_train.index],
        hard_example_ids=hard_example_ids,
    )
    if len(weights) != len(X_train):
        weights = np.ones(len(X_train))

    # ---- LightGBM (CPU) ----
    lgb_params = _get_lgb_params()
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights)
    lgb_valid = lgb.Dataset(X_test, label=y_test) if len(X_test) > 0 else None

    callbacks = [lgb.log_evaluation(period=100)]
    if lgb_valid is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))

    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=n_rounds,
        valid_sets=[lgb_valid] if lgb_valid else None,
        valid_names=["test"] if lgb_valid else None,
        callbacks=callbacks,
    )
    logger.info(f"LightGBM: {lgb_model.num_trees()} trees (CPU)")

    # ---- XGBoost (GPU) ----
    xgb_params = _get_xgb_params()
    xgb_params["scale_pos_weight"] = pos_weight
    if xgb_overrides:
        xgb_params.update(xgb_overrides)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights,
                         feature_names=feature_names)
    dtest = (
        xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        if len(X_test) > 0 else None
    )

    evals = [(dtrain, "train")]
    if dtest is not None:
        evals.append((dtest, "test"))

    try:
        xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping if dtest else None,
            verbose_eval=100,
        )
    except (OSError, Exception) as e:
        if xgb_params.get("device") == "cuda":
            logger.warning(f"XGBoost GPU failed ({e}), falling back to CPU")
            xgb_params.pop("device", None)
            xgb_params["tree_method"] = "hist"
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights,
                                 feature_names=feature_names)
            dtest = (
                xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
                if len(X_test) > 0 else None
            )
            evals = [(dtrain, "train")]
            if dtest is not None:
                evals.append((dtest, "test"))
            xgb_model = xgb.train(
                xgb_params, dtrain, num_boost_round=n_rounds,
                evals=evals,
                early_stopping_rounds=early_stopping if dtest else None,
                verbose_eval=100,
            )
        else:
            raise
    logger.info(f"XGBoost: {xgb_model.num_boosted_rounds()} rounds")

    # ---- Evaluation ----
    metrics = {}
    if len(X_test) > 0 and len(y_test) > 0:
        lgb_probs = lgb_model.predict(X_test)
        xgb_probs = xgb_model.predict(
            xgb.DMatrix(X_test, feature_names=feature_names)
        )
        ens_probs = 0.5 * lgb_probs + 0.5 * xgb_probs

        for name, probs in [("lgb", lgb_probs), ("xgb", xgb_probs),
                            ("ensemble", ens_probs)]:
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = np.nan
            try:
                ll = log_loss(y_test, probs)
            except ValueError:
                ll = np.nan
            try:
                brier = brier_score_loss(y_test, probs)
            except ValueError:
                brier = np.nan

            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)

            metrics[name] = {
                "auc": round(float(auc), 4) if not np.isnan(auc) else None,
                "logloss": round(float(ll), 4) if not np.isnan(ll) else None,
                "brier": round(float(brier), 4) if not np.isnan(brier) else None,
                "accuracy": round(float(acc), 4),
                "samples": len(y_test),
            }
            logger.info(
                f"  {name:>10}: AUC={metrics[name]['auc']}, "
                f"Brier={metrics[name]['brier']}, Acc={metrics[name]['accuracy']}"
            )

    # ---- Feature importance ----
    lgb_imp = dict(zip(feature_names, lgb_model.feature_importance(importance_type="gain")))
    xgb_imp = xgb_model.get_score(importance_type="gain")
    all_feats = set(list(lgb_imp.keys()) + list(xgb_imp.keys()))
    combined_imp = {}
    for f in all_feats:
        v1 = lgb_imp.get(f, 0)
        v2 = xgb_imp.get(f, 0)
        combined_imp[f] = round(float(v1 + v2) / 2, 2)
    combined_imp = dict(sorted(combined_imp.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top 10 features:")
    for i, (feat, imp) in enumerate(list(combined_imp.items())[:10]):
        logger.info(f"  {i + 1}. {feat}: {imp:.2f}")

    # ---- Hard examples ----
    hard_examples = set()
    try:
        hard_examples = _identify_hard_examples(
            lgb_model, xgb_model, feature_names, train_df, target,
        )
    except Exception as e:
        logger.debug(f"Hard example ID skipped: {e}")

    return {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "feature_names": feature_names,
        "metrics": metrics,
        "feature_importance": combined_imp,
        "target": target,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "hard_examples": hard_examples,
    }


def _identify_hard_examples(
    lgb_model, xgb_model, feature_names,
    train_df: pd.DataFrame, target: str,
    threshold_high: float = 0.40, threshold_low: float = 0.08,
) -> set:
    """Find races the model got very wrong for boosting next cycle."""
    X, y = _prepare_xy(train_df, target)
    if len(X) == 0 or "race_id" not in train_df.columns:
        return set()

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_names))
    probs = 0.5 * lgb_probs + 0.5 * xgb_probs

    valid = train_df.dropna(subset=[TARGET_COL]).copy()
    valid = valid[valid[TARGET_COL] > 0]
    race_ids = valid["race_id"].values[:len(probs)]

    hard_ids = set()
    for i in range(len(probs)):
        actual_pos = y.iloc[i] == 1
        if probs[i] > threshold_high and not actual_pos:
            hard_ids.add(int(race_ids[i]))
        elif probs[i] < threshold_low and actual_pos:
            hard_ids.add(int(race_ids[i]))

    logger.info(f"Hard examples: {len(hard_ids)} races for {target}")
    return hard_ids


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(result: Dict[str, Any], suffix: str = "") -> Path:
    """Save model artifacts (LGB + XGB + metadata)."""
    target = result["target"]
    model_name = f"{target}_model{suffix}"
    model_path = MODEL_DIR / model_name
    model_path.mkdir(parents=True, exist_ok=True)

    lgb_path = model_path / "lgb.txt"
    result["lgb_model"].save_model(str(lgb_path))

    xgb_path = model_path / "xgb.json"
    result["xgb_model"].save_model(str(xgb_path))

    meta = {
        "target": result["target"],
        "feature_names": result["feature_names"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "trained_at": str(date.today()),
    }
    meta_path = model_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info(f"Model saved: {model_path}")
    return model_path


def load_model(target: str = "win", suffix: str = "") -> Dict[str, Any]:
    """Load trained model from disk."""
    model_name = f"{target}_model{suffix}"
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    lgb_model = lgb.Booster(model_file=str(model_path / "lgb.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_path / "xgb.json"))

    meta = json.loads((model_path / "meta.json").read_text(encoding="utf-8"))

    logger.info(f"Loaded {target} model from {model_path}")
    return {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "feature_names": meta["feature_names"],
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_race(
    features_df: pd.DataFrame,
    target: str = "win",
) -> pd.DataFrame:
    """Generate predictions for runners in a race.
    Returns DataFrame with horse_name, win_prob/place_prob, rank."""
    model_data = load_model(target)
    lgb_model = model_data["lgb_model"]
    xgb_model = model_data["xgb_model"]
    feature_names = model_data["feature_names"]

    available = [c for c in feature_names if c in features_df.columns]
    missing = [c for c in feature_names if c not in features_df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features, filling with NaN: {missing[:5]}...")

    X = features_df.reindex(columns=feature_names).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_names))
    probs = 0.5 * lgb_probs + 0.5 * xgb_probs

    prob_col = "win_prob" if target == "win" else "place_prob"
    rank_col = "win_rank" if target == "win" else "place_rank"

    result = features_df[["horse_name"]].copy() if "horse_name" in features_df.columns else pd.DataFrame()
    result[prob_col] = probs
    result[rank_col] = result[prob_col].rank(ascending=False, method="min").astype(int)
    result = result.sort_values(prob_col, ascending=False)

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def train_all(
    features_df: pd.DataFrame,
    test_days: int = 14,
    hard_example_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Train both WIN and PLACE models. Save to disk."""
    results = {}
    all_hard = set()

    for target in ["win", "place"]:
        result = train_model(
            features_df, target=target, test_days=test_days,
            hard_example_ids=hard_example_ids,
        )
        path = save_model(result)
        all_hard.update(result.get("hard_examples", set()))

        results[target] = {
            "path": str(path),
            "metrics": result["metrics"],
            "feature_importance": dict(list(result["feature_importance"].items())[:15]),
            "train_size": result["train_size"],
            "test_size": result["test_size"],
        }

    _save_hard_examples(all_hard)
    return results


# ---------------------------------------------------------------------------
# Hard example persistence
# ---------------------------------------------------------------------------

_HARD_EXAMPLES_PATH = MODEL_DIR / "hard_examples.json"


def _save_hard_examples(hard_ids: set) -> None:
    try:
        _HARD_EXAMPLES_PATH.write_text(
            json.dumps(list(hard_ids), default=int), encoding="utf-8"
        )
        logger.info(f"Saved {len(hard_ids)} hard examples")
    except Exception as e:
        logger.debug(f"Could not save hard examples: {e}")


def load_hard_examples() -> set:
    try:
        if _HARD_EXAMPLES_PATH.exists():
            data = json.loads(_HARD_EXAMPLES_PATH.read_text(encoding="utf-8"))
            return set(int(x) for x in data)
    except Exception:
        pass
    return set()
